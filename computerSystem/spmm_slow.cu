// Correct-but-inefficient FP32 SpMM: C = A * B.
// Build: nvcc -O3 -std=c++17 -lineinfo spmm_slow.cu -o spmm_slow
// Run:   ./spmm_slow [M K N REPEATS]       (defaults: 2048 4096 128 3)
// Small correctness run: ./spmm_slow 37 53 19 1
//
// Deliberate, real inefficiencies (no sleeps, dummy work, or debug builds):
//   S1: reconstruct CSR by scanning all COO entries for every row, every repeat.
//   S2: repack row-major B into column-major B with strided source reads.
//   S3: allocate/free device arrays every repeat; use pageable host memory.
//   S4: transfer B and C one column at a time with blocking cudaMemcpy.
//   S5: launch one kernel per output column and synchronize after every launch.
//   S6: one thread per sparse row, one warp per block, skewed row lengths,
//       repeated CSR traversal, and irregular accesses across warp lanes.
//   S7: transpose the column-major GPU result back on the CPU.
//
// The independent FP64 reference is computed from the ORIGINAL COO input.
// Input/reference generation and validation are outside iteration_ms.
// kernel_and_sync_ms includes launch and synchronization, not pure GPU time.
// Actual bottlenecks and speedups depend on the target GPU and problem size.

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;

static double milliseconds(Clock::time_point begin) {
    return std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
}

static void check_cuda(cudaError_t status, const char* expression, int line) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(expression) + " at line " +
                                 std::to_string(line) + ": " + cudaGetErrorString(status));
    }
}
#define CUDA_CHECK(expression) check_cuda((expression), #expression, __LINE__)

template <class T>
struct DeviceArray {
    T* data = nullptr;
    DeviceArray() = default;
    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;
    ~DeviceArray() { if (data) cudaFree(data); }
    void allocate(std::size_t count) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(T)));
    }
    void release() {
        if (data) {
            T* old = data;
            data = nullptr;
            CUDA_CHECK(cudaFree(old));
        }
    }
};

struct Entry { int row, col; float value; };
struct Csr {
    std::vector<int> row_ptr, col;
    std::vector<float> value;
};
struct Shape { int m = 2048, k = 4096, n = 128, repeats = 3; };

static int positive_int(const char* text) {
    const std::string value(text);
    std::size_t consumed = 0;
    const long long number = std::stoll(value, &consumed);
    if (consumed != value.size() || number < 1 || number >= std::numeric_limits<int>::max())
        throw std::runtime_error("Arguments must be positive integers smaller than INT_MAX.");
    return static_cast<int>(number);
}

static Shape parse_shape(int argc, char** argv) {
    Shape s;
    if (argc != 1 && argc != 5)
        throw std::runtime_error("Usage: ./spmm_slow [M K N REPEATS]");
    if (argc == 5) {
        s.m = positive_int(argv[1]); s.k = positive_int(argv[2]);
        s.n = positive_int(argv[3]); s.repeats = positive_int(argv[4]);
    }
    // Bound demo memory use and guarantee 32-bit CSR offsets are representable.
    const std::uint64_t dense = (std::uint64_t(s.m) + s.k) * s.n;
    const int heavy_rows = std::max(1, s.m / 16);
    const std::uint64_t nnz = std::uint64_t(heavy_rows) * std::min(s.k, 512)
        + std::uint64_t(s.m - heavy_rows) * std::min(s.k, 8);
    if (dense > 32ULL * 1024 * 1024 || nnz > 16ULL * 1024 * 1024)
        throw std::runtime_error("Demo size limit: M*N+K*N <= 32M elements; nnz <= 16M.");
    return s;
}

static std::vector<Entry> make_coo(const Shape& s) {
    std::mt19937 rng(20261118u);
    std::vector<Entry> entries;
    const int heavy_rows = std::max(1, s.m / 16);
    for (int r = 0; r < s.m; ++r) {
        const int degree = std::min(s.k, r < heavy_rows ? 512 : 8);
        const int start = static_cast<int>(rng() % unsigned(s.k));
        int stride = 1 + static_cast<int>(rng() % unsigned(s.k));
        while (std::gcd(stride, s.k) != 1) stride = stride == s.k ? 1 : stride + 1;
        for (int j = 0; j < degree; ++j) {
            const int col = static_cast<int>((std::int64_t(start) + std::int64_t(j) * stride) % s.k);
            const int magnitude = 1 + static_cast<int>(rng() % 4u);
            const int sign = (rng() & 1u) ? 1 : -1;
            // Nonzero dyadic values keep this demonstration's numerical check clear.
            entries.push_back({r, col, sign * magnitude * 0.25f});
        }
    }
    return entries;
}

static std::vector<float> make_b(const Shape& s) {
    std::mt19937 rng(7919u);
    std::vector<float> b(std::size_t(s.k) * s.n);
    for (float& x : b) x = (static_cast<int>(rng() % 17u) - 8) * 0.125f;
    return b;
}

// S1: O(M * nnz) conversion, repeated even though A never changes.
static Csr build_csr_slow(const std::vector<Entry>& entries, int m) {
    Csr a;
    a.row_ptr.resize(std::size_t(m) + 1);
    for (int r = 0; r < m; ++r) {
        a.row_ptr[r] = static_cast<int>(a.col.size());
        for (const Entry& e : entries) {
            if (e.row == r) {
                a.col.push_back(e.col);
                a.value.push_back(e.value);
            }
        }
    }
    a.row_ptr[m] = static_cast<int>(a.col.size());
    return a;
}

// S2: CPU layout conversion; contiguous destination, strided source.
static std::vector<float> pack_b_slow(const std::vector<float>& b, int k, int n) {
    std::vector<float> packed(std::size_t(k) * n);
    for (int f = 0; f < n; ++f)
        for (int j = 0; j < k; ++j)
            packed[std::size_t(f) * k + j] = b[std::size_t(j) * n + f];
    return packed;
}

// Independent mathematical reference; no CSR or packed B is used here.
static std::vector<double> reference_coo(const std::vector<Entry>& entries,
                                          const std::vector<float>& b, const Shape& s) {
    std::vector<double> result(std::size_t(s.m) * s.n, 0.0);
    for (const Entry& e : entries)
        for (int f = 0; f < s.n; ++f)
            result[std::size_t(e.row) * s.n + f] +=
                double(e.value) * double(b[std::size_t(e.col) * s.n + f]);
    return result;
}

// S6: neighbouring lanes own different rows, not neighbouring feature columns.
// Each thread serially reduces a complete sparse row for ONE output column.
// Each C element has exactly one writer: no atomics or cross-block dependency.
__global__ void spmm_one_column_slow(int m, int k, int f,
                                     const int* row_ptr, const int* col,
                                     const float* value, const float* b_column_major,
                                     float* c_column_major) {
    const std::size_t r = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (r >= std::size_t(m)) return;
    const float* b_column = b_column_major + std::size_t(f) * k;
    float sum = 0.0f;
    for (int q = row_ptr[r]; q < row_ptr[r + 1]; ++q)
        sum += value[q] * b_column[col[q]];
    // This also writes zero for an empty row; no prior memset is required.
    c_column_major[std::size_t(f) * m + r] = sum;
}

// S7: CPU layout conversion; strided source, contiguous destination.
static void unpack_c_slow(const std::vector<float>& packed, std::vector<float>& c,
                          int m, int n) {
    for (int r = 0; r < m; ++r)
        for (int f = 0; f < n; ++f)
            c[std::size_t(r) * n + f] = packed[std::size_t(f) * m + r];
}

static double validate(const std::vector<float>& c, const std::vector<double>& reference) {
    double max_error = 0.0;
    for (std::size_t i = 0; i < c.size(); ++i) {
        const double error = std::abs(double(c[i]) - reference[i]);
        const double tolerance = 1e-5 + 1e-4 * std::abs(reference[i]);
        if (!std::isfinite(c[i]) || error > tolerance)
            throw std::runtime_error("Validation failed at output element " + std::to_string(i));
        max_error = std::max(max_error, error);
    }
    return max_error;
}

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--help") {
            std::cout << "Usage: ./spmm_slow [M K N REPEATS]\nDefaults: 2048 4096 128 3\n";
            return 0;
        }
        const Shape s = parse_shape(argc, argv);
        const auto entries = make_coo(s);
        const auto b = make_b(s);
        const auto reference = reference_coo(entries, b, s);

        CUDA_CHECK(cudaSetDevice(0));
        cudaDeviceProp device{};
        CUDA_CHECK(cudaGetDeviceProperties(&device, 0));
        CUDA_CHECK(cudaFree(nullptr)); // Initialize the runtime before timing iterations.
        std::cout << "GPU=" << device.name << " M=" << s.m << " K=" << s.k
                  << " N=" << s.n << " nnz=" << entries.size() << '\n';
        std::cout << "All times are host wall milliseconds. Input/reference and validation excluded.\n"
                     "kernel_and_sync_ms includes launch + kernel + synchronization.\n"
                     "The first iteration may include lazy module loading.\n";
        std::cout << std::fixed << std::setprecision(3);

        for (int iteration = 0; iteration < s.repeats; ++iteration) {
            const auto iteration_begin = Clock::now();
            auto phase_begin = Clock::now();
            Csr a = build_csr_slow(entries, s.m);
            const double csr_ms = milliseconds(phase_begin);

            phase_begin = Clock::now();
            const auto packed_b = pack_b_slow(b, s.k, s.n);
            const double pack_ms = milliseconds(phase_begin);

            // S3: ordinary std::vector host memory is pageable; all buffers recreated.
            phase_begin = Clock::now();
            std::vector<float> packed_c(std::size_t(s.m) * s.n);
            std::vector<float> c(std::size_t(s.m) * s.n);
            DeviceArray<int> d_row_ptr, d_col;
            DeviceArray<float> d_value, d_b, d_c;
            d_row_ptr.allocate(a.row_ptr.size()); d_col.allocate(a.col.size());
            d_value.allocate(a.value.size()); d_b.allocate(packed_b.size()); d_c.allocate(c.size());
            const double alloc_ms = milliseconds(phase_begin);

            phase_begin = Clock::now();
            CUDA_CHECK(cudaMemcpy(d_row_ptr.data, a.row_ptr.data(), a.row_ptr.size() * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_col.data, a.col.data(), a.col.size() * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_value.data, a.value.data(), a.value.size() * sizeof(float), cudaMemcpyHostToDevice));
            // S4: N small blocking transfers, instead of one transfer or persistent data.
            for (int f = 0; f < s.n; ++f)
                CUDA_CHECK(cudaMemcpy(d_b.data + std::size_t(f) * s.k,
                                      packed_b.data() + std::size_t(f) * s.k,
                                      std::size_t(s.k) * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaDeviceSynchronize());
            const double h2d_ms = milliseconds(phase_begin);

            phase_begin = Clock::now();
            const int threads = 32;
            const unsigned blocks = static_cast<unsigned>((std::size_t(s.m) + threads - 1) / threads);
            // S5: N launches and N device-wide synchronizations for a single SpMM.
            for (int f = 0; f < s.n; ++f) {
                spmm_one_column_slow<<<blocks, threads>>>(s.m, s.k, f, d_row_ptr.data,
                    d_col.data, d_value.data, d_b.data, d_c.data);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaDeviceSynchronize());
            }
            const double compute_ms = milliseconds(phase_begin);

            phase_begin = Clock::now();
            for (int f = 0; f < s.n; ++f)
                CUDA_CHECK(cudaMemcpy(packed_c.data() + std::size_t(f) * s.m,
                                      d_c.data + std::size_t(f) * s.m,
                                      std::size_t(s.m) * sizeof(float), cudaMemcpyDeviceToHost));
            const double d2h_ms = milliseconds(phase_begin);

            phase_begin = Clock::now();
            unpack_c_slow(packed_c, c, s.m, s.n);
            const double unpack_ms = milliseconds(phase_begin);

            phase_begin = Clock::now();
            d_row_ptr.release(); d_col.release(); d_value.release(); d_b.release(); d_c.release();
            const double free_ms = milliseconds(phase_begin);
            // Host vector destruction occurs after this boundary (not counted).
            const double iteration_ms = milliseconds(iteration_begin);
            const double max_error = validate(c, reference);
            std::cout << "iteration=" << iteration
                      << " csr_ms=" << csr_ms << " pack_b_ms=" << pack_ms
                      << " alloc_ms=" << alloc_ms << " h2d_ms=" << h2d_ms
                      << " kernel_and_sync_ms=" << compute_ms << " d2h_ms=" << d2h_ms
                      << " unpack_c_ms=" << unpack_ms << " free_ms=" << free_ms
                      << " iteration_ms=" << iteration_ms
                      << " max_abs_error=" << std::scientific << max_error << std::fixed
                      << " validation=PASS\n";
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "spmm_slow: " << error.what() << '\n';
        return 1;
    }
}