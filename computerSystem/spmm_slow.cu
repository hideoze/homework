// One deliberately inefficient C++/CUDA SpMM: C = A * B (FP32).
// Linux + CUDA, one execution path, no benchmark modes or comparison framework.
// Build: nvcc -O3 -std=c++17 -lineinfo spmm_slow.cu -o spmm_slow
// Run:   ./spmm_slow [M K N REPEATS]       (defaults: 2048 4096 128 3)
// Small correctness run: ./spmm_slow 37 53 19 1
//
// Deliberately retained problems, all in the actual data/compute path:
// P0: off-by-one table search in prefer_padded_layout(). An optimizer may turn
//     the entire function into "return true" by assuming no out-of-bounds UB.
//     N=19/32/64/128 returns before the bad access; other N has UNDEFINED
//     BEHAVIOR. This is a known teaching defect, NOT a safe input validator.
// P1: rebuild CSR in O(M*nnz) from shuffled COO on every iteration.
// P2: convert CSR to randomly scattered 128-byte linked records; serial next
//     dependencies, poor spatial locality and large metadata/padding overhead.
// P3: interleave long and short rows within warps; decode mixed representations
//     inside the innermost loop, repeatedly for every output feature.
// P4: recreate pageable, page-strided host B and touch fresh anonymous pages;
//     then read one feature across all rows before reading the next feature.
// P5: put each GPU B scalar in a 128-byte slot; initialize/copy the whole layout.
// P6: CPU-initialize fresh managed sparse nodes without prefetching, when the
//     device supports managed memory; otherwise use a normal device allocation.
// P7: allocate/free each iteration, copy B/C column by column, launch one kernel
//     per feature with 32 threads per block and synchronize after every launch.
// P8: convert the column-major result back on the CPU.
//
// For N=19/32/64/128 within the reported memory budget, all computation paths
// use initialized data, bounded indices and one writer per output element.
// The independent FP64 reference uses ORIGINAL COO and row-major B.
// Cache/TLB misses and managed-memory migration are measurement hypotheses,
// not guaranteed outcomes. Linux CPU page-fault counters are NOT GPU counters.
// No sleep, spin loop, -O0/-G build, or unused arithmetic is used to make it slow.

#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <new>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <system_error>
#include <type_traits>
#include <vector>

#ifndef __linux__
#error "This example uses Linux mmap/getrusage for the host paging path."
#endif
#include <sys/mman.h>
#include <sys/resource.h>
#include <unistd.h>

using Clock = std::chrono::steady_clock;
static double milliseconds(Clock::time_point begin) {
    return std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
}

static void check_cuda(cudaError_t status, const char* expression, int line) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(expression) + " at line " +
            std::to_string(line) + ": " + cudaGetErrorString(status));
}
#define CUDA_CHECK(expression) check_cuda((expression), #expression, __LINE__)

template <class T> struct GpuArray {
    T* data = nullptr;
    bool managed = false;
    GpuArray() = default;
    GpuArray(const GpuArray&) = delete;
    GpuArray& operator=(const GpuArray&) = delete;
    ~GpuArray() { if (data) cudaFree(data); }
    void allocate(std::size_t count, bool use_managed = false) {
        managed = use_managed;
        if (managed) {
            CUDA_CHECK(cudaMallocManaged(reinterpret_cast<void**>(&data), count * sizeof(T)));
            static_assert(std::is_trivially_destructible<T>::value, "POD arrays only");
            ::new (static_cast<void*>(data)) T[count];
        } else {
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(T)));
        }
    }
    void release() {
        if (data) {
            T* old = data;
            data = nullptr;
            CUDA_CHECK(cudaFree(old));
        }
    }
};

static std::size_t host_page_bytes() {
    const long page = ::sysconf(_SC_PAGESIZE);
    if (page <= 0 || page % sizeof(float) != 0)
        throw std::runtime_error("Unsupported host page size.");
    return static_cast<std::size_t>(page);
}

// P4: mmap + non-value-initializing placement array-new start float lifetimes
// without consuming first-touch beforehand. Every element read is written first.
struct HostFloatBuffer {
    float* data = nullptr;
    std::size_t count = 0;
    explicit HostFloatBuffer(std::size_t elements, bool base_pages = false) : count(elements) {
        void* memory = ::mmap(nullptr, count * sizeof(float), PROT_READ | PROT_WRITE,
                             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (memory == MAP_FAILED)
            throw std::system_error(errno, std::generic_category(), "mmap");
        if (base_pages && ::madvise(memory, count * sizeof(float), MADV_NOHUGEPAGE) != 0)
            std::cerr << "MADV_NOHUGEPAGE request failed; actual page policy may differ.\n";
        data = ::new (memory) float[count];
    }
    HostFloatBuffer(const HostFloatBuffer&) = delete;
    HostFloatBuffer& operator=(const HostFloatBuffer&) = delete;
    ~HostFloatBuffer() { if (data) ::munmap(data, count * sizeof(float)); }
    void release() {
        if (data) {
            if (::munmap(data, count * sizeof(float)) != 0)
                throw std::system_error(errno, std::generic_category(), "munmap");
            data = nullptr;
        }
    }
};

struct Faults { long minor, major; };
static Faults snapshot_faults() {
    rusage usage{};
    if (::getrusage(RUSAGE_SELF, &usage) != 0)
        throw std::system_error(errno, std::generic_category(), "getrusage");
    return {usage.ru_minflt, usage.ru_majflt};
}

struct Entry { int row, col; float value; };
struct Csr {
    std::vector<int> row_ptr, col;
    std::vector<float> value;
};

// P2: hot fields and cold metadata share an oversized, over-aligned AoS record.
struct alignas(128) SparseNode {
    int next, col;
    unsigned mode, payload;
    unsigned char cold_metadata[112];
};
static_assert(sizeof(SparseNode) == 128, "Expected a 128-byte sparse record");
static_assert(sizeof(float) == 4 && sizeof(unsigned) == 4, "FP32/uint32 required");
struct LinkedRows {
    std::vector<int> head;
    std::vector<SparseNode> nodes;
};
struct Shape {
    int m = 2048, k = 4096, n = 128, repeats = 3;
    std::size_t host_stride = 0, gpu_stride = 0;
};

// P0 -- INTENTIONAL UB TRAP, matching the course's table[4] example.
// Every DEFINED execution returns true. Reaching i==4 reads past the array.
// An optimizer may therefore remove the search and return true always.
// Default N=128 and small-run N=19 hit a valid element before that access.
// For other N, neither an apparently successful run nor validation proves safety.
// Correcting the bound to i<4 restores the intended layout-selection contract.
// Do not add volatile: the optimization itself is part of what is being studied.
static const int familiar_widths[4] = {19, 32, 64, 128};
__attribute__((noinline)) static bool prefer_padded_layout(int n) {
    for (int i = 0; i <= 4; ++i)
        if (familiar_widths[i] == n) return true;
    return false;
}

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
    // Budget arithmetic is well-defined. The deliberate defect is only P0.
    const std::uint64_t b_count = std::uint64_t(s.k) * s.n;
    const std::uint64_t c_count = std::uint64_t(s.m) * s.n;
    const std::uint64_t heavy = (std::uint64_t(s.m) + 15) / 16;
    const std::uint64_t nnz = heavy * std::min(s.k, 512) +
        (std::uint64_t(s.m) - heavy) * std::min(s.k, 8);
    if (b_count + c_count > 32ULL * 1024 * 1024 || nnz > 2ULL * 1024 * 1024)
        throw std::runtime_error("Demo budget: M*N+K*N <= 32M; nnz <= 2M.");
    const std::size_t page_floats = host_page_bytes() / sizeof(float);
    s.host_stride = ((std::size_t(s.n) + page_floats - 1) / page_floats) * page_floats;
    s.gpu_stride = prefer_padded_layout(s.n) ? 32 : 1;
    const std::uint64_t host_b_bytes = std::uint64_t(s.k) * s.host_stride * sizeof(float);
    const std::uint64_t gpu_b_bytes = b_count * s.gpu_stride * sizeof(float);
    // Include padding and both host/device copies, not just logical dimensions.
    const std::uint64_t estimated_peak = host_b_bytes + 2 * gpu_b_bytes +
        nnz * 280 + 4 * b_count + 20 * c_count + 16 * std::uint64_t(s.m) + 4096;
    if (host_b_bytes > 256ULL * 1024 * 1024 || gpu_b_bytes > 256ULL * 1024 * 1024 ||
        estimated_peak > 1024ULL * 1024 * 1024)
        throw std::runtime_error("Expanded-layout budget exceeded (256 MiB per B layout; ~1 GiB total).");
    return s;
}

static std::vector<Entry> make_coo(const Shape& s) {
    std::mt19937 rng(20261119u);
    std::vector<Entry> entries;
    for (int r = 0; r < s.m; ++r) {
        // P3: long rows occupy lanes 0 and 16; short rows in the same warp idle.
        const int degree = std::min(s.k, r % 16 == 0 ? 512 : 8);
        const int start = static_cast<int>(rng() % unsigned(s.k));
        int stride = 1 + static_cast<int>(rng() % unsigned(s.k));
        while (std::gcd(stride, s.k) != 1) stride = stride == s.k ? 1 : stride + 1;
        for (int j = 0; j < degree; ++j) {
            const int col = static_cast<int>((std::int64_t(start) + std::int64_t(j) * stride) % s.k);
            const int magnitude = 1 + static_cast<int>(rng() % 4u);
            const int sign = (rng() & 1u) ? 1 : -1;
            entries.push_back({r, col, sign * magnitude * 0.25f});
        }
    }
    std::shuffle(entries.begin(), entries.end(), rng);
    return entries;
}

static std::vector<float> make_b(const Shape& s) {
    std::mt19937 rng(7919u);
    std::vector<float> b(std::size_t(s.k) * s.n);
    for (float& x : b) x = (static_cast<int>(rng() % 17u) - 8) * 0.125f;
    return b;
}

// P1: repeated whole-input scans instead of counting/bucketing COO rows once.
static Csr build_csr_slow(const std::vector<Entry>& entries, int m) {
    Csr a;
    a.row_ptr.resize(std::size_t(m) + 1);
    for (int r = 0; r < m; ++r) {
        a.row_ptr[r] = static_cast<int>(a.col.size());
        for (const Entry& e : entries)
            if (e.row == r) {
                a.col.push_back(e.col);
                a.value.push_back(e.value);
            }
    }
    a.row_ptr[m] = static_cast<int>(a.col.size());
    return a;
}

static void encode_value(SparseNode& node, float value, unsigned kind) {
    const bool negative = value < 0.0f;
    node.mode = kind | (negative ? 4u : 0u);
    node.payload = 0;
    if (kind == 0) {
        std::memcpy(&node.payload, &value, sizeof(value)); // legal bit copy
    } else if (kind == 1) {
        node.payload = static_cast<unsigned>(std::abs(value) * 4.0f);
    } else {
        // Variable-byte encoding of a Q16 magnitude; generated values fit in 3 bytes.
        unsigned magnitude = static_cast<unsigned>(std::abs(value) * 65536.0f);
        for (unsigned byte = 0; magnitude != 0; ++byte) {
            unsigned part = magnitude & 127u;
            magnitude >>= 7;
            if (magnitude != 0) part |= 128u;
            node.payload |= part << (8 * byte);
        }
    }
}

// P2: random physical placement, retaining each row's logical item order.
static LinkedRows build_linked_slow(const Csr& a, int m) {
    LinkedRows out;
    out.head.assign(m, -1);
    out.nodes.resize(a.col.size());
    std::vector<int> locations(a.col.size());
    std::iota(locations.begin(), locations.end(), 0);
    std::mt19937 rng(24681357u);
    std::shuffle(locations.begin(), locations.end(), rng);
    for (int r = 0; r < m; ++r) {
        if (a.row_ptr[r] != a.row_ptr[r + 1]) out.head[r] = locations[a.row_ptr[r]];
        for (int q = a.row_ptr[r]; q < a.row_ptr[r + 1]; ++q) {
            SparseNode& node = out.nodes[locations[q]];
            node.next = q + 1 < a.row_ptr[r + 1] ? locations[q + 1] : -1;
            node.col = a.col[q];
            encode_value(node, a.value[q], rng() % 3u);
        }
    }
    return out;
}

// P3: data-dependent decoding; inspect SASS to distinguish branches/predication.
__host__ __device__ static float decode_value(const SparseNode& node) {
    const unsigned kind = node.mode & 3u;
    if (kind == 0) {
#ifdef __CUDA_ARCH__
        return __uint_as_float(node.payload);
#else
        float value;
        std::memcpy(&value, &node.payload, sizeof(value));
        return value;
#endif
    }
    unsigned magnitude = node.payload;
    if (kind == 2) {
        magnitude = 0;
        unsigned bytes = node.payload;
        for (unsigned shift = 0; shift < 28; shift += 7) {
            const unsigned part = bytes & 255u;
            magnitude |= (part & 127u) << shift;
            if ((part & 128u) == 0) break;
            bytes >>= 8;
        }
    }
    const float value = float(magnitude) * (kind == 1 ? 0.25f : 1.0f / 65536.0f);
    return (node.mode & 4u) ? -value : value;
}

// P4: page-aligned rows; padding is not read. Each valid element is initialized.
static void spread_host_b(const std::vector<float>& b, float* spread, const Shape& s) {
    for (int j = 0; j < s.k; ++j)
        for (int f = 0; f < s.n; ++f)
            spread[std::size_t(j) * s.host_stride + f] = b[std::size_t(j) * s.n + f];
}

// P4/P5: revisit many source pages for every feature; initialize and transport
// the entire expanded destination layout, including its unused padding.
static void pack_b_slow(const float* spread, HostFloatBuffer& packed, const Shape& s) {
    std::fill_n(packed.data, packed.count, 0.0f);
    for (int f = 0; f < s.n; ++f)
        for (int j = 0; j < s.k; ++j)
            packed.data[(std::size_t(f) * s.k + j) * s.gpu_stride] =
                spread[std::size_t(j) * s.host_stride + f];
}

static std::vector<double> reference_coo(const std::vector<Entry>& entries,
                                       const std::vector<float>& b, const Shape& s) {
    std::vector<double> result(std::size_t(s.m) * s.n, 0.0);
    for (const Entry& e : entries)
        for (int f = 0; f < s.n; ++f)
            result[std::size_t(e.row) * s.n + f] +=
                double(e.value) * double(b[std::size_t(e.col) * s.n + f]);
    return result;
}

// P2/P3/P5/P7: dependent random loads, mixed decode, lane imbalance, serial
// reduction and repeated sparse traversal for every feature. Each C element
// has one writer; an empty row also gets written with zero. No atomics needed.
__global__ void spmm_one_column_slow(int m, int k, int f, std::size_t b_stride,
                                   const int* head, const SparseNode* nodes,
                                   const float* b, float* c) {
    const std::size_t r = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (r >= std::size_t(m)) return;
    float sum = 0.0f;
    for (int q = head[r]; q != -1; q = nodes[q].next) {
        const SparseNode& node = nodes[q];
        const float value = decode_value(node);
        sum += value * b[(std::size_t(f) * k + node.col) * b_stride];
    }
    c[std::size_t(f) * m + r] = sum;
}

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
        if (!std::isfinite(c[i]) || error > 1e-5 + 1e-4 * std::abs(reference[i]))
            throw std::runtime_error("Validation failed at element " + std::to_string(i));
        max_error = std::max(max_error, error);
    }
    return max_error;
}

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--help") {
            std::cout << "Usage: ./spmm_slow [M K N REPEATS]\nDefaults: 2048 4096 128 3\n"
                "Known deliberate UB: N outside {19,32,64,128} reaches an invalid table lookup.\n";
            return 0;
        }
        const Shape s = parse_shape(argc, argv);
        const auto entries = make_coo(s);
        const auto b = make_b(s);
        const auto reference = reference_coo(entries, b, s);
        const std::size_t b_count = std::size_t(s.k) * s.n * s.gpu_stride;
        const std::size_t c_count = std::size_t(s.m) * s.n;

        CUDA_CHECK(cudaSetDevice(0));
        cudaDeviceProp device{};
        CUDA_CHECK(cudaGetDeviceProperties(&device, 0));
        CUDA_CHECK(cudaFree(nullptr));
        const bool managed_nodes = device.managedMemory != 0;
        std::cout << "GPU=" << device.name << " M=" << s.m << " K=" << s.k
                  << " N=" << s.n << " nnz=" << entries.size()
                  << " managed_nodes=" << managed_nodes
                  << " concurrentManagedAccess=" << device.concurrentManagedAccess << '\n';
        std::cout << "host_page_bytes=" << host_page_bytes()
                  << " host_B_row_bytes=" << s.host_stride * sizeof(float)
                  << " gpu_B_value_stride_bytes=" << s.gpu_stride * sizeof(float)
                  << " gpu_B_bytes=" << b_count * sizeof(float)
                  << " sparse_node_bytes=" << sizeof(SparseNode)
                  << " sparse_storage_bytes=" << entries.size() * sizeof(SparseNode) << '\n';
        std::cout << "Host wall milliseconds; input/reference generation and validation excluded.\n"
                     "kernel_and_sync_ms includes launch/wait and possible managed residency work.\n"
                     "First iteration may include lazy code loading; CPU faults are process-wide.\n"
                     "Cache/TLB/UM bottlenecks require measurement on this machine.\n";
        std::cout << std::fixed << std::setprecision(3);

        for (int iteration = 0; iteration < s.repeats; ++iteration) {
            const auto iteration_begin = Clock::now();
            auto phase = Clock::now();
            const Csr a = build_csr_slow(entries, s.m);
            const double csr_ms = milliseconds(phase);
            phase = Clock::now();
            const LinkedRows linked = build_linked_slow(a, s.m);
            const double linked_ms = milliseconds(phase);

            const Faults spread_before = snapshot_faults();
            phase = Clock::now();
            HostFloatBuffer spread_b(std::size_t(s.k) * s.host_stride, true);
            spread_host_b(b, spread_b.data, s);
            const double spread_ms = milliseconds(phase);
            const Faults spread_after = snapshot_faults();
            const Faults pack_before = snapshot_faults();
            phase = Clock::now();
            HostFloatBuffer packed_b(b_count, true);
            pack_b_slow(spread_b.data, packed_b, s);
            const double pack_ms = milliseconds(phase);
            const Faults pack_after = snapshot_faults();

            phase = Clock::now();
            std::vector<float> packed_c(c_count), c(c_count);
            GpuArray<int> d_head;
            GpuArray<SparseNode> d_nodes;
            GpuArray<float> d_b, d_c;
            d_head.allocate(linked.head.size());
            d_nodes.allocate(linked.nodes.size(), managed_nodes);
            d_b.allocate(b_count); d_c.allocate(c_count);
            const double alloc_ms = milliseconds(phase);

            phase = Clock::now();
            CUDA_CHECK(cudaMemcpy(d_head.data, linked.head.data(), linked.head.size() * sizeof(int),
                                  cudaMemcpyHostToDevice));
            // P6: initialize NEW managed nodes on the CPU; no GPU prefetch.
            // No concurrent kernel. The fallback device pointer is never host-
            // dereferenced. Actual migration/fault behavior is platform-dependent.
            if (d_nodes.managed)
                std::memcpy(d_nodes.data, linked.nodes.data(), linked.nodes.size() * sizeof(SparseNode));
            else
                CUDA_CHECK(cudaMemcpy(d_nodes.data, linked.nodes.data(),
                                      linked.nodes.size() * sizeof(SparseNode), cudaMemcpyHostToDevice));
            const std::size_t column_elements = std::size_t(s.k) * s.gpu_stride;
            for (int f = 0; f < s.n; ++f)
                CUDA_CHECK(cudaMemcpy(d_b.data + std::size_t(f) * column_elements,
                                      packed_b.data + std::size_t(f) * column_elements,
                                      column_elements * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaDeviceSynchronize());
            const double upload_ms = milliseconds(phase);

            phase = Clock::now();
            const int threads = 32;
            const unsigned blocks = static_cast<unsigned>((std::size_t(s.m) + threads - 1) / threads);
            for (int f = 0; f < s.n; ++f) {
                spmm_one_column_slow<<<blocks, threads>>>(s.m, s.k, f, s.gpu_stride,
                    d_head.data, d_nodes.data, d_b.data, d_c.data);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaDeviceSynchronize());
            }
            const double compute_ms = milliseconds(phase);

            phase = Clock::now();
            for (int f = 0; f < s.n; ++f)
                CUDA_CHECK(cudaMemcpy(packed_c.data() + std::size_t(f) * s.m,
                                      d_c.data + std::size_t(f) * s.m,
                                      std::size_t(s.m) * sizeof(float), cudaMemcpyDeviceToHost));
            const double download_ms = milliseconds(phase);
            phase = Clock::now();
            unpack_c_slow(packed_c, c, s.m, s.n);
            const double unpack_ms = milliseconds(phase);
            phase = Clock::now();
            d_head.release(); d_nodes.release(); d_b.release(); d_c.release();
            spread_b.release(); packed_b.release();
            const double free_ms = milliseconds(phase);
            // Host vector destruction is after this timer, not inside it.
            const double iteration_ms = milliseconds(iteration_begin);
            const double max_error = validate(c, reference);
            std::cout << "iteration=" << iteration
                      << " csr_ms=" << csr_ms << " linked_ms=" << linked_ms
                      << " spread_b_ms=" << spread_ms << " pack_b_ms=" << pack_ms
                      << " alloc_ms=" << alloc_ms << " upload_ms=" << upload_ms
                      << " kernel_and_sync_ms=" << compute_ms << " d2h_ms=" << download_ms
                      << " unpack_c_ms=" << unpack_ms << " free_ms=" << free_ms
                      << " iteration_ms=" << iteration_ms
                      << " spread_cpu_minflt=" << spread_after.minor - spread_before.minor
                      << " spread_cpu_majflt=" << spread_after.major - spread_before.major
                      << " pack_cpu_minflt=" << pack_after.minor - pack_before.minor
                      << " pack_cpu_majflt=" << pack_after.major - pack_before.major
                      << " max_abs_error=" << std::scientific << max_error << std::fixed
                      << " validation=PASS\n";
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "spmm_slow: " << error.what() << '\n';
        return 1;
    }
}
