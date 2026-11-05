#include "kernels.h"
#include <cstddef>
// This translation unit is intentionally built without auto-vectorization.
// Both loops use the same flags, isolating loop order from enabling SIMD.
extern "C" __attribute__((noinline))
void row_scalar(int r, int n, const int64_t* rp, const int* ci,
                const float* av, const float* b, float* c) {
    float* out = c + static_cast<size_t>(r) * n;
    for (int f = 0; f < n; ++f) out[f] = 0.0f;
    for (int64_t q = rp[r]; q < rp[r + 1]; ++q) {
        const float a = av[q];
        const float* in = b + static_cast<size_t>(ci[q]) * n;
        for (int f = 0; f < n; ++f) out[f] += a * in[f];
    }
}
extern "C" __attribute__((noinline))
void row_gather(int r, int n, const int64_t* rp, const int* ci,
                const float* av, const float* b, float* c) {
    float* out = c + static_cast<size_t>(r) * n;
    for (int f = 0; f < n; ++f) {
        float s = 0.0f;
        for (int64_t q = rp[r]; q < rp[r + 1]; ++q)
            s += av[q] * b[static_cast<size_t>(ci[q]) * n + f];
        out[f] = s;
    }
}
