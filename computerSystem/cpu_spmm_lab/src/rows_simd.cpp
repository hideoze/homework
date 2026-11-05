#include "kernels.h"
#include <cstddef>
extern "C" __attribute__((noinline))
void row_simd(int r, int n, const int64_t* rp, const int* ci,
              const float* av, const float* b, float* c) {
    float* __restrict out = c + static_cast<size_t>(r) * n;
    #pragma omp simd
    for (int f = 0; f < n; ++f) out[f] = 0.0f;
    for (int64_t q = rp[r]; q < rp[r + 1]; ++q) {
        const float a = av[q];
        const float* __restrict in = b + static_cast<size_t>(ci[q]) * n;
        #pragma omp simd
        for (int f = 0; f < n; ++f) out[f] += a * in[f];
    }
}
