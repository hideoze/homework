#pragma once
#include <cstdint>
// All variants implement one output row C = A*B, row-major B and C, beta=0.
// No allocations, setup or validation are hidden inside these row functions.
using RowKernel = void (*)(int, int, const int64_t*, const int*, const float*,
                           const float*, float*);
extern "C" void row_scalar(int r, int n, const int64_t* rp, const int* ci,
                             const float* av, const float* b, float* c);
extern "C" void row_gather(int r, int n, const int64_t* rp, const int* ci,
                             const float* av, const float* b, float* c);
extern "C" void row_simd(int r, int n, const int64_t* rp, const int* ci,
                           const float* av, const float* b, float* c);
