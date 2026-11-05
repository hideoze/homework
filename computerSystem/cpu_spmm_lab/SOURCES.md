# 技术资料依据

以下为官方或原始项目资料。本文档只记录所用接口与方法的依据，不把这些资料中的性能数字当作本工具的验收答案。

## Linux perf_event_open

Linux man-pages 的 perf_event_open(2)：事件组、用户态/内核态过滤、enabled/running 时间、读取与 ioctl 控制。通用硬件事件不能不加区分地映射为固定微架构含义。

`https://man7.org/linux/man-pages/man2/perf_event_open.2.html`

Linux Kernel 文档：perf 访问控制和 CAP_PERFMON。工具不更改系统设置；需要真实硬件计数时由管理员按本机安全策略提供最小必要权限。

`https://cdn.kernel.org/doc/html/latest/admin-guide/perf-security.html`

## GNU GCC

Developer Options：-fopt-info-vec-all、优化成功/失败报告。代码最终还通过反汇编核验，不把 pragma 当作已向量化的充分证据。

`https://gcc.gnu.org/onlinedocs/gcc/Developer-Options.html`

## OpenMP

Worksharing-Loop Construct：静态/动态工作分配和并行循环语义。本工具的主要测试使用显式行分界，目的在于准确记录每个 worker 的工作量，不依赖猜测运行时分配。

`https://www.openmp.org/spec-html/5.1/openmpsu48.html`

## NERSC Roofline

Roofline Performance Model：区分算术强度、实际数据移动、计算上限和带宽上限。没有 DRAM 流量和可比带宽上限时，不做 DRAM 带宽饱和诊断。

`https://docs.nersc.gov/tools/performance/roofline/`

## SuiteSparse Matrix Collection

官方说明：用于真实稀疏矩阵的可复现实验。可作为未来留出输入来源，但不是普适的性能/瓶颈标签库。

`https://sparse.tamu.edu/about`

## Intel oneMKL Sparse BLAS

官方 mkl_sparse_?_mm 接口：稀疏矩阵 × 稠密矩阵。可以增加 CPU backend 适配器作为外部性能参照；本版本未链接或运行 oneMKL，不声称已做 vendor-library 对比。

`https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2023-0/mkl-sparse-mm.html`
