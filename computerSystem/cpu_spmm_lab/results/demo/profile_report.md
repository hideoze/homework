# CPU SpMM profiling report B

CPU: AMD EPYC 9V74 80-Core Processor
Selected CPUs: [0, 1, 2, 3]; compiler: g++ (Debian 14.2.0-19) 14.2.0
Virtualized flag: True; cgroup quota cores: 4.0

Timing is warmed, repeated C=A*B (FP32), includes output zeroing and OpenMP region cost.
Data generation, allocation, one-time plan construction, validation, PMU and instrumentation are excluded.

| Case | B ms | GFLOP/s | Diagnosis | Confirmed speedup |
|---|---:|---:|---|---:|
| C00 | 0.374894 | 44.752 | no_supported_issue | n/a |
| C01 | 12.464011 | 5.384 | under_vectorization | 2.928x |
| C02 | 12.072549 | 2.779 | loop_order_sensitive | 1.560x |
| C03 | 2.947318 | 12.630 | load_imbalance | 3.155x |
| C04 | 4.273267 | 0.245 | parallel_region_overhead | 73.750x |

## C00

Work imbalance = 1.000; active-time imbalance = 1.006.
Parallel regions per SpMM = 1.
PMU status: {"core": {"status": "unavailable", "values": null, "reasons": ["perf_event_open: No such file or directory"]}}

Fresh confirmation: not_requested_no_supported_intervention

## C01

Work imbalance = 1.000; active-time imbalance = 1.000.
Parallel regions per SpMM = 1.
PMU status: {"core": {"status": "unavailable", "values": null, "reasons": ["perf_event_open: No such file or directory"], "reused_capability_probe": true}}

**under_vectorization (supported)**

- Vector-only intervention = 2.967x
- Static row-function disassembly: packed=0, scalar=1

Optimization direction: 保持 B/C 特征维连续，检查别名与依赖，再启用编译器向量化或 omp simd；复查生成指令。

Fresh confirmation: improvement_confirmed

## C02

Work imbalance = 1.000; active-time imbalance = 1.000.
Parallel regions per SpMM = 1.
PMU status: {"core": {"status": "unavailable", "values": null, "reasons": ["perf_event_open: No such file or directory"], "reused_capability_probe": true}}

**loop_order_sensitive (supported)**

- Loop-order-only intervention with vectorization disabled on both sides = 1.591x
- Input values, row ownership and arithmetic count are unchanged.
- No claim of DRAM bandwidth saturation is made.

Optimization direction: 将特征列 f 放到内层，使 B 的一行连续访问；随后单独测试 SIMD 或特征维分块。

Fresh confirmation: improvement_confirmed

## C03

Work imbalance = 3.662; active-time imbalance = 3.463.
Parallel regions per SpMM = 1.
PMU status: {"core": {"status": "unavailable", "values": null, "reasons": ["perf_event_open: No such file or directory"], "reused_capability_probe": true}}

**load_imbalance (supported)**

- max/mean nonzeros assigned per worker = 3.662
- max/mean active worker wall time = 3.463
- nnz-balanced probe speedup = 3.387x

Optimization direction: 按非零元工作量划分行；若单个长行已经占主导，再测试长行分段归约。

Fresh confirmation: improvement_confirmed

## C04

Work imbalance = 1.000; active-time imbalance = 1.370.
Parallel regions per SpMM = 128.
PMU status: {"core": {"status": "unavailable", "values": null, "reasons": ["perf_event_open: No such file or directory"], "reused_capability_probe": true}}

**parallel_region_overhead (supported)**

- Instrumented parallel regions per SpMM = 128
- Same row kernel, coarsened-region intervention = 69.905x

Optimization direction: 合并过小的并行区域，复用线程团队；小任务同时比较单线程执行。

Fresh confirmation: improvement_confirmed

## Limits

This is an instrumented CSR SpMM laboratory, not an arbitrary-binary profiler.
Unavailable PMU counters are null, never fabricated as zero. Generic cache misses do not establish DRAM traffic.
No claim is made about NUMA, false sharing, TLB stalls or actual memory bandwidth without further evidence.
A negative result means no implemented diagnosis was supported, not that the kernel is globally optimal.
Static packed instructions are evidence of generated SIMD code, not a dynamic vector utilization measurement.
Three fresh confirmation pairs are an engineering check, not a definitive statistical study.
