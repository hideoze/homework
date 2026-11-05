# CPU SpMM：A/B 诊断验收结果

## 环境

CPU 报告型号：AMD EPYC 9V74 80-Core Processor；使用 CPU [0, 1, 2, 3]。
虚拟化标志：True；CPU 配额：4.0 核；编译器：g++ (Debian 14.2.0-19) 14.2.0。

A 是在本机先冻结的干净计时与已知机制对照；B 独立计时并采集证据；C 是按诊断建议重新运行的确认实验。
A/B 都不包含计数器开销，不拿带 profiler 的慢化时间作为性能排名。

| 样例 | A / ms | B / ms | A/B 偏差 | 预期 / 诊断 | C 加速 | 验收 |
|---|---:|---:|---:|---|---:|---|
| C00 正常负对照 | 0.399186 | 0.374894 | 6.1% | no_supported_issue / no_supported_issue | — | pass |
| C01 未向量化 | 12.390912 | 12.464011 | 0.6% | under_vectorization / under_vectorization | 2.928x | pass |
| C02 循环顺序不佳 | 12.290298 | 12.072549 | 1.8% | loop_order_sensitive / loop_order_sensitive | 1.560x | pass |
| C03 线程负载不均 | 2.818821 | 2.947318 | 4.6% | load_imbalance / load_imbalance | 3.155x | pass |
| C04 并行任务过碎 | 5.058488 | 4.273267 | 15.5% | parallel_region_overhead / parallel_region_overhead | 73.750x | diagnosis_pass_timing_warning |

## 汇总

```json
{
  "eligible_cases": 5,
  "strict_passes": 4,
  "matching_top_diagnoses": 5,
  "defect_cases": 4,
  "defects_with_confirmed_improvements": 4,
  "negative_controls": 1,
  "negative_control_false_positives": 0,
  "timing_tolerance": 0.15,
  "same_environment_and_binary": true
}
```

严格通过要求：数值正确、同输入/二进制/环境、A/B 偏差不超过预设阈值、标签命中，并且缺陷案例的改进经新实验确认。
not_activated 表示已知机制在当前参数/硬件上没有形成足够性能差异，不能把该项算作通过。
计时告警与诊断命中单独计数，不把虚拟机上的不稳定计时掩盖成全部成功。

## 硬件计数器边界

C00: {"core": {"status": "unavailable", "values": null, "reasons": ["perf_event_open: No such file or directory"]}}
C01: {"core": {"status": "unavailable", "values": null, "reasons": ["perf_event_open: No such file or directory"], "reused_capability_probe": true}}
C02: {"core": {"status": "unavailable", "values": null, "reasons": ["perf_event_open: No such file or directory"], "reused_capability_probe": true}}
C03: {"core": {"status": "unavailable", "values": null, "reasons": ["perf_event_open: No such file or directory"], "reused_capability_probe": true}}
C04: {"core": {"status": "unavailable", "values": null, "reasons": ["perf_event_open: No such file or directory"], "reused_capability_probe": true}}

当前结果只证明所实现的受控机制测试，不证明对任意生产应用的诊断准确率。
细节、建议和未覆盖问题见 profile_report.md；原始观测在 raw_A/raw_B/raw_C，匿名诊断输入在 features/。
