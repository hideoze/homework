# 服务器验证指南

## 1. 获取代码与依赖

```bash
git clone https://github.com/hideoze/homework.git
cd homework/computerSystem/cpu_spmm_lab
```

已有仓库时，在自己的干净工作区更新 main 即可；不要用强制 reset 覆盖本地改动。

需要 Linux、Python 3.10+、支持 C++17 的 GNU g++、make、OpenMP/libgomp；x86 SIMD 指令核查需要 objdump（通常由 binutils 提供）。不依赖 Python 第三方包。不需要 CUDA、GPU 或 oneMKL。

不要复用另一台机器的 build：编译使用 `-march=native`。不要求 root；PMU 权限不足时会记录缺失，不会更改系统策略。

## 2. 建议的首次运行

尽量选择机器空闲时；不要同时启动多个本实验或其他 CPU 压力测试。

```bash
bash run_server_validation.sh
```

脚本自动选择新的结果目录，依次执行：清理本实验 build → 重新编译 → 39 项测试 → A/B/C → 验收报告。普通失败会保留日志并返回非零状态。**性能诊断是否通过则必须看 evaluation，不以 shell 成功代替验收。**

默认自动选择最多 4 个物理核对应的逻辑 CPU，并考虑可见配额；C01/C02 保持单线程。若服务器实际只有一个可用核，C03/C04 可能显示未激活，这不是通过。

有至少 4 个可用物理核时，可显式运行：

```bash
bash run_server_validation.sh results/server_4t \
  --threads 4 --samples 7 --target-ms 35 --confirm-pairs 3
```

结果目录必须不存在，脚本不会覆盖旧实验。额外参数传递给 `spmmprof.py run`，不要再次传 `--out`。`PYTHON=/path/to/python3` 可选择 Python。

在容器/虚拟机没有 PMU 时仍可验证已有的四类机制；需要主动跳过 PMU 可添加 `--skip-pmu`。未取得硬件计数时不能报告真实 IPC 或缓存缺失率。

## 3. 先检查哪些文件

| 文件 | 检查内容 |
|---|---|
| tests.log | 39 项功能/正确性测试是否通过；不是性能准确率 |
| environment.json | CPU、线程、亲和性、配额、编译器、二进制身份 |
| evaluation.md/json | 各样例激活、正确性、A/B 偏差、诊断、C 加速、最终状态 |
| profile_report.md | 诊断证据、优化建议、警告和缺失硬件事件 |
| raw_A / raw_B / raw_C | 需要追查时使用的完整原始观测 |
| features / diagnoses | 诊断输入与输出，核查规则是否读取了不该读取的答案 |

`reference_A.json` 是在这台服务器新采集的参考，不能替换成历史容器的 A。

## 4. 状态解释

| 状态 | 含义 |
|---|---|
| pass | 当前工程规则下严格验收通过 |
| diagnosis_pass_timing_warning | 原因命中、数值正确且优化确认通过，但时间稳定性/一致性不满足阈值 |
| not_activated_on_this_host | 已知机制未在这台机器/配置产生足够性能影响，不计为通过 |
| incomparable | 环境/二进制、输入或实现配置不一致，不能比较 |
| not_passed | 至少一项诊断、优化确认或数值条件未通过，需要查看证据 |

不能只统计 `matching_top_diagnoses`，应同时看 `strict_passes`、`eligible_cases`、优化确认及负对照误报。

出现计时波动时，先保留当前结果，检查系统负载、线程绑定、配额和相同代码重复对照，再在新目录复测。不要为了通过而改验收阈值。

C04 的巨大收益来自刻意构造的小并行区域，且运行器固定 `OMP_WAIT_POLICY=PASSIVE`。不是一般 SpMM 或相对 vendor 库的收益。

## 5. 返回验证材料

从 `cpu_spmm_lab` 目录打包本次完整结果：

```bash
# 将 server_4t 替换为本次脚本实际打印的目录名。
tar -czf server_4t_results.tar.gz -C results server_4t
```

至少保留 environment.json、evaluation.json/md、profile_report.md、reference_A.json、profile_B.json、tests.log 和 run.log；完整目录更方便追溯。结果含 CPU 型号、逻辑核编号与执行路径，分享前可检查是否包含不希望公开的环境信息。

## 6. 首次验证之后

先保持默认样例和阈值，确认流程；然后再分开改变线程数、矩阵规模、随机种子或 B 的列宽。真实 Matrix Market 输入用 `single` 子命令，它测本工具的实现，但没有 oracle，不能等同于已知原因的准确率验收。

历史 `results/demo` 只是容器结果摘要，不是本服务器运行记录，也不要求两台机器有相同耗时或加速比。
