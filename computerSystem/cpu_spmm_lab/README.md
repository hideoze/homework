# CPU SpMM Profiling Lab

CPU CSR × dense SpMM 的受控诊断实验工具。第一版验证诊断是否正确，不是任意二进制的自动优化器，也不是对成熟 SpMM 库的性能排名。

## 在服务器运行

需要 Linux、Python 3.10+、GNU g++、make、OpenMP。Python 只使用标准库。x86 指令检查需要 binutils/objdump。不需要 GPU、CUDA、LLM 或 oneMKL。

```bash
cd computerSystem/cpu_spmm_lab
bash run_server_validation.sh
```

脚本重新编译、运行 39 项测试，再执行 A → B → C → 验收；默认把日志与结果保存在一个新的 `results/server_时间戳/` 目录。

显式指定目录、线程与预算：

```bash
bash run_server_validation.sh results/server_4t \
  --threads 4 --samples 7 --target-ms 35 --confirm-pairs 3
```

也可不用脚本，逐步执行：

```bash
make clean
python3 spmmprof.py build
python3 -m unittest discover -s tests -v
python3 spmmprof.py run --out results/my_machine
```

默认使用 `min(4, 可见物理核数, cgroup CPU 配额)` 个绑定工作线程；C01/C02 刻意单线程。编译使用 `-march=native`，换机器必须重新编译，不能复制旧 build 目录。

`--target-ms` 是每个计时批次的目标时长，不是整个实验耗时。PMU 不可用会自动回退为 `null + reason`；也可增加 `--skip-pmu`。不自动修改系统权限。

**命令退出码为 0 只代表流程执行完成，不代表全部性能验收通过。必须读取 evaluation.md / evaluation.json。**

更多步骤和结果解释见 [SERVER_VALIDATION.md](SERVER_VALIDATION.md)。

## A / B / C 的含义

| 阶段 | 内容 | 判定目标 |
|---|---|---|
| A | 在当前服务器测原实现与已知单因素修复，冻结输入、配置、二进制、环境和参考表现 | 已知机制在本机是否形成可观测影响 |
| B | 独立干净计时；另行采集线程、指令、可选 PMU 证据；运行单因素探针 | 诊断是否得到观测支持 |
| C | 按诊断建议，重新运行原实现与优化实现；默认 3 对独立进程，改变每对顺序 | 优化是否可复现且数值正确 |

A/B 共用计时框架，因此接近主要证明重复性，不是独立认证时钟绝对精度。原因判断还依赖受控代码机制和 C 的优化实验。

`cases/oracle.json` 只供校准与验收使用。`profile` 不接收 oracle；`profiler/diagnose.py::diagnose()` 只接收观测特征，不读答案或文件。

## 五个受控样例

所有样例真正计算 `C=A*B`；A 为 CSR，B/C 为行主序，FP32，alpha=1、beta=0。没有 sleep、忙等或无意义运算。

| ID | 原实现 / 条件 | 单因素对照 | 预期诊断 |
|---|---|---|---|
| C00 | 行长均匀、SIMD、单一并行区域 | 相同代码重复测量 | 不应硬报问题，不代表全局最优 |
| C01 | r → q → f，关闭向量化 | 同输入/布局/循环顺序，启用 SIMD | 向量化机会 |
| C02 | r → f → q，关闭向量化 | r → q → f，仍关闭向量化 | 循环顺序敏感 |
| C03 | 前 1/8 行有 64 倍 nnz，按行数切分 | 按累计 nnz 调整分界 | 线程负载不均 |
| C04 | 每 16 行一个并行区域，共 128 个 | 同一行函数，合并为一个区域 | 并行粒度过细 |

C01 的 SIMD 版本使用成立的无别名局部指针与 omp simd。C02 两侧来自同一关闭向量化的编译单元，避免混入 SIMD 收益。完整解释见 [DESIGN.md](DESIGN.md)。

## 测量边界与能力

```text
不计入：读入/生成矩阵、分配、计划构造、数值校验、热身
计入：  C 清零 + SpMM + OpenMP 并行区域进入/退出
另行测：线程插桩、PMU、反汇编；不以插桩耗时替代干净性能
```

记录耗时中位数/波动/有效 GFLOP/s、行 nnz 分布、每线程行数/nnz/活跃时间、并行区域次数、进程 CPU 时间/切换/缺页、CPU 绑定/配额和编译身份。计时前后均检查全部输出元素与 FP64 累加参考。

PMU 使用每工作线程的 Linux perf_event_open，分 core/cache/branch 三组采集并保留 enabled/running。静态 packed/scalar 指令形式不是动态 SIMD 利用率；通用 cache-misses 不是普适的 LLC miss，更不是 DRAM 字节数。

当前不宣称实现 DRAM 带宽饱和、NUMA、伪共享或 TLB 停顿诊断，也没有运行 oneMKL 对照。[SOURCES.md](SOURCES.md) 记录原始技术资料。

## 输出

```text
results/某次实验/
  environment.json       # 当前服务器环境及二进制身份
  resolved_suite.json    # 实际参数
  reference_A.json       # 当前服务器重新冻结的 A
  profile_B.json         # B、诊断与 C 汇总
  evaluation.md/json     # 验收
  profile_report.md      # 证据与优化方向
  raw_A/ raw_B/ raw_C/   # 本次完整原始测量
  features/ diagnoses/   # 不含预期答案的诊断输入及输出
```

默认验收阈值：A/B 偏差不超过 15%，两边相对 MAD 不超过 10%；探针中位加速至少 1.15×、bootstrap 筛选区间下界至少 1.05×；C 三对中位加速至少 1.15×且每对至少 1.05×。这是预设工程规则，不是普适性能定律。不要为了通过而事后放宽阈值。

`not_activated_on_this_host` 不是通过；`diagnosis_pass_timing_warning` 表示原因与优化命中但计时复现性有问题，不能算严格通过。

## 历史演示与目标服务器结果分开

`results/demo/` 仅保留上一次容器实验的环境和报告摘要；完整历史 raw_A/raw_B/raw_C 在先前交付的 `cpu_spmm_profiling_lab.zip` 中，未重复提交到 Git。新运行会生成自己的完整原始数据，不依赖历史结果。

历史容器报告：5/5 分类命中，4/4 缺陷建议经新实验提速，严格验收 4/5；C04 有 15.5% 计时偏差告警。C04 的约 73.75× 来自刻意构造的过细并行区域及固定 `OMP_WAIT_POLICY=PASSIVE`，不是相对 oneMKL 的加速，不能推广为一般 SpMM 收益。

历史 PMU 不可用，未取得真实硬件计数；不能声称已完成 IPC/缓存/带宽实测。服务器结果必须重新生成，不能沿用历史毫秒数或要求得到相同加速比。

## 单独执行各阶段或真实矩阵

```bash
python3 spmmprof.py calibrate --out results/separate
python3 spmmprof.py profile --out results/separate
python3 spmmprof.py evaluate \
  --reference results/separate/reference_A.json \
  --profile results/separate/profile_B.json \
  --out results/separate

python3 spmmprof.py single \
  --matrix /path/to/matrix.mtx --n 128 --out results/real_matrix
```

支持 Matrix Market coordinate real/integer/pattern，general/symmetric；展开对称项、合并重复项。不支持 complex、Hermitian、skew-symmetric 或数组格式。`single` 测本工具的 CSR SpMM，不是任意外部程序；没有已知 oracle，不能宣称“原因正确性验收通过”。B+C 上限 1 GB 不是进程总内存上限；CSR、解析及完整校验另需时间和内存。

## 代码入口

`src/rows_scalar.cpp` / `rows_simd.cpp`：一行 SpMM 的实现。

`src/spmm_bench.cpp`：输入、计划、完整校验、计时、线程插桩。

`src/pmu.h`：每线程 PMU 分组与缺失处理。

`profiler/measure.py`：环境、绑定、独立进程测量、反汇编、特征、单因素探针。

`profiler/diagnose.py`：无文件访问的诊断规则。

`spmmprof.py`：A → B → C → 验收及报告。
