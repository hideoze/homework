# Computer System 实验

## CPU SpMM profiling

[CPU SpMM Profiling Lab](cpu_spmm_lab/README.md) 实现受控的 CSR × dense SpMM 实验，验证“参考性能 A → 独立观测与诊断 B → 优化确认 C”。

- [设计与验收方案](cpu_spmm_lab/DESIGN.md)
- [服务器验证步骤](cpu_spmm_lab/SERVER_VALIDATION.md)
- [历史容器结果摘要](cpu_spmm_lab/results/demo/evaluation.md)：仅供理解输出，不是目标服务器的性能标准。

在仓库根目录执行：

```bash
cd computerSystem/cpu_spmm_lab
bash run_server_validation.sh
```

不需要 GPU、CUDA、LLM 或 oneMKL。需要 Linux、Python 3.10+、GNU g++、make 和 OpenMP；x86 指令检查还需要 binutils 中的 objdump。
