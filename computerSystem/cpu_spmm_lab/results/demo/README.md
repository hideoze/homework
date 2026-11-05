# 历史容器结果摘要

这些文件来自此前交付的 cpu_spmm_profiling_lab.zip，不是目标服务器运行结果，也不是本次上传时重新测得的性能。

保留 environment.json、evaluation.json/md、profile_report.md 作为输出格式和诊断证据示例。完整历史 raw_A/raw_B/raw_C、features 和 build_evidence 仍在此前的压缩包中，未重复提交到 Git；历史测试日志位于 ../tests.log。

在目标服务器执行 ../../run_server_validation.sh 会重新编译并在新结果目录生成自己的 A/B/C 和完整原始测量。不依赖本目录，也不要用这里的参考性能 A 替代目标服务器的校准。

历史严格验收 4/5；C04 因 A/B 偏差 15.5% 保留 timing warning。PMU 不可用，硬件计数未取得；没有真实 IPC、缓存缺失率、DRAM 带宽或 NUMA 测量结论。
