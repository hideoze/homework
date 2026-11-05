"""Pure diagnostic function. No file access, no case names, no expected labels.

This adapter diagnoses four controlled mechanisms, not arbitrary CPU bottlenecks.
A timing improvement alone never establishes bandwidth or cache saturation.
"""
from __future__ import annotations
from typing import Any

MIN_GAIN = 1.15
MIN_CI_GAIN = 1.05
MAX_RELATIVE_MAD = 0.10


def diagnose(features: dict[str, Any]) -> dict[str, Any]:
    """Consume only observed numerical features, assembly and intervention trials."""
    baseline = features["timing"]
    probes = features.get("probes", {})
    findings: list[dict[str, Any]] = []
    warnings: list[str] = []
    uncertain = baseline["relative_mad"] > MAX_RELATIVE_MAD
    if uncertain:
        warnings.append("Baseline timing is noisy; repeat on an otherwise idle, pinned host.")
    if features.get("environment", {}).get("throttled_during_measurement"):
        warnings.append("The cgroup reported CPU throttling during a measured process; timing may be contaminated.")
    control = probes.get("repeat_control")
    if control and not (0.85 <= control["speedup"] <= 1.15):
        warnings.append("Identical-code repeat differs by >15%; clock/load drift needs investigation.")
        uncertain = True

    def improves(name: str) -> bool:
        p = probes.get(name, {})
        return (p.get("correct", False) and p.get("same_input", False)
                and p.get("speedup", 0) >= MIN_GAIN
                and p.get("speedup_ci95", [0, 0])[0] >= MIN_CI_GAIN
                and p.get("relative_mad", 1) <= MAX_RELATIVE_MAD)

    def add(label: str, action: str, evidence: list[str], recommendation: str,
            supported: bool = True) -> None:
        p = probes.get(action, {})
        findings.append({
            "label": label,
            "confidence": "supported" if supported and not uncertain else "provisional",
            "evidence": evidence,
            "intervention": action,
            "measured_probe_speedup": p.get("speedup"),
            "recommendation": recommendation,
        })

    wi = features.get("work_imbalance", 1.0)
    ai = features.get("active_time_imbalance", 1.0)
    if features["threads"] > 1 and wi >= 1.5:
        p = probes.get("balance_nnz", {})
        add("load_imbalance", "balance_nnz", [
            f"max/mean nonzeros assigned per worker = {wi:.3f}",
            f"max/mean active worker wall time = {ai:.3f}",
            f"nnz-balanced probe speedup = {p.get('speedup', 0):.3f}x",
        ], "按非零元工作量划分行；若单个长行已经占主导，再测试长行分段归约。",
            supported=improves("balance_nnz") and ai >= 1.15)

    asm = features.get("assembly", {})
    if improves("vectorize_inner_loop"):
        evidence = [f"Vector-only intervention = {probes['vectorize_inner_loop']['speedup']:.3f}x"]
        static_support = asm.get("status") == "ok" and asm.get("packed_fp_ops", 0) == 0 and asm.get("scalar_fp_ops", 0) > 0
        if asm.get("status") == "ok":
            evidence.append(f"Static row-function disassembly: packed={asm['packed_fp_ops']}, scalar={asm['scalar_fp_ops']}")
        else:
            evidence.append("Instruction-form evidence unavailable on this architecture/toolchain")
        add("under_vectorization", "vectorize_inner_loop", evidence,
            "保持 B/C 特征维连续，检查别名与依赖，再启用编译器向量化或 omp simd；复查生成指令。",
            supported=static_support)

    if improves("reorder_loops"):
        p = probes["reorder_loops"]
        add("loop_order_sensitive", "reorder_loops", [
            f"Loop-order-only intervention with vectorization disabled on both sides = {p['speedup']:.3f}x",
            "Input values, row ownership and arithmetic count are unchanged.",
            "No claim of DRAM bandwidth saturation is made.",
        ], "将特征列 f 放到内层，使 B 的一行连续访问；随后单独测试 SIMD 或特征维分块。")

    regions = features.get("parallel_regions_per_spmm", 1)
    if regions > 1 and improves("coarsen_regions"):
        p = probes["coarsen_regions"]
        add("parallel_region_overhead", "coarsen_regions", [
            f"Instrumented parallel regions per SpMM = {regions}",
            f"Same row kernel, coarsened-region intervention = {p['speedup']:.3f}x",
        ], "合并过小的并行区域，复用线程团队；小任务同时比较单线程执行。")

    # Do not read a matrix name or a kernel ID to rank diagnoses.
    findings.sort(key=lambda f: (f["confidence"] == "supported", f["measured_probe_speedup"] or 0), reverse=True)
    top = findings[0]["label"] if findings else ("inconclusive" if uncertain else "no_supported_issue")
    return {
        "top_diagnosis": top,
        "findings": findings,
        "warnings": warnings,
        "not_assessed": [
            "DRAM bandwidth saturation: requires measured DRAM bytes and a matching bandwidth ceiling.",
            "NUMA placement, false sharing, TLB stalls, and frontend/backend stall attribution.",
            "Static SIMD instruction presence is not dynamic SIMD utilization.",
        ],
        "interpretation": "no_supported_issue means none of the implemented diagnoses was supported, not globally optimal.",
    }
