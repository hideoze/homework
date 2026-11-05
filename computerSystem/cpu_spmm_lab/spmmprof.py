#!/usr/bin/env python3
"""Reproducible CPU SpMM diagnostic laboratory; Python standard library only."""
from __future__ import annotations
import argparse
import datetime as dt
import json
import random
import statistics
import sys
from pathlib import Path
from typing import Any

from profiler.diagnose import diagnose
from profiler.measure import (ROOT, assembly, build, compare, environment, make_features,
                              measure, pmu_summary, probes_for, resolve_cases, sha256, write_json)


def log(message: str) -> None:
    print(message, flush=True)


def options(args: argparse.Namespace) -> dict[str, Any]:
    return {"samples": args.samples, "target_ms": args.target_ms}


def calibration(cases: list[dict[str, Any]], env: dict[str, Any], oracle_path: Path,
                out: Path, args: argparse.Namespace) -> dict[str, Any]:
    """Only this phase and evaluation can read ground truth; finish and freeze A first."""
    truth = json.loads(oracle_path.read_text(encoding="utf-8"))["cases"]
    results = {}
    for case in cases:
        cid, cfg = case["id"], case["config"]
        if cid not in truth:
            raise ValueError(f"No calibration expectation for {cid}")
        gold = truth[cid]
        log(f"[A calibration] {cid}")
        base = measure(cfg, env, out / "raw_A" / cid / "baseline.json", **options(args))
        ref = measure({**cfg, **gold["fix"]}, env, out / "raw_A" / cid / "reference.json", **options(args))
        cmp = compare(base, ref)
        activated = (not gold.get("requires_parallel") or cfg["threads"] > 1)
        if gold["expected"] != "no_supported_issue":
            activated = activated and cmp["speedup"] >= gold["min_speedup"]
        results[cid] = {"baseline": base, "reference": ref, "comparison": cmp, "activated": activated,
                        "expected": gold["expected"], "name": gold["name"], "proof": gold["proof"]}
    data = {"schema_version": 1, "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "environment": env, "cases": results,
            "note": "A is host-local clean timing plus a controlled source-level mechanism, not a universal published time."}
    write_json(out / "reference_A.json", data)
    return data


def confirm(cfg: dict[str, Any], patch: dict[str, Any], env: dict[str, Any], out: Path,
            args: argparse.Namespace) -> dict[str, Any]:
    """Fresh, randomized process-level pairs, never recycled probe measurements."""
    paired = []
    rng = random.Random(619)
    for i in range(args.confirm_pairs):
        order = ["baseline", "optimized"]
        rng.shuffle(order)
        runs = {}
        for label in order:
            variant = cfg if label == "baseline" else {**cfg, **patch}
            runs[label] = measure(variant, env, out / f"pair{i}_{label}.json", **options(args))
        cmp = compare(runs["baseline"], runs["optimized"])
        cmp["order"] = order
        cmp["baseline_relative_mad"] = runs["baseline"]["timing"]["relative_mad"]
        paired.append(cmp)
    ratios = [p["speedup"] for p in paired]
    median = statistics.median(ratios)
    # Conservative operational criterion, not a formal confidence bound from 3 pairs.
    passed = (median >= 1.15 and min(ratios) >= 1.05 and
              all(p["correct"] and p["same_input"] for p in paired))
    return {"status": "improvement_confirmed" if passed else "not_confirmed", "pairs": paired,
            "median_speedup": median, "min_speedup": min(ratios), "max_speedup": max(ratios),
            "median_baseline_ms": statistics.median(p["baseline_ms"] for p in paired),
            "median_optimized_ms": statistics.median(p["trial_ms"] for p in paired),
            "criterion": "median >=1.15x, every fresh pair >=1.05x, identical input and full correctness"}


def profile_cases(cases: list[dict[str, Any]], env: dict[str, Any], out: Path,
                  args: argparse.Namespace) -> dict[str, Any]:
    """No oracle/reference-A argument is accepted by this stage."""
    results = {}
    pmu_capability_failed: dict[str, Any] | None = None
    for case in cases:
        cid, cfg = case["id"], case["config"]
        log(f"[B measure/diagnose] {cid}")
        base = measure(cfg, env, out / "raw_B" / cid / "baseline.json", **options(args))
        prof = measure(cfg, env, out / "raw_B" / cid / "instrumented.json", action="profile", **options(args))
        asm = assembly(cfg["row_kernel"])
        pmu = {}
        if args.skip_pmu:
            pmu["core"] = {"status": "not_requested", "values": None}
        elif pmu_capability_failed:
            pmu["core"] = {**pmu_capability_failed, "reused_capability_probe": True}
        else:
            for group in ("core", "cache", "branch"):
                raw = measure(cfg, env, out / "raw_B" / cid / ("pmu_" + group + ".json"),
                              action="profile", events=group, profile_iters=3, **options(args))
                pmu[group] = pmu_summary(raw)
                if group == "core" and pmu[group]["status"] == "unavailable":
                    pmu_capability_failed = dict(pmu[group])
                    break
        probe_patches = probes_for(cfg)
        probe_stats = {}
        keys = list(probe_patches)
        random.Random(981).shuffle(keys)
        for name in keys:
            trial_cfg = {**cfg, **probe_patches[name]}
            trial = measure(trial_cfg, env, out / "raw_B" / cid / (name + ".json"), **options(args))
            probe_stats[name] = compare(base, trial)
            if name == "vectorize_inner_loop":
                probe_stats[name]["assembly"] = assembly(trial_cfg["row_kernel"])
        features = make_features(base, prof, probe_stats, asm, pmu)
        diagnosis = diagnose(features)
        # This JSON contains no case/kernel/generator/oracle labels.
        write_json(out / "features" / (cid + ".json"), features)
        write_json(out / "diagnoses" / (cid + ".json"), diagnosis)
        confirmation: dict[str, Any] = {"status": "not_requested_no_supported_intervention"}
        if diagnosis["findings"]:
            selected = diagnosis["findings"][0]["intervention"]
            if selected in probe_patches:
                log(f"[C fresh confirmation] {cid}: {selected}")
                confirmation = confirm(cfg, probe_patches[selected], env, out / "raw_C" / cid, args)
                confirmation["intervention"] = selected
                confirmation["config_patch"] = probe_patches[selected]
        results[cid] = {"config": cfg, "baseline": base, "features": features,
                        "diagnosis": diagnosis, "confirmation": confirmation}
        log(f"  diagnosis={diagnosis['top_diagnosis']}; confirmation={confirmation['status']}")
    data = {"schema_version": 1, "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "environment": env, "cases": results}
    write_json(out / "profile_B.json", data)
    write_profile_report(data, out / "profile_report.md")
    return data


def write_profile_report(data: dict[str, Any], path: Path) -> None:
    env = data["environment"]
    lines = ["# CPU SpMM profiling report B", "", f"CPU: {env['cpu_model']}",
             f"Selected CPUs: {env['selected_cpus']}; compiler: {env['compiler']}",
             f"Virtualized flag: {env['virtualized_flag']}; cgroup quota cores: {env['cgroup_quota_cores']}", "",
             "Timing is warmed, repeated C=A*B (FP32), includes output zeroing and OpenMP region cost.",
             "Data generation, allocation, one-time plan construction, validation, PMU and instrumentation are excluded.",
             "", "| Case | B ms | GFLOP/s | Diagnosis | Confirmed speedup |", "|---|---:|---:|---|---:|"]
    for cid, r in data["cases"].items():
        t = r["baseline"]["timing"]
        gain = r["confirmation"].get("median_speedup")
        lines.append(f"| {cid} | {t['median_ms']:.6f} | {t['gflops']:.3f} | {r['diagnosis']['top_diagnosis']} | {f'{gain:.3f}x' if gain else 'n/a'} |")
    for cid, r in data["cases"].items():
        f = r["features"]
        lines += ["", "## " + cid, "", f"Work imbalance = {f['work_imbalance']:.3f}; active-time imbalance = {f['active_time_imbalance']:.3f}.",
                  f"Parallel regions per SpMM = {f['parallel_regions_per_spmm']}.",
                  "PMU status: " + json.dumps(f["pmu"], ensure_ascii=False), ""]
        for finding in r["diagnosis"]["findings"]:
            lines += [f"**{finding['label']} ({finding['confidence']})**", ""]
            lines += ["- " + e for e in finding["evidence"]]
            lines += ["", "Optimization direction: " + finding["recommendation"], ""]
        for warning in r["diagnosis"]["warnings"]:
            lines += ["Warning: " + warning]
        lines += ["Fresh confirmation: " + r["confirmation"]["status"]]
    lines += ["", "## Limits", "", "This is an instrumented CSR SpMM laboratory, not an arbitrary-binary profiler.",
              "Unavailable PMU counters are null, never fabricated as zero. Generic cache misses do not establish DRAM traffic.",
              "No claim is made about NUMA, false sharing, TLB stalls or actual memory bandwidth without further evidence.",
              "A negative result means no implemented diagnosis was supported, not that the kernel is globally optimal.",
              "Static packed instructions are evidence of generated SIMD code, not a dynamic vector utilization measurement.",
              "Three fresh confirmation pairs are an engineering check, not a definitive statistical study."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate(reference: dict[str, Any], profile: dict[str, Any], oracle_path: Path,
             out: Path, timing_tolerance: float = 0.15) -> dict[str, Any]:
    truth = json.loads(oracle_path.read_text(encoding="utf-8"))["cases"]
    comparable = reference["environment"]["fingerprint"] == profile["environment"]["fingerprint"]
    rows = []
    for cid, b in profile["cases"].items():
        if cid not in reference["cases"] or cid not in truth:
            raise ValueError(f"Missing reference/expectation for {cid}")
        a, gold = reference["cases"][cid], truth[cid]
        at, bt = a["baseline"]["timing"], b["baseline"]["timing"]
        drift = abs(bt["median_ms"] / at["median_ms"] - 1)
        expected, predicted = gold["expected"], b["diagnosis"]["top_diagnosis"]
        same_input = a["baseline"]["matrix"]["input_hash"] == b["baseline"]["matrix"]["input_hash"]
        same_configuration = a["baseline"]["config"] == b["baseline"]["config"]
        timing_ok = (comparable and same_input and same_configuration and drift <= timing_tolerance
                     and at["relative_mad"] <= 0.10 and bt["relative_mad"] <= 0.10)
        detected = predicted == expected
        fix_ok = (expected == "no_supported_issue" or b["confirmation"]["status"] == "improvement_confirmed")
        numerical_ok = a["baseline"]["validation"]["passed"] and b["baseline"]["validation"]["passed"]
        if not a["activated"]:
            status = "not_activated_on_this_host"
        elif not comparable or not same_input or not same_configuration:
            status = "incomparable"
        elif detected and fix_ok and numerical_ok and timing_ok:
            status = "pass"
        elif detected and fix_ok and numerical_ok:
            status = "diagnosis_pass_timing_warning"
        else:
            status = "not_passed"
        rows.append({"id": cid, "name": gold["name"], "expected": expected, "predicted": predicted,
                     "A_ms": at["median_ms"], "B_ms": bt["median_ms"], "relative_timing_delta": drift,
                     "timing_consistent": timing_ok, "same_configuration": same_configuration, "diagnosis_match": detected,
                     "A_reference_speedup": a["comparison"]["speedup"],
                     "C_confirmed_speedup": b["confirmation"].get("median_speedup"),
                     "C_optimized_ms": b["confirmation"].get("median_optimized_ms"),
                     "C_status": b["confirmation"]["status"], "correct": numerical_ok,
                     "activated": a["activated"], "status": status})
    eligible = [r for r in rows if r["activated"]]
    defects = [r for r in eligible if r["expected"] != "no_supported_issue"]
    controls = [r for r in eligible if r["expected"] == "no_supported_issue"]
    summary = {"eligible_cases": len(eligible), "strict_passes": sum(r["status"] == "pass" for r in eligible),
               "matching_top_diagnoses": sum(r["diagnosis_match"] for r in eligible),
               "defect_cases": len(defects), "defects_with_confirmed_improvements": sum(r["C_status"] == "improvement_confirmed" for r in defects),
               "negative_controls": len(controls), "negative_control_false_positives": sum(r["predicted"] not in ("no_supported_issue", "inconclusive") for r in controls),
               "timing_tolerance": timing_tolerance, "same_environment_and_binary": comparable}
    result = {"schema_version": 1, "summary": summary, "cases": rows}
    write_json(out / "evaluation.json", result)
    env = profile["environment"]
    lines = ["# CPU SpMM：A/B 诊断验收结果", "", "## 环境", "",
             f"CPU 报告型号：{env['cpu_model']}；使用 CPU {env['selected_cpus']}。",
             f"虚拟化标志：{env['virtualized_flag']}；CPU 配额：{env['cgroup_quota_cores']} 核；编译器：{env['compiler']}。", "",
             "A 是在本机先冻结的干净计时与已知机制对照；B 独立计时并采集证据；C 是按诊断建议重新运行的确认实验。",
             "A/B 都不包含计数器开销，不拿带 profiler 的慢化时间作为性能排名。", "",
             "| 样例 | A / ms | B / ms | A/B 偏差 | 预期 / 诊断 | C 加速 | 验收 |", "|---|---:|---:|---:|---|---:|---|"]
    for r in rows:
        gain = f"{r['C_confirmed_speedup']:.3f}x" if r["C_confirmed_speedup"] else "—"
        lines.append(f"| {r['id']} {r['name']} | {r['A_ms']:.6f} | {r['B_ms']:.6f} | {100*r['relative_timing_delta']:.1f}% | {r['expected']} / {r['predicted']} | {gain} | {r['status']} |")
    lines += ["", "## 汇总", "", "```json", json.dumps(summary, ensure_ascii=False, indent=2), "```", "",
              "严格通过要求：数值正确、同输入/二进制/环境、A/B 偏差不超过预设阈值、标签命中，并且缺陷案例的改进经新实验确认。",
              "not_activated 表示已知机制在当前参数/硬件上没有形成足够性能差异，不能把该项算作通过。",
              "计时告警与诊断命中单独计数，不把虚拟机上的不稳定计时掩盖成全部成功。", "",
              "## 硬件计数器边界", ""]
    for cid, r in profile["cases"].items():
        lines.append(f"{cid}: " + json.dumps(r["features"]["pmu"], ensure_ascii=False))
    lines += ["", "当前结果只证明所实现的受控机制测试，不证明对任意生产应用的诊断准确率。",
              "细节、建议和未覆盖问题见 profile_report.md；原始观测在 raw_A/raw_B/raw_C，匿名诊断输入在 features/。"]
    (out / "evaluation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "latest")
    parser.add_argument("--threads", type=int, default=None, help="default min(4, available physical cores, CPU quota)")
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--target-ms", type=float, default=35)
    parser.add_argument("--confirm-pairs", type=int, default=3)
    parser.add_argument("--skip-pmu", action="store_true")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build", help="Build the native benchmark with GNU g++ and OpenMP")
    for command in ("run", "calibrate", "profile"):
        p = sub.add_parser(command)
        add_common(p)
        p.add_argument("--suite", type=Path, default=ROOT / "cases" / "suite.json")
        if command != "profile":
            p.add_argument("--oracle", type=Path, default=ROOT / "cases" / "oracle.json")
    p = sub.add_parser("evaluate")
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--profile", type=Path, required=True)
    p.add_argument("--oracle", type=Path, default=ROOT / "cases" / "oracle.json")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--timing-tolerance", type=float, default=0.15)
    p = sub.add_parser("single", help="Profile a Matrix Market matrix without claiming known ground truth")
    add_common(p)
    p.add_argument("--matrix", type=Path, required=True)
    p.add_argument("--n", type=int, default=128)
    p.add_argument("--row-kernel", choices=["simd", "scalar", "gather"], default="simd")
    p.add_argument("--partition", choices=["rows", "nnz"], default="rows")
    p.add_argument("--chunk-rows", type=int, default=0)
    args = parser.parse_args()
    if args.command == "build":
        build(); return 0
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.command == "evaluate":
        result = evaluate(json.loads(args.reference.read_text()), json.loads(args.profile.read_text()),
                          args.oracle, args.out, args.timing_tolerance)
        log(json.dumps(result["summary"], indent=2));return 0
    if args.samples < 3 or args.samples > 101 or not 0 < args.target_ms <= 5000 or args.confirm_pairs < 1:
        parser.error("Use 3..101 samples, target-ms in (0,5000], and >=1 confirmation pairs.")
    build()
    env = environment(args.threads)
    write_json(args.out / "environment.json", env)
    if args.command == "single":
        cfg = {"matrix_file": str(args.matrix.resolve()), "n": args.n, "threads": env["threads"],
               "row_kernel": args.row_kernel, "partition": args.partition, "chunk_rows": args.chunk_rows, "seed": 271}
        profile_cases([{"id": "external", "config": cfg}], env, args.out, args)
        return 0
    cases = resolve_cases(args.suite, env)
    write_json(args.out / "resolved_suite.json", {"cases": cases, "suite_sha256": sha256(args.suite)})
    if args.command in ("run", "calibrate"):
        ref = calibration(cases, env, args.oracle, args.out, args)
    if args.command in ("run", "profile"):
        prof = profile_cases(cases, env, args.out, args)
    if args.command == "run":
        result = evaluate(ref, prof, args.oracle, args.out)
        log(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    log(f"Results: {args.out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError, OSError, KeyError) as exc:
        print(f"spmmprof: {exc}", file=sys.stderr)
        raise SystemExit(2)
