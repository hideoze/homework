"""Linux/GCC measurement adapter. All timing runs are separate from PMU runs."""
from __future__ import annotations
import hashlib
import json
import math
import os
import platform
import random
import re
import shutil
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "build" / "spmm_bench"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def read_text(path: str | Path) -> str | None:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build() -> None:
    if not shutil.which("g++") or not shutil.which("make"):
        raise RuntimeError("Requires GNU g++, GNU make and the OpenMP runtime (Linux).")
    subprocess.run(["make", "-j2"], cwd=ROOT, check=True)


def cgroup_stat() -> dict[str, int]:
    # cgroup-v2 mount root, the usual container layout. Other layouts are recorded as unavailable.
    text = read_text("/sys/fs/cgroup/cpu.stat") or ""
    result = {}
    for line in text.splitlines():
        try:
            k, v = line.split()
            result[k] = int(v)
        except ValueError:
            pass
    return result


def environment(requested_threads: int | None = None) -> dict[str, Any]:
    allowed = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else list(range(os.cpu_count() or 1))
    physical, seen = [], set()
    for cpu in allowed:
        base = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        package, core = read_text(base / "physical_package_id"), read_text(base / "core_id")
        key = (package, core) if package is not None and core is not None else ("unknown", str(cpu))
        if key not in seen:
            physical.append(cpu)
            seen.add(key)
    cap = len(physical)
    quota_raw = read_text("/sys/fs/cgroup/cpu.max")
    quota_cores = None
    if quota_raw:
        parts = quota_raw.split()
        if len(parts) == 2 and parts[0] != "max":
            quota_cores = int(parts[0]) / int(parts[1])
            cap = min(cap, max(1, math.floor(quota_cores)))
    default_threads = max(1, min(4, cap))
    threads = requested_threads or default_threads
    if threads < 1 or threads > cap:
        raise ValueError(f"Requested {threads} threads; usable physical-core/quota cap is {cap}.")
    cpus = physical[:threads]
    cpuinfo = read_text("/proc/cpuinfo") or ""
    model = next((s.split(":", 1)[1].strip() for s in cpuinfo.splitlines() if s.startswith("model name")), platform.processor())
    flags = next((s for s in cpuinfo.splitlines() if s.startswith("flags")), "")
    caches = []
    for p in sorted(Path(f"/sys/devices/system/cpu/cpu{cpus[0]}/cache").glob("index*")):
        caches.append({k: read_text(p / k) for k in ("level", "type", "size", "shared_cpu_list")})
    env = {
        "os": platform.platform(), "machine": platform.machine(), "cpu_model": model,
        "virtualized_flag": "hypervisor" in flags.split(), "allowed_cpus": allowed,
        "selected_cpus": cpus, "threads": threads, "cgroup_quota_cores": quota_cores,
        "cgroup_cpu_max": quota_raw, "perf_event_paranoid": read_text("/proc/sys/kernel/perf_event_paranoid"),
        "compiler": subprocess.check_output(["g++", "--version"], text=True).splitlines()[0],
        "binary_sha256": sha256(BIN), "cache_topology_reported": caches,
        "memory_policy": "default OS policy; initialization is serial; no NUMA binding is implemented",
        "pinning": "one selected logical CPU per reported physical core; OMP_PLACES explicit",
    }
    env["fingerprint"] = hashlib.sha256(json.dumps(env, sort_keys=True).encode()).hexdigest()
    return env


def resolve_cases(suite: Path, env: dict[str, Any]) -> list[dict[str, Any]]:
    cases = json.loads(suite.read_text(encoding="utf-8"))["cases"]
    if len({c["id"] for c in cases}) != len(cases):
        raise ValueError("Case IDs must be unique.")
    for case in cases:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", case["id"]):
            raise ValueError("Case IDs must be safe directory names.")
        cfg = case["config"]
        if cfg.get("threads") == "parallel":
            cfg["threads"] = env["threads"]
        if not 1 <= int(cfg.get("threads", 1)) <= len(env["selected_cpus"]):
            raise ValueError("A case requests more threads than the selected affinity allows.")
    return cases


def measure(config: dict[str, Any], env: dict[str, Any], path: Path,
            *, action: str = "bench", events: str = "none", samples: int = 7,
            target_ms: float = 35, profile_iters: int = 5) -> dict[str, Any]:
    args = [str(BIN)]
    for key, value in config.items():
        args.extend(["--" + key.replace("_", "-"), str(value)])
    args += ["--action", action, "--events", events, "--samples", str(samples),
             "--target-ms", str(target_ms), "--profile-iters", str(profile_iters)]
    child_env = os.environ.copy()
    cpus = env["selected_cpus"]
    child_env.update(OMP_NUM_THREADS=str(config.get("threads", 1)), OMP_DYNAMIC="FALSE",
                     OMP_PROC_BIND="TRUE", OMP_PLACES=",".join("{" + str(c) + "}" for c in cpus),
                     OMP_WAIT_POLICY="PASSIVE")
    before = cgroup_stat()
    start = time.perf_counter()
    proc = subprocess.run(args, cwd=ROOT, env=child_env, capture_output=True, text=True, timeout=240)
    elapsed = (time.perf_counter() - start) * 1000
    after = cgroup_stat()
    if proc.returncode:
        raise RuntimeError(f"Benchmark failed ({proc.returncode}): {proc.stderr[-3000:]}")
    data = json.loads(proc.stdout)
    if not data["validation"]["passed"]:
        raise RuntimeError("Rejected numerically incorrect benchmark result.")
    data["provenance"] = {
        "command": args, "binary_sha256": env["binary_sha256"], "environment_fingerprint": env["fingerprint"],
        "omp_env": {k: child_env[k] for k in ("OMP_NUM_THREADS", "OMP_DYNAMIC", "OMP_PROC_BIND", "OMP_PLACES", "OMP_WAIT_POLICY")},
        "whole_process_ms": elapsed, "stderr": proc.stderr,
        "cgroup_delta_whole_process_not_roi": {k: after[k] - before.get(k, after[k]) for k in after},
    }
    write_json(path, data)
    return data


def assembly(row_kernel: str) -> dict[str, Any]:
    if platform.machine() not in ("x86_64", "AMD64") or not shutil.which("objdump"):
        return {"status": "unavailable", "reason": "Static instruction parser currently supports x86 objdump only."}
    symbol = {"scalar": "row_scalar", "simd": "row_simd", "gather": "row_gather"}[row_kernel]
    text = subprocess.check_output(["objdump", "-d", "--no-show-raw-insn", "--disassemble=" + symbol, str(BIN)], text=True)
    (ROOT / "build" / (symbol + ".asm")).write_text(text)
    packed = re.findall(r"\b(?:v?(?:add|sub|mul|div)p[sd]|v(?:fmadd|fmsub|fnmadd|fnmsub)\d*p[sd])\b", text)
    scalar = re.findall(r"\b(?:v?(?:add|sub|mul|div)s[sd]|v(?:fmadd|fmsub|fnmadd|fnmsub)\d*s[sd])\b", text)
    return {"status": "ok" if packed or scalar else "unavailable", "packed_fp_ops": len(packed),
            "scalar_fp_ops": len(scalar), "scope": "static row-function instruction forms, NOT dynamic instruction counts"}


def ratio_ci(base: list[float], trial: list[float], n: int = 1200) -> list[float]:
    """Bootstrap screening interval over batched samples; not a cross-machine guarantee."""
    rng = random.Random(1717)
    ratios = sorted(statistics.median(rng.choices(base, k=len(base))) /
                    statistics.median(rng.choices(trial, k=len(trial))) for _ in range(n))
    return [ratios[int(0.025*n)], ratios[min(n-1, int(0.975*n))]]


def compare(base: dict[str, Any], trial: dict[str, Any]) -> dict[str, Any]:
    bt, tt = base["timing"], trial["timing"]
    return {
        "speedup": bt["median_ms"] / tt["median_ms"],
        "speedup_ci95": ratio_ci(bt["samples_ms"], tt["samples_ms"]),
        "relative_mad": tt["relative_mad"],
        "correct": bool(base["validation"]["passed"] and trial["validation"]["passed"]),
        "same_input": base["matrix"]["input_hash"] == trial["matrix"]["input_hash"],
        "baseline_ms": bt["median_ms"], "trial_ms": tt["median_ms"],
    }


def probes_for(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Mechanism-preserving adapter transformations, no ground-truth access."""
    probes: dict[str, dict[str, Any]] = {"repeat_control": {}}
    if config["row_kernel"] == "scalar":
        probes["vectorize_inner_loop"] = {"row_kernel": "simd"}
    if config["row_kernel"] == "gather":
        probes["reorder_loops"] = {"row_kernel": "scalar"}
    if config.get("threads", 1) > 1 and config.get("partition", "rows") == "rows":
        probes["balance_nnz"] = {"partition": "nnz"}
    if config.get("chunk_rows", 0) > 0:
        probes["coarsen_regions"] = {"chunk_rows": 0}
    return probes


def pmu_summary(pass_data: dict[str, Any]) -> dict[str, Any]:
    workers = pass_data["workers"]
    status = [w["pmu"]["status"] for w in workers]
    if not workers or any(s not in ("ok", "low_coverage") for s in status):
        return {"status": "unavailable", "values": None,
                "reasons": sorted({w["pmu"]["reason"] for w in workers if w["pmu"]["reason"]})}
    scaled = [sum(w["pmu"]["scaled"][i] for w in workers) for i in range(2)]
    names = {"core": ["cycles", "instructions"], "cache": ["generic_cache_references", "generic_cache_misses"],
             "branch": ["branches", "branch_misses"]}[pass_data["event_group"]]
    ratio = min(w["pmu"]["running_ns"] / w["pmu"]["enabled_ns"] for w in workers)
    return {"status": "ok" if ratio >= 0.9 else "low_coverage", "values": dict(zip(names, scaled)),
            "minimum_running_enabled_ratio": ratio,
            "scope": "user-mode row loops, summed across workers and profile repetitions; barriers excluded",
            "ratio": (scaled[1] / scaled[0]) if scaled[0] else None,
            "ratio_name": "IPC" if pass_data["event_group"] == "core" else "miss_fraction",
            "warning": "Generic cache misses are not universally LLC misses, and do not measure DRAM bytes."}


def make_features(base: dict[str, Any], profile: dict[str, Any], probes: dict[str, Any],
                  asm: dict[str, Any], pmu: dict[str, Any]) -> dict[str, Any]:
    workers = profile["workers"]
    work = [w["nnz"] for w in workers]
    active = [w["active_wall_ms"] for w in workers]
    expected_work = profile["matrix"]["nnz"] * profile["profile_iters"]
    if sum(work) != expected_work:
        raise RuntimeError("Worker nnz totals do not match the exact sparse workload.")
    expected_rows = profile["config"]["m"] * profile["profile_iters"]
    if sum(w["rows"] for w in workers) != expected_rows:
        raise RuntimeError("Worker row coverage is inconsistent.")
    features = {
        "timing": base["timing"], "threads": base["config"]["threads"],
        "parallel_regions_per_spmm": base["work"]["parallel_regions_per_spmm"],
        "work_imbalance": max(work) / statistics.mean(work) if sum(work) else 1.0,
        "active_time_imbalance": max(active) / statistics.mean(active) if sum(active) else 1.0,
        "row_nnz_cv": base["matrix"]["row_nnz_cv"], "work_coverage_checked": True,
        "assembly": asm, "probes": probes, "pmu": pmu,
        "environment": {"throttled_during_measurement": base["provenance"]["cgroup_delta_whole_process_not_roi"].get("nr_throttled", 0) > 0},
        "scope_notes": ["Worker active times exclude final region barriers; elapsed-active is not measured barrier wait.",
                        "Cgroup flags cover the whole process, not just its kernel timing boundary."],
    }
    # Intentionally omit IDs, generator names, kernel names and oracle labels.
    return features
