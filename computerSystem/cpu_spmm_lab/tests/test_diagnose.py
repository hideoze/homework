"""Synthetic unit fixtures test rules, and are never reported as measured performance."""
import copy
import unittest
from profiler.diagnose import diagnose
from profiler.measure import probes_for, ratio_ci


def features():
    return {"timing": {"relative_mad": 0.01}, "threads": 4,
            "work_imbalance": 1.0, "active_time_imbalance": 1.0,
            "parallel_regions_per_spmm": 1, "probes": {},
            "assembly": {"status": "ok", "packed_fp_ops": 0, "scalar_fp_ops": 2}}


def probe(gain=2.0):
    return {"speedup": gain, "speedup_ci95": [gain * 0.95, gain * 1.05],
            "relative_mad": 0.01, "correct": True, "same_input": True}


class DiagnosticTests(unittest.TestCase):
    def test_negative_control(self):
        self.assertEqual(diagnose(features())["top_diagnosis"], "no_supported_issue")

    def test_vectorization(self):
        f=features();f["probes"]["vectorize_inner_loop"]=probe()
        d=diagnose(f)
        self.assertEqual(d["top_diagnosis"], "under_vectorization")
        self.assertEqual(d["findings"][0]["confidence"], "supported")

    def test_missing_assembly_is_provisional(self):
        f=features();f["assembly"]={"status":"unavailable"};f["probes"]["vectorize_inner_loop"]=probe()
        self.assertEqual(diagnose(f)["findings"][0]["confidence"], "provisional")

    def test_loop_order(self):
        f=features();f["probes"]["reorder_loops"]=probe(1.7)
        self.assertEqual(diagnose(f)["top_diagnosis"], "loop_order_sensitive")

    def test_load_imbalance(self):
        f=features();f.update(work_imbalance=3.6, active_time_imbalance=3.2)
        f["probes"]["balance_nnz"]=probe(3)
        self.assertEqual(diagnose(f)["top_diagnosis"], "load_imbalance")

    def test_no_intervention_is_not_confirmed(self):
        f=features();f.update(work_imbalance=3.6, active_time_imbalance=3.2)
        d=diagnose(f)
        self.assertEqual(d["findings"][0]["confidence"], "provisional")

    def test_parallel_granularity(self):
        f=features();f["parallel_regions_per_spmm"]=100;f["probes"]["coarsen_regions"]=probe(10)
        self.assertEqual(diagnose(f)["top_diagnosis"], "parallel_region_overhead")

    def test_region_count_required(self):
        f=features();f["probes"]["coarsen_regions"]=probe(10)
        self.assertEqual(diagnose(f)["top_diagnosis"], "no_supported_issue")

    def test_incorrect_faster_result_rejected(self):
        f=features();p=probe();p["correct"]=False;f["probes"]["reorder_loops"]=p
        self.assertEqual(diagnose(f)["top_diagnosis"], "no_supported_issue")

    def test_different_input_rejected(self):
        f=features();p=probe();p["same_input"]=False;f["probes"]["reorder_loops"]=p
        self.assertEqual(diagnose(f)["top_diagnosis"], "no_supported_issue")

    def test_weak_speedup_rejected(self):
        f=features();f["probes"]["reorder_loops"]=probe(1.03)
        self.assertEqual(diagnose(f)["top_diagnosis"], "no_supported_issue")

    def test_confidence_interval_required(self):
        f=features();p=probe();p["speedup_ci95"]=[0.9,3];f["probes"]["reorder_loops"]=p
        self.assertEqual(diagnose(f)["top_diagnosis"], "no_supported_issue")

    def test_noise_causes_abstention(self):
        f=features();f["timing"]["relative_mad"]=0.5
        self.assertEqual(diagnose(f)["top_diagnosis"], "inconclusive")

    def test_control_drift_causes_abstention(self):
        f=features();f["probes"]["repeat_control"]=probe(1.5)
        self.assertEqual(diagnose(f)["top_diagnosis"], "inconclusive")

    def test_no_cache_miss_to_bandwidth_shortcut(self):
        f=features();f["pmu"]={"cache":{"status":"ok","miss_fraction":0.99}}
        self.assertEqual(diagnose(f)["top_diagnosis"], "no_supported_issue")

    def test_no_ground_truth_dependency(self):
        f=features();d=diagnose(f);f["case_id"]="pretend_known_bad_case";f["expected"]="load_imbalance"
        self.assertEqual(diagnose(f), d)

    def test_function_does_not_mutate_input(self):
        f=features();saved=copy.deepcopy(f);diagnose(f);self.assertEqual(f,saved)

    def test_probe_generation_is_program_not_oracle_driven(self):
        cfg={"row_kernel":"scalar","threads":1,"partition":"rows","chunk_rows":0}
        self.assertEqual(set(probes_for(cfg)),{"repeat_control","vectorize_inner_loop"})

    def test_bootstrap_on_constant_measurements(self):
        self.assertEqual(ratio_ci([2,2,2],[1,1,1],100),[2,2])


if __name__=="__main__":
    unittest.main()
