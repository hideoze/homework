"""Acceptance gates use synthetic fixtures, not benchmark-performance claims."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from spmmprof import evaluate


class AcceptanceTests(unittest.TestCase):
    def setup_fixture(self):
        env={"fingerprint":"test-host","cpu_model":"unit-test","selected_cpus":[0],"virtualized_flag":True,
             "cgroup_quota_cores":1,"compiler":"fixture"}
        baseline={"timing":{"median_ms":1,"relative_mad":0.01},"matrix":{"input_hash":"same"},
                  "config":{"row_kernel":"simd"},"validation":{"passed":True}}
        a={"environment":env,"cases":{"T":{"baseline":copy.deepcopy(baseline),"activated":True,"comparison":{"speedup":1}}}}
        b={"environment":copy.deepcopy(env),"cases":{"T":{"baseline":copy.deepcopy(baseline),
           "features":{"pmu":{"core":{"status":"unavailable","values":None}}},
           "diagnosis":{"top_diagnosis":"no_supported_issue"},"confirmation":{"status":"not_requested"}}}}
        return a,b

    def evaluate_fixture(self,a,b):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);gold=p/'oracle.json'
            gold.write_text(json.dumps({"cases":{"T":{"expected":"no_supported_issue","name":"test"}}}))
            return evaluate(a,b,gold,p)["cases"][0]

    def test_pass(self):
        self.assertEqual(self.evaluate_fixture(*self.setup_fixture())["status"],"pass")

    def test_changed_host_is_not_comparable(self):
        a,b=self.setup_fixture();b['environment']['fingerprint']='other'
        self.assertEqual(self.evaluate_fixture(a,b)["status"],"incomparable")

    def test_changed_input_is_not_comparable(self):
        a,b=self.setup_fixture();b['cases']['T']['baseline']['matrix']['input_hash']='other'
        self.assertEqual(self.evaluate_fixture(a,b)["status"],"incomparable")

    def test_changed_kernel_is_not_comparable(self):
        a,b=self.setup_fixture();b['cases']['T']['baseline']['config']['row_kernel']='scalar'
        self.assertEqual(self.evaluate_fixture(a,b)["status"],"incomparable")

    def test_timing_drift_is_visible(self):
        a,b=self.setup_fixture();b['cases']['T']['baseline']['timing']['median_ms']=1.2
        self.assertEqual(self.evaluate_fixture(a,b)["status"],"diagnosis_pass_timing_warning")

    def test_not_activated_is_not_a_pass(self):
        a,b=self.setup_fixture();a['cases']['T']['activated']=False
        self.assertEqual(self.evaluate_fixture(a,b)["status"],"not_activated_on_this_host")

    def test_incorrect_is_not_a_pass(self):
        a,b=self.setup_fixture();b['cases']['T']['baseline']['validation']['passed']=False
        self.assertEqual(self.evaluate_fixture(a,b)["status"],"not_passed")


if __name__=='__main__':
    unittest.main()
