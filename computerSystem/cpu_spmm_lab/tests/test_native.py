"""End-to-end native correctness/coverage tests, not timing assertions."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from profiler.measure import ROOT, BIN, build, environment


class NativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        build()
        cls.env = environment()
        cls.threads = min(2,cls.env["threads"])
        cls.child = os.environ.copy()
        cls.child.update(OMP_PROC_BIND="TRUE", OMP_DYNAMIC="FALSE", OMP_WAIT_POLICY="PASSIVE",
                         OMP_PLACES=",".join("{"+str(c)+"}" for c in cls.env["selected_cpus"]))

    def run_case(self,*args,expect_ok=True):
        cmd=[str(BIN),"--m","17","--k","23","--n","31","--degree","5", "--samples","3","--target-ms","1",*args]
        p=subprocess.run(cmd,cwd=ROOT,env=self.child,text=True,capture_output=True,timeout=30)
        if not expect_ok:
            self.assertNotEqual(p.returncode,0);return None
        self.assertEqual(p.returncode,0,p.stderr)
        r=json.loads(p.stdout)
        self.assertTrue(r["validation"]["passed"])
        return r

    def test_three_kernels_same_input_and_correct(self):
        runs=[self.run_case("--row-kernel",k) for k in ("scalar","simd","gather")]
        self.assertEqual(len({r["matrix"]["input_hash"] for r in runs}),1)
        for r in runs:
            self.assertGreater(r["timing"]["median_ms"],0)
            self.assertLessEqual(r["validation"]["max_scaled_error"],1)

    def test_feature_width_tails(self):
        for n in (1,3,7,16,65,129):
            with self.subTest(n=n):
                self.run_case("--n",str(n))

    def test_thread_and_work_coverage(self):
        for partition in ("rows","nnz"):
            r=self.run_case("--threads",str(self.threads),"--action","profile","--partition",partition,"--matrix","skew")
            self.assertEqual(sum(x["nnz"] for x in r["workers"]),r["matrix"]["nnz"]*r["profile_iters"])
            self.assertEqual(sum(x["rows"] for x in r["workers"]),17*r["profile_iters"])

    def test_tiny_chunks_and_empty_worker_ranges(self):
        r=self.run_case("--threads",str(self.threads),"--chunk-rows","1","--action","profile")
        self.assertEqual(r["work"]["parallel_regions_per_spmm"],17)
        self.assertEqual(sum(x["rows"] for x in r["workers"]),17*r["profile_iters"])

    def test_banded(self):
        self.run_case("--matrix","banded")

    def test_pmu_unavailable_is_null_or_real_counts(self):
        r=self.run_case("--action","profile","--events","core","--threads",str(self.threads))
        for w in r["workers"]:
            p=w["pmu"]
            if p["status"] in ("ok","low_coverage"):
                self.assertGreater(p["running_ns"],0)
                self.assertEqual(len(p["scaled"]),2)
            else:
                self.assertIsNone(p["scaled"])
                self.assertIsNone(p["raw"])
                self.assertTrue(p["reason"])

    def test_pmu_never_mixed_into_benchmark(self):
        self.run_case("--events","core",expect_ok=False)

    def test_invalid_dimensions(self):
        self.run_case("--m","-1",expect_ok=False)

    def test_invalid_kernel(self):
        self.run_case("--row-kernel","does_not_exist",expect_ok=False)

    def test_matrix_market_duplicates_empty_row(self):
        text="%%MatrixMarket matrix coordinate real general\n% duplicate entries cancel\n4 5 5\n1 1 2\n1 1 -2\n2 4 3\n4 2 -1\n4 5 4\n"
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"a.mtx";p.write_text(text)
            r=self.run_case("--matrix-file",str(p),"--n","7")
            self.assertEqual(r["matrix"]["nnz"],3)
            self.assertEqual(r["config"]["m"],4)
            self.assertEqual(r["config"]["k"],5)

    def test_matrix_market_symmetric_pattern(self):
        text="%%MatrixMarket matrix coordinate pattern symmetric\n3 3 3\n1 1\n2 1\n3 2\n"
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"a.mtx";p.write_text(text)
            r=self.run_case("--matrix-file",str(p))
            self.assertEqual(r["matrix"]["nnz"],5)

    def test_empty_matrix(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"a.mtx";p.write_text("%%MatrixMarket matrix coordinate real general\n3 5 0\n")
            r=self.run_case("--matrix-file",str(p),"--partition","nnz","--threads",str(self.threads),"--action","profile")
            self.assertEqual(r["matrix"]["nnz"],0)
            self.assertEqual(r["validation"]["checksum"],0)

    def test_invalid_matrix_market(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"a.mtx";p.write_text("%%MatrixMarket matrix coordinate real general\n2 2 1\n9 1 2\n")
            self.run_case("--matrix-file",str(p),expect_ok=False)


if __name__=="__main__":
    unittest.main()
