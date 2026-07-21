import os
import shutil
import tempfile
import unittest

from asrq.exp import build_evaluation_output_paths


class EvaluationPathTests(unittest.TestCase):
    def test_build_evaluation_output_paths_creates_results_directory(self) -> None:
        temp_dir = tempfile.mkdtemp(prefix="asrq-exp-paths-")
        try:
            cwd = os.getcwd()
            os.chdir(temp_dir)
            try:
                results_csv, results_cfg = build_evaluation_output_paths(
                    model_name="nvidia/parakeet-ctc-1.1b",
                    method="asrq",
                    quantizer_name="gptq",
                    transform_name="rotation",
                    weight_bits=4,
                    activation_bits=8,
                    timestamp="2026-07-18_08-46-12",
                )
            finally:
                os.chdir(cwd)

            self.assertTrue(os.path.isdir(os.path.join(temp_dir, "results", "evaluations")))
            self.assertTrue(results_csv.endswith("_results.csv"))
            self.assertTrue(results_cfg.endswith("_config.yaml"))
        finally:
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    unittest.main()
