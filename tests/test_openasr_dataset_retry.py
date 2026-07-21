import unittest
from unittest.mock import patch

from datasets import DownloadMode
from datasets.exceptions import NonMatchingSplitsSizesError

from asrq.evaluation.openasr import load_openasr_dataset


class LoadOpenASRDatasetTests(unittest.TestCase):
    def test_retries_with_force_redownload_on_split_size_mismatch(self) -> None:
        expected_dataset = object()
        calls = []

        def fake_load_dataset(*args, **kwargs):
            calls.append((args, kwargs))
            if len(calls) == 1:
                raise NonMatchingSplitsSizesError("split mismatch")
            return expected_dataset

        with patch("asrq.evaluation.openasr.load_dataset", side_effect=fake_load_dataset):
            dataset = load_openasr_dataset(
                dataset_path="hf-audio/esb-datasets-test-only-sorted",
                dataset_name="tedlium",
                split="test",
            )

        self.assertIs(dataset, expected_dataset)
        self.assertEqual(len(calls), 2)
        self.assertNotIn("download_mode", calls[0][1])
        self.assertEqual(calls[1][1]["download_mode"], DownloadMode.FORCE_REDOWNLOAD)

    def test_falls_back_to_fresh_cache_after_failed_force_redownload(self) -> None:
        expected_dataset = object()
        calls = []

        def fake_load_dataset(*args, **kwargs):
            calls.append((args, kwargs))
            if len(calls) < 3:
                raise NonMatchingSplitsSizesError("split mismatch")
            return expected_dataset

        with patch("asrq.evaluation.openasr.load_dataset", side_effect=fake_load_dataset):
            with patch("asrq.evaluation.openasr.tempfile.mkdtemp", return_value="/tmp/asrq-tedlium-fresh-cache"):
                dataset = load_openasr_dataset(
                    dataset_path="hf-audio/esb-datasets-test-only-sorted",
                    dataset_name="tedlium",
                    split="test",
                )

        self.assertIs(dataset, expected_dataset)
        self.assertEqual(len(calls), 3)
        self.assertEqual(calls[1][1]["download_mode"], DownloadMode.FORCE_REDOWNLOAD)
        self.assertEqual(calls[2][1]["download_mode"], DownloadMode.FORCE_REDOWNLOAD)
        self.assertEqual(calls[2][1]["cache_dir"], "/tmp/asrq-tedlium-fresh-cache")


if __name__ == "__main__":
    unittest.main()
