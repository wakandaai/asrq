import os
import shutil
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from asrq.evaluation.base import evaluate_openasr
from asrq.evaluation.openasr import generate_parakeet, generate_whisper


class _DummyModel:
    def __init__(self) -> None:
        self.device = "cuda"

    def to(self, *_args, **_kwargs):
        return self

    def eval(self):
        return self


class _DummyModelQ:
    def __init__(self) -> None:
        self.model = _DummyModel()
        self.processor = None

    def for_activation_quantization(self):
        return []


def _build_cfg() -> SimpleNamespace:
    return SimpleNamespace(
        activation_bits=16,
        method="asrq",
        quantizer=SimpleNamespace(name="gptq", bits=4),
        transform=SimpleNamespace(name="rotation"),
        model=SimpleNamespace(name="nvidia/parakeet-ctc-1.1b", eval_batch_size=1),
    )


class EvaluationAudioFileTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp_dir = tempfile.mkdtemp(prefix="asrq-eval-audio-files-")
        self._cwd = os.getcwd()
        os.chdir(self._temp_dir)

    def tearDown(self) -> None:
        os.chdir(self._cwd)
        shutil.rmtree(self._temp_dir)

    def test_parakeet_forces_audio_file_creation(self) -> None:
        cfg = _build_cfg()
        model_q = _DummyModelQ()

        with patch("asrq.evaluation.base.evaluate_model", return_value={"wer": 0.0}) as mock_eval:
            evaluate_openasr(
                modelQ=model_q,
                cfg=cfg,
                generate_fn=generate_parakeet,
                evaluation_results_file="results.csv",
                create_audio_files=False,
            )

        self.assertTrue(mock_eval.called)
        self.assertTrue(mock_eval.call_args.kwargs["create_audio_files"])

    def test_whisper_keeps_audio_file_creation_disabled(self) -> None:
        cfg = _build_cfg()
        cfg.model.name = "openai/whisper-large-v3"
        model_q = _DummyModelQ()

        with patch("asrq.evaluation.base.evaluate_model", return_value={"wer": 0.0}) as mock_eval:
            evaluate_openasr(
                modelQ=model_q,
                cfg=cfg,
                generate_fn=generate_whisper,
                evaluation_results_file="results.csv",
                create_audio_files=False,
            )

        self.assertTrue(mock_eval.called)
        self.assertFalse(mock_eval.call_args.kwargs["create_audio_files"])


if __name__ == "__main__":
    unittest.main()
