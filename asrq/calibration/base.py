# pyright: reportMissingImports=false
from asrq.transforms.base import TransformConfig
from omegaconf import DictConfig



class CalibConfig:
    def __init__(self, cfg: DictConfig) -> None:
        self.num_samples = cfg.num_samples
        # A calibration set written by asrq.calibration.data. When set, GPTQ and rotation
        # learning both read their audio from it, teacher-forced with the model's own
        # transcripts; when empty, LibriSpeech reference text is streamed as before.
        self.path = cfg.get("path", None)
        # "transcript" for the model's own transcripts, "reference" for LibriSpeech text.
        self.text_source = cfg.get("text_source", "transcript")

    @staticmethod
    def from_dictconfig(cfg: DictConfig) -> "CalibConfig":
        return CalibConfig(cfg)
    