

# Model Names
# Specify the huggingface model names of supported ASR models
from dataclasses import dataclass, fields


@dataclass
class ModelNames:
    OPENAI_WHISPER_LARGE_V3: str = "openai/whisper-large-v3"
    NVIDIA_CANARY_QWEN_2_5B: str = "nvidia/canary-qwen-2.5b"
    NVIDIA_CANARY_1B: str = "nvidia/canary-1b"
    KYUTAI_STT_2_6B_EN: str = "kyutai/stt-2.6b-en"
    IBM_GRANITE_SPEECH_3_3_8B: str = "ibm-granite/granite-speech-3.3-8b"
    NVIDIA_PARAKEET_CTC_1_1B: str = "nvidia/parakeet-ctc-1.1b"
    NVIDIA_PARAKEET_RNNT_1_1B: str = "nvidia/parakeet-rnnt-1.1b"
    NVIDIA_PARAKEET_TDT_1_1B: str = "nvidia/parakeet-tdt-1.1b"
    
    @classmethod
    def values(cls):
        return [
            field.default for field in fields(cls)
        ]

# ========== Model Registry ==========
ModelQ_Registry = {
    
}

def register_model(cls_name):
    def decorator(cls):
        if cls_name in ModelQ_Registry:
            raise ValueError(f"Class name {cls_name} already registered.")
        ModelQ_Registry[cls_name] = cls
        return cls
    return decorator


# ========== Quantizer Registry ==========
@dataclass
class QuantizerNames:
    RTN: str = "rtn"
    GPTQ: str = "gptq"
    ASRQ: str = "asrq"
    ULBQ: str = "ulbq"

    @classmethod
    def values(cls):
        return [
            field.default for field in fields(cls)
        ]


Quantizer_Registry = {}

def register_quantizer(name):
    def decorator(cls):
        Quantizer_Registry[name] = cls
        return cls
    return decorator


def get_quant_cls(name):
    if name not in Quantizer_Registry:
        raise ValueError(f"Quantization method '{name}' not found in registry.")
    return Quantizer_Registry[name]


# ========== Quantizer Config Registry ==========
QuantizerConfig_Registry = {}

def register_quantizer_config(name):
    def decorator(cls):
        QuantizerConfig_Registry[name] = cls
        return cls
    return decorator


def get_quantizer_config_cls(name):
    if name not in QuantizerConfig_Registry:
        raise ValueError(f"Quantizer config '{name}' not found in registry.")
    return QuantizerConfig_Registry[name]


# ========== Transform Registry ==========
Transform_Registry = {}
Transform_Config_Registry = {}

class TransformNames:
    rotation: str = "rotation"
    scaling: str = "scaling"
    shrinking: str = "shrinking"
    rotscaling: str = "RotScaling"

def register_transform(name):
    def decorator(cls):
        Transform_Registry[name] = cls
        return cls
    return decorator

def get_transform_cls(name):
    if name not in Transform_Registry:
        raise ValueError(f"Transform '{name}' not found in registry.")
    return Transform_Registry[name]

def register_transform_config(name):
    def decorator(cls):
        Transform_Config_Registry[name] = cls
        return cls
    return decorator

def get_transform_config_cls(name):
    if name not in Transform_Config_Registry:
        raise ValueError(f"Transform config '{name}' not found in registry.")
    return Transform_Config_Registry[name]


# ========== Calibration Registry ==========
Calib_Registry = {}
Calib_Config_Registry = {}


