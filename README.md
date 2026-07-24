# ASRQ — ASR Quantization Toolkit
ASRQ lets you quantize large pre-trained ASR models to lower bit-widths (2–8 bit) while preserving transcription quality.

---

## Supported Models

| Model | Architecture | Source |
|-------|-------------|--------|
| **Whisper Large-V3** | Attention encoder-decoder | OpenAI / HuggingFace |
| **Parakeet CTC 1.1B** | Conformer CTC | NVIDIA NeMo |
| **Canary-Qwen 2.5B** | Conformer encoder + Qwen3 decoder | NVIDIA NeMo |

## Quantization Methods

| Method | Description |
|--------|-------------|
| **RTN** | Round-to-nearest baseline. No calibration needed. |
| **GPTQ** | Hessian-aware quantization that minimizes quantization-induced loss. |
| **ULBQ** | Ultra-low-bit mixed-precision quantization with K-Means clustering and outlier handling. |

## Pre-Quantization Transforms

Transforms improve weight and activation distributions before quantization:

- **Rotation** — Learns orthogonal rotation matrices (Hadamard-based) to decorrelate weights, followed by LayerNorm → RMSNorm fusion.
- **Shrinking** — Reduces activation range for better quantization.
- **Scaling** — Learns per-layer weight scaling factors.
- **RotScaling** — Combines rotation + scaling for maximum quantization friendliness.

---

## Installation


For NeMo models (Parakeet, Canary):

```bash
pip install "nemo-toolkit[asr]"
```

```bash
pip install "transformers==4.57.6" "datasets==3.6.0" evaluate lhotse soundfile scikit-learn
```


## Quick Start

ASRQ uses [Hydra](https://hydra.cc/) for configuration. Run via the CLI:

```bash
# Quantize Whisper with GPTQ (4-bit, group size 128)
python asrq/exp.py model=whisper quantizer=gptq

# Quantize with RTN (no calibration)
python asrq/exp.py model=whisper quantizer=rtn

# Learn rotation transform
python -m asrq.rot-exp

# Learn rotation transform with local Q2 search
python -m asrq.rot-exp transform.type=search

# Learn rotation transform with alternating Q2 -> Qe -> Q2 search
python -m asrq.rot-exp transform.type=search transform.search_mode=alternating \
  transform.outer_rounds=3 transform.outer_patience=1 transform.qe_min_delta=1e-3 \
  transform.qe_generations=8 transform.q2_refine_generations=6

# Apply rotation transform before quantizing
python asrq/exp.py model=whisper quantizer=gptq transform=rotation


# Override specific parameters
python asrq/exp.py model=whisper quantizer=gptq quantizer.bits=3 quantizer.group_size=64 transform=scaling
```

---

## Project Structure

```
asrq/
├── exp.py                  # Hydra-based CLI entry point
├── rot-exp.py               # Learn Rotation transform
├── configs/
│   ├── config.yaml         # Main config (defaults for model, quantizer, transform)
│   ├── model/              # Per-model configs (whisper, parakeet, canary_qwen)
│   ├── quantizer/          # Per-method configs (rtn, gptq, asrq, ulbq)
│   └── transform/          # Transform configs (rotation, shrinking, scaling, rotscaling)
├── core/
│   ├── model.py            # ModelQ base class — load, quantize, evaluate
│   ├── linear.py           # LinearQ — quantized linear layer with packed weights
│   ├── registry.py         # Decorator-based registry for models, quantizers, transforms
│   ├── types.py            # Type aliases
│   └── utils.py            # CUDA utilities
├── models/
│   ├── whisper.py          # Whisper quantization-aware model
│   ├── parakeet_ctc.py     # Parakeet CTC quantization-aware model
│   └── canary_qwen.py      # Canary-Qwen quantization-aware model
├── quantizers/
│   ├── base.py             # Quantizer base class, HessianAddBatchMixin
│   ├── rtn.py              # Round-to-nearest
│   ├── gptq.py             # GPTQ (Hessian-aware)
│   ├── asrq.py             # ASRQ method
│   └── ulbq.py             # Ultra-low-bit with K-Means
├── transforms/
│   ├── base.py             # Transform base class
│   ├── rotation/           # Rotation transform + Hadamard utilities
│   ├── shrinking.py        # Shrinking transform
│   ├── scaling.py          # Scaling transform
│   └── rotscaling.py       # RotScaling transform
├── calibration/
│   ├── base.py             # Calibration config (LibriSpeech loader)
│   └── data.py             # Custom calibration data
├── kernel/
│   └── gemm.cu  # Custom CUDA kernels for quantization
│   ├── bindings.cpp  # Pybind11 bindings for CUDA kernels
│   └── bindings.py  # Python bindings for CUDA kernels
└── evaluation/
    └── openasr.py          # OpenASR evaluation (WER on ESB datasets)
```

## Pipeline

```
Load Config (Hydra YAML)
        ↓
Load Model + Processor
        ↓
Apply Transforms (rotation, scaling, shrinking)
        ↓
Calibrate (LibriSpeech samples — skipped for RTN - Obtain hessians for GPTQ/ULBQ)
        ↓
Quantize (RTN / GPTQ / ASRQ / ULBQ / SpinQuant / SmoothQuant)
        ↓
Evaluate (WER on OpenASR benchmarks)
```

## Configuration

Configs live in `asrq/configs/` and follow Hydra's hierarchical override system. The main config composes defaults:

```yaml
# config.yaml
defaults:
  - model: whisper
  - quantizer: gptq
  - transform: rotation
```

Override any value from the command line:

```bash
asrq/asrq-exp.py model=parakeet quantizer.bits=2 quantizer.group_size=64 transform=scaling
```

### Alternating Rotation Search

The search-based rotation path now supports two modes:

- `transform.search_mode=local_q2`
  - existing per-layer local `Q2` search using fake-quant block-output NMSE
- `transform.search_mode=alternating`
  - hierarchical `Q2 -> Qe -> Q2` search
  - local `Q2` stages use block-output NMSE
  - global `Qe` stage uses fake-quant task loss
    - Parakeet: CTC loss
    - Whisper: teacher-forced seq2seq cross-entropy

Useful alternating knobs:

```yaml
transform.type: search
transform.search_mode: alternating
transform.outer_rounds: 3
transform.outer_patience: 1
transform.qe_min_delta: 1e-3
transform.qe_generations: 8
transform.q2_refine_generations: 6
```

Alternating search saves:

- `*_search_history.pt`
- `*_search_progress.png`
- `*_global_search_history.pt`
- `*_global_search_progress.png`

## Evaluation

Evaluation runs on the [ESB benchmark](https://huggingface.co/datasets/hf-audio/esb-datasets-test-only-sorted) datasets and reports:

- **WER** — Word Error Rate (%)
- **RTFx** — Real-Time Factor (audio duration / inference time)
