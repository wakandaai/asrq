# ASRQ — ASR Quantization

## Supported models

| Config name | Model | Architecture |
|---|---|---|
| `whisper` | `openai/whisper-large-v3` | Transformer encoder-decoder (Hugging Face) |
| `parakeet` | `nvidia/parakeet-ctc-1.1b` | Conformer encoder, CTC head (NeMo) |
| `canary_qwen` | `nvidia/canary-qwen-2.5b` | Conformer encoder + Qwen3 LLM decoder (NeMo SALM, LoRA merged) |

## Installation

```bash
git submodule update --init third_party/open_asr_leaderboard third_party/humming

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install nemo_toolkit[asr]
pip install peft==0.20.0 evaluate==0.4.6 jiwer==4.0.0 datasets==5.0.1 torchcodec==0.16.0 tqdm wandb num2words
pip install --force-reinstall "torchcodec==0.16.0+cpu" --index-url https://download.pytorch.org/whl/cpu
pip install ninja
```

- **`third_party/open_asr_leaderboard`:** supplies the English text normalizer used for WER. Its pinned commit
  decides what counts as a match, so re-run the baselines after bumping it.
- **`third_party/humming`:** a fork of humming (branch `hadamard-sign-seed`). Its kernels generate the random
  Hadamard signs from a seed, and ASRQ's rotation checkpoints store that seed. Install it over the
  `humming-kernels` package. Its kernels are JIT-compiled on first use, which needs `ninja` and a CUDA toolkit.

## Calibration data

Calibration uses LibriSpeech `train.clean.360` audio, paired with each model's own transcripts. Build the set
once per model; the audio is selected on the first run and shared after that:

```bash
python -m asrq.calibration.build --model openai/whisper-large-v3 \
    --calibration-dir outputs/calibration/librispeech_train_clean_360_2048
python -m asrq.calibration.build --model nvidia/parakeet-ctc-1.1b \
    --calibration-dir outputs/calibration/librispeech_train_clean_360_2048
python -m asrq.calibration.build --model nvidia/canary-qwen-2.5b \
    --calibration-dir outputs/calibration/librispeech_train_clean_360_2048
```

`calibration.path` in `asrq/configs/config.yaml` points at this directory. If it's empty, calibration streams
LibriSpeech with its reference text instead.

---

## Workflow

The two entry points share one Hydra config (`asrq/configs/config.yaml`):

1. **`asrq/rot-exp.py`** learns a rotation and saves it to `transform.path`.
2. **`asrq/exp.py`** takes a model through these steps:
   1. loads it;
   2. applies the saved transform;
   3. quantizes it, or loads a previously quantized copy;
   4. optionally converts it to humming kernels;
   5. evaluates it.

The same activation settings (`activation_bits`, `activation_group_size`, `activation_symmetric`,
`activation_groupwise_roles`) and weight settings (`quantizer.*`) are passed to the rotation search and to
evaluation. So a rotation is searched against the quantization it will be evaluated with. Use the same
overrides for both commands.

---

## Tests

```bash
pytest -m "not integration" tests
pytest -m integration tests
```

The integration tests need a GPU and the cached models. They learn a rotation, then quantize and evaluate each
model on 8 utterances.
