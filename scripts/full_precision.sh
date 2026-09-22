#!/bin/bash
#
# The full-precision models, evaluated on every dataset: no quantization and no transform.
#
# Question: what WER does each model reach before anything is done to it? Every quantized result is read against
#           this one.
#
# Varies:  the model (whisper, parakeet, canary_qwen).
# Fixed:   quantize=false, so nothing is quantized, and transform=none. Weights and activations stay in 16 bits
#          (activation_bits=16). Canary-Qwen's LoRA adapters are merged into the LLM on load, as in every other
#          run. Nothing reads the calibration set, so only 16 utterances are loaded.
#          Evaluation: all seven leaderboard datasets, run in eval_dtype (bfloat16).
# Output:  results/evaluations/full_precision_<model>_fp_none_w16_a16_none_.../{results.csv,config.yaml}, one
#          directory per model. Every run is logged to wandb in the group full_precision.
#
# Run a subset with MODELS=whisper bash scripts/full_precision.sh
#
# Usage: bash scripts/full_precision.sh [conda activate script] [project directory]
CONDA_ACTIVATE=${1:-/home/ubuntu/miniconda3/bin/activate}
PROJECT_DIR=${2:-/home/ubuntu/asrq}
source "$CONDA_ACTIVATE" asrq
cd "$PROJECT_DIR"
declare -A BATCH=( [whisper]=64 [parakeet]=128 [canary_qwen]=128 )
stamp() { echo "[$(date +%H:%M:%S)] $*"; }
EXPERIMENT="exp_name=full_precision wandb.enabled=true"

for model in ${MODELS:-parakeet whisper canary_qwen}; do
  stamp "$model full precision evaluate start"
  python -u asrq/exp.py model=$model transform=none quantize=false \
    $EXPERIMENT activation_bits=16 calibration.num_samples=16 quantized_path=null \
    model.eval_batch_size=${BATCH[$model]} create_audio_files=False method=fp \
    2>&1 | tr '\r' '\n'
  stamp "$model full precision evaluate end"
done
stamp "done"
