#!/bin/bash
#
# ULBQ, weight only, at W4 and W2, with no transform.
#
# Question: how does this project's k-means quantizer compare with RTN and GPTQ at the same widths?
#
# Varies:  the model (whisper, parakeet, canary_qwen) and the weight width (4, 2).
# Fixed:   activations in full precision (activation_bits=16), no transform, no scale recovery and no block
#          output refitting.
#          Quantization: ULBQ -- k-means codebooks with GPTQ-style error compensation, blocks of 128 columns
#          and 1% Hessian damping, on 2048 calibration utterances. It has no group size or symmetry setting,
#          unlike RTN and GPTQ, and at 2 bits it turns on its dynamic outlier columns by itself.
#          Evaluation: all seven leaderboard datasets.
# Output:  results/evaluations/ulbq_<model>_ulbq_w<bits>_.../{results.csv,config.yaml}, one directory per run.
#          Every run is logged to wandb in the group ulbq.
#
# Run a subset with MODELS=whisper BITS=2 bash scripts/ulbq_w4_w2.sh
#
# Usage: bash scripts/ulbq_w4_w2.sh [conda activate script] [project directory]
CONDA_ACTIVATE=${1:-/home/ubuntu/miniconda3/bin/activate}
PROJECT_DIR=${2:-/home/ubuntu/asrq}
source "$CONDA_ACTIVATE" asrq
cd "$PROJECT_DIR"
declare -A BATCH=( [whisper]=64 [parakeet]=128 [canary_qwen]=128 )
stamp() { echo "[$(date +%H:%M:%S)] $*"; }
EXPERIMENT="exp_name=ulbq wandb.enabled=true"

for model in ${MODELS:-parakeet whisper canary_qwen}; do
  for bits in ${BITS:-4 2}; do
    stamp "$model W$bits quantize and evaluate start"
    python -u asrq/exp.py model=$model transform=none \
      quantizer=ulbq quantizer.bits=$bits \
      $EXPERIMENT activation_bits=16 calibration.num_samples=2048 quantized_path=null \
      model.eval_batch_size=${BATCH[$model]} create_audio_files=False method=ulbq_w$bits \
      2>&1 | tr '\r' '\n'
    stamp "$model W$bits quantize and evaluate end"
  done
done
stamp "done"
