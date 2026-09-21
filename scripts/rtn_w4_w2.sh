#!/bin/bash
#
# Round to nearest, weight only, at W4 and W2, with no transform at all.
#
# Question: what does the simplest possible quantization give? Every other line in the set -- a rotation, GPTQ,
#           scaling, the recovery steps -- is an improvement over this one, so it fixes the bottom of the range.
#
# Varies:  the model (whisper, parakeet, canary_qwen) and the weight width (4, 2).
# Fixed:   groups of 128, W4 symmetric and W2 asymmetric as in the other weight-only runs, activations in full
#          precision (activation_bits=16).
#          Transform: none. Quantization: RTN, which reads no calibration data at all -- each weight is rounded
#          to its group's grid, so the calibration set is loaded but never used.
#          Evaluation: all seven leaderboard datasets.
# Output:  results/evaluations/rtn_<model>_rtn_w<bits>_.../{results.csv,config.yaml}, one directory per run.
#          Every run is logged to wandb in the group rtn.
#
# Quantization takes minutes rather than hours here, so a run is dominated by the evaluation.
# Run a subset with MODELS=whisper BITS=2 bash scripts/rtn_w4_w2.sh
#
# Usage: bash scripts/rtn_w4_w2.sh [conda activate script] [project directory]
CONDA_ACTIVATE=${1:-/home/ubuntu/miniconda3/bin/activate}
PROJECT_DIR=${2:-/home/ubuntu/asrq}
source "$CONDA_ACTIVATE" asrq
cd "$PROJECT_DIR"
declare -A BATCH=( [whisper]=64 [parakeet]=128 [canary_qwen]=128 )
# W2 asymmetric and W4 symmetric, as in the other weight-only runs.
declare -A SYMMETRIC=( [4]=True [2]=False )
stamp() { echo "[$(date +%H:%M:%S)] $*"; }
EXPERIMENT="exp_name=rtn wandb.enabled=true"

for model in ${MODELS:-parakeet whisper canary_qwen}; do
  for bits in ${BITS:-4 2}; do
    stamp "$model W$bits quantize and evaluate start"
    python -u asrq/exp.py model=$model transform=none \
      quantizer=rtn quantizer.bits=$bits quantizer.symmetric=${SYMMETRIC[$bits]} quantizer.group_size=128 \
      $EXPERIMENT activation_bits=16 calibration.num_samples=16 quantized_path=null \
      model.eval_batch_size=${BATCH[$model]} create_audio_files=False method=rtn_w$bits \
      2>&1 | tr '\r' '\n'
    stamp "$model W$bits quantize and evaluate end"
  done
done
stamp "done"
