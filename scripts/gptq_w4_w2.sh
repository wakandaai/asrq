#!/bin/bash
#
# Plain GPTQ, weight only, at W4 and W2, with no transform at all.
#
# Question: what does GPTQ alone reach on these models? This is the floor the rotation and the recovery steps
#           are measured against, and the only line in the set with nothing applied before quantization.
#
# Varies:  the model (whisper, parakeet, canary_qwen) and the weight width (4, 2).
# Fixed:   groups of 128, W4 symmetric and W2 asymmetric as in the other weight-only runs, activations in full
#          precision (activation_bits=16).
#          Transform: none. Without a rotation there are no inserted block-output Linears, so the Conformer
#          models keep their original output norms and there is nothing to quantize or refit there.
#          Quantization: GPTQ on 2048 calibration utterances and nothing else -- no scale recovery, no block
#          output refitting.
#          Evaluation: all seven leaderboard datasets.
# Output:  results/evaluations/gptq_<model>_gptq_w<bits>_.../{results.csv,config.yaml}, one directory per run.
#          Every run is logged to wandb in the group gptq.
#
# Run a subset with MODELS=whisper BITS=2 bash scripts/gptq_w4_w2.sh
#
# Usage: bash scripts/gptq_w4_w2.sh [conda activate script] [project directory]
CONDA_ACTIVATE=${1:-/home/ubuntu/miniconda3/bin/activate}
PROJECT_DIR=${2:-/home/ubuntu/asrq}
source "$CONDA_ACTIVATE" asrq
cd "$PROJECT_DIR"
declare -A BATCH=( [whisper]=64 [parakeet]=128 [canary_qwen]=128 )
# W2 asymmetric and W4 symmetric, as in the other weight-only runs.
declare -A SYMMETRIC=( [4]=True [2]=False )
stamp() { echo "[$(date +%H:%M:%S)] $*"; }
EXPERIMENT="exp_name=gptq wandb.enabled=true"

for model in ${MODELS:-parakeet whisper canary_qwen}; do
  for bits in ${BITS:-4 2}; do
    stamp "$model W$bits quantize and evaluate start"
    python -u asrq/exp.py model=$model transform=none \
      quantizer=gptq quantizer.bits=$bits quantizer.symmetric=${SYMMETRIC[$bits]} quantizer.group_size=128 \
      $EXPERIMENT activation_bits=16 calibration.num_samples=2048 quantized_path=null \
      model.eval_batch_size=${BATCH[$model]} create_audio_files=False method=gptq_w$bits \
      2>&1 | tr '\r' '\n'
    stamp "$model W$bits quantize and evaluate end"
  done
done
stamp "done"
