#!/bin/bash
#
# GPTQ with the recovery steps, weight only, at W4 and W2, with no transform.
#
# Question: how much of the gap between plain GPTQ and the full recipe do scale recovery and the block output
#           refit close on their own, without a rotation? Compared line for line with gptq_w4_w2.sh, which is
#           the same run with the recovery switched off.
#
# Varies:  the model (whisper, parakeet, canary_qwen) and the weight width (4, 2).
# Fixed:   groups of 128, W4 symmetric and W2 asymmetric as in the other weight-only runs, activations in full
#          precision (activation_bits=16), no transform.
#          Quantization: GPTQ on 2048 calibration utterances with scale recovery and block output refitting on.
#          Without a rotation there is no Linear after a Conformer block's output norm for the refit to use, so
#          the refit finds nothing to do, and scale recovery covers every block instead: the norms of Whisper's
#          encoder and decoder, of Canary-Qwen's LLM, and -- only when unrotated -- the four norms of each
#          Conformer block of Parakeet and of Canary-Qwen's encoder.
#          Evaluation: all seven leaderboard datasets.
# Output:  results/evaluations/gptq_recovery_<model>_gptq_recovery_w<bits>_.../{results.csv,config.yaml}, one
#          directory per run. Every run is logged to wandb in the group gptq_recovery.
#
# Run a subset with MODELS=whisper BITS=2 bash scripts/gptq_recovery_w4_w2.sh
#
# Usage: bash scripts/gptq_recovery_w4_w2.sh [conda activate script] [project directory]
CONDA_ACTIVATE=${1:-/home/ubuntu/miniconda3/bin/activate}
PROJECT_DIR=${2:-/home/ubuntu/asrq}
source "$CONDA_ACTIVATE" asrq
cd "$PROJECT_DIR"
declare -A BATCH=( [whisper]=64 [parakeet]=128 [canary_qwen]=128 )
# W2 asymmetric and W4 symmetric, as in the other weight-only runs.
declare -A SYMMETRIC=( [4]=True [2]=False )
stamp() { echo "[$(date +%H:%M:%S)] $*"; }
EXPERIMENT="exp_name=gptq_recovery wandb.enabled=true"

for model in ${MODELS:-parakeet whisper canary_qwen}; do
  for bits in ${BITS:-4 2}; do
    stamp "$model W$bits quantize and evaluate start"
    python -u asrq/exp.py model=$model transform=none \
      quantizer=gptq quantizer.bits=$bits quantizer.symmetric=${SYMMETRIC[$bits]} quantizer.group_size=128 \
      quantizer.scale_recovery=true quantizer.block_output_refit=true \
      $EXPERIMENT activation_bits=16 calibration.num_samples=2048 quantized_path=null \
      model.eval_batch_size=${BATCH[$model]} create_audio_files=False method=gptq_recovery_w$bits \
      2>&1 | tr '\r' '\n'
    stamp "$model W$bits quantize and evaluate end"
  done
done
stamp "done"
