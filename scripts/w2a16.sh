#!/bin/bash
#
# W2A16: 2-bit weights, full-precision activations, with an unlearned Hadamard rotation, scale recovery and
# weight recovery against the previous block's error.
#
# Question: how far does 2-bit weight-only quantization get with the full recovery recipe, when the rotation
#           is taken as given rather than learned?
#
# Varies:  the model (whisper, parakeet, canary_qwen).
# Fixed:   W2 asymmetric weights in groups of 128, every layer included: the Linears a rotation inserts after
#          each Conformer block's output norm are quantized at 2 bits like the rest. Activations stay in full
#          precision (activation_bits=16).
#          Rotation: a random Hadamard R1, a random head-wise Hadamard R2 per attention module and a randomized
#          online Hadamard on every fc2 input, saved with transform.epochs=0, so no Cayley step is taken.
#          Nothing could be learned here anyway: every rewrite a rotation makes is exact, so with the
#          activations in full precision the model's output does not depend on the rotation. It earns its place
#          by making the weights easier to round. It is seeded by cfg.seed, and the calibration arguments of
#          that step are kept small because the data is only loaded to verify the rewrite.
#          Quantization: GPTQ on 2048 calibration utterances, with
#            - scale_recovery: a per-input-channel scale on every layer a norm feeds, fitted inside GPTQ and
#              folded into that norm. It reaches the transformer blocks only: Whisper's, and Canary-Qwen's
#              LLM. A rotated Conformer block has no scale recovery targets, so Parakeet gets none.
#            - weight_recovery: each Conformer block's output Linear is refit after the block's other layers
#              are quantized, and then quantized itself. Whisper has no such layer.
#            - layer_weight_recovery: every quantized layer is refit just before it is quantized. It also
#              makes each block quantize its layers in the order they run.
#            - recover_previous_block_error: both refits are fitted against the previous block's output from
#              before that block was quantized, so they also recover the previous block's error.
#          Evaluation: all seven leaderboard datasets.
# Output:  outputs/rotation/hadamard_<model>.pt, and
#          results/evaluations/w2a16_<model>_w2a16_.../{results.csv,config.yaml}, one directory per model.
#          Every run is logged to wandb in the group w2a16.
#
# The rotation is saved once per model and reused, so re-running the script does not redraw it.
# Run a subset with MODELS=whisper bash scripts/w2a16.sh
#
# Usage: bash scripts/w2a16.sh [conda activate script] [project directory]
CONDA_ACTIVATE=${1:-/home/ubuntu/miniconda3/bin/activate}
PROJECT_DIR=${2:-/home/ubuntu/asrq}
source "$CONDA_ACTIVATE" asrq
cd "$PROJECT_DIR"
declare -A BATCH=( [whisper]=64 [parakeet]=128 [canary_qwen]=128 )
# Parakeet's loss is the CTC loss, whose backward has no deterministic CUDA kernel. No step is taken here, but
# the flag costs nothing and keeps the command the same as the scripts that do learn a rotation.
declare -A DETERMINISTIC=( [parakeet]="deterministic=false" [whisper]="" [canary_qwen]="" )
stamp() { echo "[$(date +%H:%M:%S)] $*"; }
EXPERIMENT="exp_name=w2a16 wandb.enabled=true"
GRID="quantizer=gptq quantizer.bits=2 quantizer.symmetric=False quantizer.group_size=128"
RECOVERY="quantizer.scale_recovery=true quantizer.weight_recovery=true quantizer.layer_weight_recovery=true"
RECOVERY="$RECOVERY quantizer.recover_previous_block_error=true"

for model in ${MODELS:-whisper parakeet canary_qwen}; do
  rotation=outputs/rotation/hadamard_${model}.pt

  # 1. transform: draw and save the Hadamard rotation
  if [ ! -f $rotation ]; then
    stamp "$model Hadamard rotation start"
    python -u asrq/rot-exp.py model=$model quantizer=gptq $EXPERIMENT activation_bits=16 \
      ${DETERMINISTIC[$model]} method=hadamard calibration.num_samples=8 \
      transform=rotation transform.path=$rotation \
      transform.num_samples=8 transform.batch_size=4 transform.epochs=0 2>&1 | tr '\r' '\n'
    stamp "$model Hadamard rotation end"
  else
    stamp "$model rotation exists at $rotation, reusing it"
  fi

  # 2. apply the rotation, quantize with the recoveries, and evaluate
  stamp "$model W2A16 quantize and evaluate start"
  python -u asrq/exp.py model=$model transform=rotation transform.path=$rotation $GRID $EXPERIMENT \
    $RECOVERY activation_bits=16 calibration.num_samples=2048 quantized_path=null \
    model.eval_batch_size=${BATCH[$model]} create_audio_files=False \
    method=w2a16 2>&1 | tr '\r' '\n'
  stamp "$model W2A16 quantize and evaluate end"
done
stamp "done"
