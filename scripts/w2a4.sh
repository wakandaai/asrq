#!/bin/bash
#
# W2A4: 2-bit weights and 4-bit activations, with a learned rotation, scale recovery and weight recovery
# against the previous block's error.
#
# Question: what does the full recipe reach at W2A4? Read it beside
#           scripts/w2a8.sh, scripts/w4a4.sh and scripts/w2a16.sh,
#           which share its recipe and differ only in the weight and activation widths.
#
# Varies:  the model (whisper, parakeet, canary_qwen).
# Fixed:   W2 asymmetric weights in groups of 128, every layer included: the Linears a rotation inserts after
#          each Conformer block's output norm are quantized at 2 bits like the rest. Activations 4-bit
#          symmetric, per token, except attn_out and fc2, which keep one scale per group of 128
#          (activation_groupwise_roles).
#          Rotation: Cayley SGD on R1 and a head-wise R2 per attention module, from random Hadamards, over 800
#          calibration utterances for one epoch at learning rate 0.5, minimising the model's own training loss
#          with the activations fake-quantized at 4 bits. The rotation depends on the activation settings
#          only, not on the weight width, so it is saved as cayley_a4_<model>.pt and shared with every other
#          script that quantizes activations at 4 bits.
#          Quantization: GPTQ on 2048 calibration utterances, with
#            - scale_recovery: a per-input-channel scale on every layer a norm feeds, fitted inside GPTQ and
#              folded into that norm. asrq/experiment.py normally turns it off when the activations are
#              quantized, because the scale is fitted for the weights alone and changes what the activation
#              quantizer rounds; scale_recovery_with_quantized_activations keeps it on here. It reaches the
#              transformer blocks only: Whisper's, and Canary-Qwen's LLM. A rotated Conformer block has no
#              scale recovery targets, so Parakeet gets none.
#            - weight_recovery: each Conformer block's output Linear is refit after the block's other layers
#              are quantized, and then quantized itself. Whisper has no such layer.
#            - layer_weight_recovery: every quantized layer is refit just before it is quantized. It also
#              makes each block quantize its layers in the order they run.
#            - recover_previous_block_error: both refits are fitted against the previous block's output from
#              before that block was quantized, so they also recover the previous block's error.
#          GPTQ and the recoveries run with the activations in full precision: activation quantization is
#          applied while the rotation is learned and at evaluation only.
#          Evaluation: all seven leaderboard datasets, with the activations fake-quantized.
# Output:  outputs/rotation/cayley_a4_<model>.pt, and
#          results/evaluations/w2a4_<model>_w2a4_.../{results.csv,config.yaml}, one directory per model.
#          Every run is logged to wandb in the group w2a4.
#
# A rotation is learned only when its file does not exist yet, so the script can be re-run after an interruption.
# Run a subset with MODELS=whisper bash scripts/w2a4.sh
#
# Usage: bash scripts/w2a4.sh [conda activate script] [project directory]
CONDA_ACTIVATE=${1:-/home/ubuntu/miniconda3/bin/activate}
PROJECT_DIR=${2:-/home/ubuntu/asrq}
source "$CONDA_ACTIVATE" asrq
cd "$PROJECT_DIR"
declare -A BATCH=( [whisper]=64 [parakeet]=128 [canary_qwen]=128 )
# Parakeet's loss is the CTC loss, and ctc_loss_backward has no deterministic CUDA kernel, so its learning
# warns instead of refusing. The initial rotations and the dataloader stay seeded; only the gradient is affected.
declare -A DETERMINISTIC=( [parakeet]="deterministic=false" [whisper]="" [canary_qwen]="" )
stamp() { echo "[$(date +%H:%M:%S)] $*"; }
EXPERIMENT="exp_name=w2a4 wandb.enabled=true"
ACTIVATIONS="activation_bits=4 activation_symmetric=True activation_group_size=128"
GRID="quantizer=gptq quantizer.bits=2 quantizer.symmetric=False quantizer.group_size=128"
RECOVERY="quantizer.scale_recovery=true quantizer.scale_recovery_with_quantized_activations=true"
RECOVERY="$RECOVERY quantizer.weight_recovery=true quantizer.layer_weight_recovery=true"
RECOVERY="$RECOVERY quantizer.recover_previous_block_error=true"

for model in ${MODELS:-whisper parakeet canary_qwen}; do
  rotation=outputs/rotation/cayley_a4_${model}.pt

  # 1. transform: learn the rotation against 4-bit activations
  if [ ! -f $rotation ]; then
    stamp "$model A4 rotation start"
    python -u asrq/rot-exp.py model=$model $GRID $EXPERIMENT $ACTIVATIONS \
      ${DETERMINISTIC[$model]} method=cayley_a4 calibration.num_samples=800 \
      transform=rotation transform.path=$rotation \
      transform.num_samples=800 transform.epochs=1 \
      transform.learning_rate=0.5 2>&1 | tr '\r' '\n'
    stamp "$model A4 rotation end"
  else
    stamp "$model rotation exists at $rotation, reusing it"
  fi

  # 2. apply the rotation, quantize with the recoveries, and evaluate
  stamp "$model W2A4 quantize and evaluate start"
  python -u asrq/exp.py model=$model transform=rotation transform.path=$rotation $GRID $EXPERIMENT \
    $ACTIVATIONS $RECOVERY calibration.num_samples=2048 quantized_path=null \
    model.eval_batch_size=${BATCH[$model]} create_audio_files=False \
    method=w2a4 2>&1 | tr '\r' '\n'
  stamp "$model W2A4 quantize and evaluate end"
done
stamp "done"
