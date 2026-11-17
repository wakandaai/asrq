#!/bin/bash
#
# W2A16 with the scaling transform in place of a rotation: smoothing scales, then GPTQ with scale recovery
# and weight recovery against the previous block's error.
#
# Question: what does the recovery recipe reach at W2A16 when the transform is channel scaling and not a
#           rotation? Read it beside scripts/w2a16.sh, the same widths and recoveries behind a rotation.
#
# Varies:  the model (whisper, parakeet, canary_qwen).
# Fixed:   W2 asymmetric weights in groups of 128. Activations stay in full precision (activation_bits=16).
#          Transform: smoothquant scales with one migration strength for every entry, alpha = 0:
#          s = amax^alpha / wmax^(1 - alpha), from the per-channel input maxima over the calibration
#          utterances and the per-channel weight maxima. Nothing is learned. With alpha = 0 the scale is
#          s = 1 / wmax: it depends on the weights alone and gives every input channel of a weight the same
#          range, which is all that matters when only the weights are quantized.
#          The scales are folded in: into the previous layer's rows where one exists, and into an InputScale
#          module after the activation where a nonlinearity sits in between (fc2, pointwise_conv2).
#          Quantization: GPTQ on 2048 calibration utterances, with
#            - scale_recovery: a per-input-channel scale on every layer a norm feeds, fitted inside GPTQ and
#              folded into that norm. Without a rotation it reaches every model: Whisper's blocks,
#              Canary-Qwen's LLM and Conformer encoder, and Parakeet's Conformer blocks.
#            - layer_weight_recovery: every quantized layer is refit just before it is quantized. It also
#              makes each block quantize its layers in the order they run.
#            - recover_previous_block_error: the refit is fitted against the previous block's output from
#              before that block was quantized, so it also recovers the previous block's error.
#            - weight_recovery is passed for the same recipe as the rotation scripts, and has nothing to act
#              on: the block output Linear it refits only exists once a rotation has inserted it.
#          The activations are in full precision throughout.
#          Evaluation: all seven leaderboard datasets.
# Output:  outputs/scaling/scaling_alpha0_<model>.pt, and
#          results/evaluations/scaling_w2a16_<model>_scaling_w2a16_.../{results.csv,config.yaml}, one
#          directory per model. Every run is logged to wandb in the group scaling_w2a16.
#
# A scale file that already exists is reused, so the script can be re-run after an interruption.
# Run a subset with MODELS=whisper bash scripts/scaling_w2a16.sh
#
# Usage: bash scripts/scaling_w2a16.sh [conda activate script] [project directory]
CONDA_ACTIVATE=${1:-/home/ubuntu/miniconda3/bin/activate}
PROJECT_DIR=${2:-/home/ubuntu/asrq}
source "$CONDA_ACTIVATE" asrq
cd "$PROJECT_DIR"
mkdir -p outputs/scaling
declare -A BATCH=( [whisper]=64 [parakeet]=128 [canary_qwen]=128 )
stamp() { echo "[$(date +%H:%M:%S)] $*"; }
EXPERIMENT="exp_name=scaling_w2a16 wandb.enabled=true"
ACTIVATIONS="activation_bits=16 activation_symmetric=True activation_group_size=128"
GRID="quantizer=gptq quantizer.bits=2 quantizer.symmetric=False quantizer.group_size=128"
RECOVERY="quantizer.scale_recovery=true quantizer.weight_recovery=true"
RECOVERY="$RECOVERY quantizer.layer_weight_recovery=true quantizer.recover_previous_block_error=true"

for model in ${MODELS:-whisper parakeet canary_qwen}; do
  scales=outputs/scaling/scaling_alpha0_${model}.pt
  if [ -f $scales ]; then
    obtain=transform.obtain_scales=false
    stamp "$model scales exist at $scales, reusing them"
  else
    obtain=transform.obtain_scales=true
  fi

  # obtain (or load) the scales, apply them, quantize with the recoveries, and evaluate
  stamp "$model W2A16 scale, quantize and evaluate start"
  python -u asrq/exp.py model=$model transform=scaling transform.path=$scales $obtain \
    transform.type=smoothquant transform.alpha=0 $GRID $EXPERIMENT \
    $ACTIVATIONS $RECOVERY calibration.num_samples=2048 quantized_path=null \
    model.eval_batch_size=${BATCH[$model]} create_audio_files=False \
    method=scaling_w2a16 2>&1 | tr '\r' '\n'
  stamp "$model W2A16 scale, quantize and evaluate end"
done
stamp "done"
