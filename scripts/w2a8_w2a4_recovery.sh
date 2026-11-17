#!/bin/bash
#
# W2A8 and W2A4 with and without weight recovery, on LibriSpeech test-clean and test-other.
#
# Question: what does weight recovery buy at 2-bit weights once the activations are quantized too, and does
#           recovering the previous block's error (quantizer.recover_previous_block_error) add to it?
#
# Varies:  the model (parakeet, canary_qwen, whisper), the activation width (8, 4) and the recovery:
#            - none:     plain GPTQ.
#            - layer:    weight_recovery and layer_weight_recovery. Every layer is refit just before it is
#                        quantized, and each Conformer block's output Linear is refit after the block.
#            - previous: the same, with recover_previous_block_error: the refits are fitted against the
#                        previous block's output from before that block was quantized.
#            - scale:    layer, plus scale recovery kept on although the activations are quantized
#                        (scale_recovery_with_quantized_activations). Compare it with layer. It reaches the
#                        transformer blocks only: Whisper's, and Canary-Qwen's LLM. A rotated Conformer block
#                        has no scale recovery targets, so on Parakeet this variant is the same run as layer.
# Fixed:   W2 asymmetric weights in groups of 128, every layer included: the Linears a rotation inserts after
#          each Conformer block's output norm are quantized at 2 bits like the rest. Activations symmetric,
#          per token, except attn_out and fc2, which keep one scale per group of 128.
#          Rotation: Cayley SGD on R1 and a head-wise R2 per attention module, from random Hadamards, over 800
#          calibration utterances for one epoch at learning rate 0.5, learned once per model and activation
#          width with the activations fake-quantized, and shared by the three recovery variants.
#          Quantization: GPTQ on 2048 calibration utterances. Scale recovery is used only by the scale
#          variant: asrq/experiment.py otherwise turns it off when the activations are quantized.
#          Evaluation: LibriSpeech test.clean and test.other, with the activations fake-quantized. Set
#          ALL_SPLITS=1 for all seven leaderboard datasets.
# Output:  outputs/rotation/cayley_a<bits>_<model>.pt (shared with w2a8.sh, w2a4.sh and w4a4.sh), and
#          results/evaluations/w2_recovery_<model>_w2a<bits>_<recovery>_.../{results.csv,config.yaml}, one
#          directory per run. Every run is logged to wandb in the group w2_recovery.
#
# Note: GPTQ and both recoveries run with the activations in full precision. Activation quantization is only
#       applied while the rotation is learned and at evaluation, so the refits correct weight error and do not
#       see the activation quantizer.
#
# A rotation is learned only when its file does not exist yet, so the script can be re-run after an interruption.
# Run a subset with MODELS=whisper ABITS=8 RECOVERIES="none previous" bash scripts/w2a8_w2a4_recovery.sh
#
# Usage: bash scripts/w2a8_w2a4_recovery.sh [conda activate script] [project directory]
CONDA_ACTIVATE=${1:-/home/ubuntu/miniconda3/bin/activate}
PROJECT_DIR=${2:-/home/ubuntu/asrq}
source "$CONDA_ACTIVATE" asrq
cd "$PROJECT_DIR"
declare -A BATCH=( [whisper]=64 [parakeet]=128 [canary_qwen]=128 )
# Parakeet's loss is the CTC loss, and ctc_loss_backward has no deterministic CUDA kernel, so its learning
# warns instead of refusing. The initial rotations and the dataloader stay seeded; only the gradient is affected.
declare -A DETERMINISTIC=( [parakeet]="deterministic=false" [whisper]="" [canary_qwen]="" )
declare -A RECOVERY=(
  [none]=""
  [layer]="quantizer.weight_recovery=true quantizer.layer_weight_recovery=true"
  [previous]="quantizer.weight_recovery=true quantizer.layer_weight_recovery=true quantizer.recover_previous_block_error=true"
  [scale]="quantizer.weight_recovery=true quantizer.layer_weight_recovery=true quantizer.scale_recovery=true quantizer.scale_recovery_with_quantized_activations=true"
)
stamp() { echo "[$(date +%H:%M:%S)] $*"; }
EXPERIMENT="exp_name=w2_recovery wandb.enabled=true"
ACTIVATIONS="activation_symmetric=True activation_group_size=128"
GRID="quantizer=gptq quantizer.bits=2 quantizer.symmetric=False quantizer.group_size=128"
SPLITS=("eval_datasets=[{dataset:librispeech,split:test.clean},{dataset:librispeech,split:test.other}]")
if [ -n "$ALL_SPLITS" ]; then SPLITS=(); fi

for model in ${MODELS:-parakeet whisper canary_qwen}; do
  for abits in ${ABITS:-8 4}; do
    rotation=outputs/rotation/cayley_a${abits}_${model}.pt

    if [ ! -f $rotation ]; then
      stamp "$model W2A$abits rotation start"
      python -u asrq/rot-exp.py model=$model $GRID $EXPERIMENT $ACTIVATIONS activation_bits=$abits \
        ${DETERMINISTIC[$model]} method=cayley_a$abits calibration.num_samples=800 \
        transform=rotation transform.path=$rotation \
        transform.num_samples=800 transform.epochs=1 \
        transform.learning_rate=0.5 2>&1 | tr '\r' '\n'
      stamp "$model W2A$abits rotation end"
    else
      stamp "$model W2A$abits rotation exists, skipping learning"
    fi

    for recovery in ${RECOVERIES:-none layer previous scale}; do
      stamp "$model W2A$abits $recovery start"
      python -u asrq/exp.py model=$model transform=rotation transform.path=$rotation $GRID $EXPERIMENT \
        $ACTIVATIONS activation_bits=$abits ${RECOVERY[$recovery]} \
        calibration.num_samples=2048 quantized_path=null \
        model.eval_batch_size=${BATCH[$model]} "${SPLITS[@]}" \
        create_audio_files=False method=w2a${abits}_${recovery} 2>&1 | tr '\r' '\n'
      stamp "$model W2A$abits $recovery end"
    done
  done
done
stamp "done"
