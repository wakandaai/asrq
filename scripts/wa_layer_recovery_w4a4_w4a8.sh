#!/bin/bash
#
# Joint weight-activation quantization: a Cayley SGD rotation, GPTQ, and per-layer weight recovery.
#
# Question: what does per-layer weight recovery add to a learned rotation at W4A4 and W4A8? Compared line for
#           line with spinquant_w4a4_w4a8.sh, which is the same rotation and the same grid with no recovery at
#           all, so the two differ by this one step.
#
# Varies:  the model (whisper, parakeet, canary_qwen) and the activation width (4, 8).
# Fixed:   W4 symmetric weights in groups of 128. Activations symmetric, per token, except attn_out and fc2,
#          which keep one scale per group of 128 (activation_groupwise_roles).
#          Rotation: Cayley SGD on R1 and a head-wise R2 per attention module, from random Hadamards, over 800
#          calibration utterances for one epoch at learning rate 0.5, with the activations fake-quantized as
#          evaluation will quantize them. The objective is transform.objective=auto, which for Cayley is the
#          model's own training loss on the calibration transcripts. Here the search has something to find:
#          the rotation only changes the output because the activations are quantized, and what it is fitting
#          is how well the rotation spreads the outliers the quantizer has to round. One rotation per model and
#          activation width, since the objective depends on the activation grid, and asrq/exp.py refuses a
#          rotation searched under other activation settings. On Parakeet the loss is the CTC loss, whose
#          backward has no deterministic CUDA kernel, so its search passes deterministic=false.
#          Quantization: GPTQ on 2048 calibration utterances with layer_weight_recovery -- every quantized
#          weight refit, just before it is quantized, onto the output it had before anything in its block was
#          quantized, which also makes each block quantize its layers in the order they run.
#          No scale recovery: it folds a per-input-channel scale into the norm feeding a layer, which is free
#          for the weights and not for the activations, since the activation quantizer then has to round the
#          scaled channels and a per-channel scale pulls apart the outliers the rotation spread. It is a
#          weight-only step, and asrq/experiment.py turns it off here and says so, so passing it would record
#          as off anyway.
#          No block output refit either, so the Conformer models' inserted block-output Linears are corrected
#          by layer recovery like any other layer. They are still quantized at 4 bits, as in
#          spinquant_w4a4_w4a8.sh, so the two runs carry the same weights. Add quantizer.weight_recovery=true
#          to correct those blocks at their output layer instead, which is what wa_w4a4_w4a8.sh does.
#          Evaluation: all seven leaderboard datasets, with the activations fake-quantized.
# Output:  outputs/rotation/cayley_wa_w4a<bits>_<model>.pt, and
#          results/evaluations/wa_layer_recovery_<model>_wa_layer_recovery_w4a<bits>_.../{results.csv,config.yaml},
#          one directory per run. Every run is logged to wandb in the group wa_layer_recovery.
#
# A search is skipped when its rotation file already exists, so the script can be re-run after an interruption.
# Run a subset with MODELS=whisper ABITS=4 bash scripts/wa_layer_recovery_w4a4_w4a8.sh
#
# Usage: bash scripts/wa_layer_recovery_w4a4_w4a8.sh [conda activate script] [project directory]
CONDA_ACTIVATE=${1:-/home/ubuntu/miniconda3/bin/activate}
PROJECT_DIR=${2:-/home/ubuntu/asrq}
source "$CONDA_ACTIVATE" asrq
cd "$PROJECT_DIR"
declare -A BATCH=( [whisper]=64 [parakeet]=128 [canary_qwen]=128 )
# Parakeet's CE objective is the CTC loss, and ctc_loss_backward has no deterministic CUDA kernel, so its search
# warns instead of refusing. The candidate draws and the dataloader stay seeded; only the gradient is affected.
declare -A DETERMINISTIC=( [parakeet]="deterministic=false" [whisper]="" [canary_qwen]="" )
# Only the Conformer models have block-output Linears; quantizing them at 4 bits keeps this run's weights
# comparable with spinquant_w4a4_w4a8.sh's. Whisper has no such setting.
declare -A BLOCK_OUT=(
  [parakeet]="model.quantize_block_output_linear=true quantizer.block_output_linear_bits=4"
  [canary_qwen]="model.quantize_block_output_linear=true quantizer.block_output_linear_bits=4"
  [whisper]=""
)
stamp() { echo "[$(date +%H:%M:%S)] $*"; }
EXPERIMENT="exp_name=wa_layer_recovery wandb.enabled=true"
# The search and the quantization must agree on these; asrq/exp.py refuses a rotation searched under other
# activation settings.
ACTIVATIONS="activation_symmetric=True activation_group_size=128"
GRID="quantizer=gptq quantizer.bits=4 quantizer.symmetric=True quantizer.group_size=128"

for model in ${MODELS:-parakeet whisper canary_qwen}; do
  for abits in ${ABITS:-4 8}; do
    rotation=outputs/rotation/cayley_wa_w4a${abits}_${model}.pt

    if [ ! -f $rotation ]; then
      stamp "$model W4A$abits Cayley rotation start"
      python -u asrq/rot-exp.py model=$model $GRID $EXPERIMENT $ACTIVATIONS activation_bits=$abits \
        ${DETERMINISTIC[$model]} method=cayley_wa_w4a$abits calibration.num_samples=800 \
        transform=rotation transform.path=$rotation \
        transform.num_samples=800 transform.epochs=1 \
        transform.learning_rate=0.5 2>&1 | tr '\r' '\n'
      stamp "$model W4A$abits Cayley rotation end"
    else
      stamp "$model W4A$abits rotation exists, skipping the search"
    fi

    stamp "$model W4A$abits quantize and evaluate start"
    python -u asrq/exp.py model=$model transform=rotation transform.path=$rotation $GRID $EXPERIMENT \
      $ACTIVATIONS activation_bits=$abits quantizer.layer_weight_recovery=true ${BLOCK_OUT[$model]} \
      calibration.num_samples=2048 quantized_path=null \
      model.eval_batch_size=${BATCH[$model]} create_audio_files=False \
      method=wa_layer_recovery_w4a$abits 2>&1 | tr '\r' '\n'
    stamp "$model W4A$abits quantize and evaluate end"
  done
done
stamp "done"
