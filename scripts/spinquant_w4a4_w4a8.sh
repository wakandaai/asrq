#!/bin/bash
#
# SpinQuant-style baseline: Cayley SGD rotations, then plain GPTQ, at W4A4 and W4A8.
#
# Question: what do learned dense rotations and GPTQ alone reach, with none of this project's recovery steps?
#           This is the comparison the scale recovery and block output refit results are measured against, and
#           the closest thing here to SpinQuant: a rotation trained by Cayley SGD on the model's own loss.
#
# Varies:  the model (whisper, parakeet, canary_qwen) and the activation width (4, 8).
# Fixed:   W4 symmetric weights in groups of 128. Activations symmetric, per token, except attn_out and fc2,
#          which keep one scale per group of 128 (activation_groupwise_roles).
#          Rotation: Cayley SGD on R1 and a head-wise R2 per attention module, from random Hadamards, over 800
#          calibration utterances for one epoch at learning rate 0.5, with the activations fake-quantized as
#          evaluation will quantize them. The objective is transform.objective=auto, which for Cayley is the
#          model's own training loss on the calibration transcripts, as SpinQuant optimises, rather than the KL
#          to the full-precision model. On Parakeet that loss is the CTC loss, whose backward has no
#          deterministic CUDA kernel, so its search passes deterministic=false.
#          Quantization: GPTQ on 2048 calibration utterances and nothing else -- no scale recovery and no block
#          output refitting. The Linears a rotation inserts after each Conformer block's output norm are still
#          quantized at 4 bits, so this baseline carries the same weights as the recipe it is compared with;
#          with the refit off they are quantized in the block loop like any other layer.
#          Evaluation: all seven leaderboard datasets, with the activations fake-quantized.
# Output:  outputs/rotation/cayley_w4a<bits>_<model>.pt, and
#          results/evaluations/spinquant_<model>_spinquant_w4a<bits>_.../{results.csv,config.yaml}, one
#          directory per run. Every run is logged to wandb in the group spinquant.
#
# A search is skipped when its rotation file already exists, so the script can be re-run after an interruption.
# Run a subset with MODELS=whisper ABITS=4 bash scripts/spinquant_w4a4_w4a8.sh
#
# Usage: bash scripts/spinquant_w4a4_w4a8.sh [conda activate script] [project directory]
CONDA_ACTIVATE=${1:-/home/ubuntu/miniconda3/bin/activate}
PROJECT_DIR=${2:-/home/ubuntu/asrq}
source "$CONDA_ACTIVATE" asrq
cd "$PROJECT_DIR"
declare -A BATCH=( [whisper]=64 [parakeet]=128 [canary_qwen]=128 )
# Parakeet's CE objective is the CTC loss, and ctc_loss_backward has no deterministic CUDA kernel, so its search
# warns instead of refusing. The candidate draws and the dataloader stay seeded; only the gradient is affected.
declare -A DETERMINISTIC=( [parakeet]="deterministic=false" [whisper]="" [canary_qwen]="" )
# Only the Conformer models have block-output Linears; quantizing them at 4 bits keeps this baseline's weights
# comparable with the recipe's, even though nothing here refits them. Whisper has no such setting.
declare -A BLOCK_OUT=(
  [parakeet]="model.quantize_block_output_linear=true quantizer.block_output_linear_bits=4"
  [canary_qwen]="model.quantize_block_output_linear=true quantizer.block_output_linear_bits=4"
  [whisper]=""
)
stamp() { echo "[$(date +%H:%M:%S)] $*"; }
EXPERIMENT="exp_name=spinquant wandb.enabled=true"
# The search and the quantization must agree on these; asrq/exp.py refuses a rotation searched under other
# activation settings.
ACTIVATIONS="activation_symmetric=True activation_group_size=128"
GRID="quantizer=gptq quantizer.bits=4 quantizer.symmetric=True quantizer.group_size=128"

for model in ${MODELS:-parakeet whisper canary_qwen}; do
  for abits in ${ABITS:-4 8}; do
    rotation=outputs/rotation/cayley_w4a${abits}_${model}.pt

    if [ ! -f $rotation ]; then
      stamp "$model W4A$abits Cayley rotation start"
      python -u asrq/rot-exp.py model=$model $GRID $EXPERIMENT $ACTIVATIONS activation_bits=$abits \
        ${DETERMINISTIC[$model]} method=cayley_w4a$abits calibration.num_samples=800 \
        transform=rotation transform.path=$rotation \
        transform.search=cayley transform.num_samples=800 transform.epochs=1 \
        transform.learning_rate=0.5 2>&1 | tr '\r' '\n'
      stamp "$model W4A$abits Cayley rotation end"
    else
      stamp "$model W4A$abits rotation exists, skipping the search"
    fi

    stamp "$model W4A$abits quantize and evaluate start"
    python -u asrq/exp.py model=$model transform=rotation transform.path=$rotation $GRID $EXPERIMENT \
      $ACTIVATIONS activation_bits=$abits ${BLOCK_OUT[$model]} \
      calibration.num_samples=2048 quantized_path=null \
      model.eval_batch_size=${BATCH[$model]} create_audio_files=False method=spinquant_w4a$abits \
      2>&1 | tr '\r' '\n'
    stamp "$model W4A$abits quantize and evaluate end"
  done
done
stamp "done"
