#!/bin/bash
#
# Weight and activation quantization, W4A4 and W4A8, on all three models, over the full Open ASR Leaderboard.
#
# Question: how far does 4-bit weight quantization get when the activations are quantized too, at 4 and 8 bits,
#           with an evolutionary Hadamard rotation searched against the same activation quantizer, GPTQ, scale
#           recovery, and refit block-output layers?
#
# Varies:  the model (whisper, parakeet, canary_qwen) and the activation width (4, 8).
# Fixed:   W4 symmetric weights in groups of 128. Activations symmetric, per token, except attn_out and fc2,
#          which keep one scale per group of 128 (activation_groupwise_roles).
#          Rotation: evolutionary search over the signs of R1 = diag(s1) H diag(s2), fitness is the KL to the
#          full-precision model with the activations fake-quantized exactly as evaluation will, 128 calibration
#          utterances, stages 16/64/128 keeping 16 then 4 children, 32 offspring of which half are drawn at
#          random, 8 generations. One rotation per model and activation width, since the fitness depends on the
#          activation quantizer; the weights are not quantized during the search.
#          Quantization: GPTQ on 2048 calibration utterances, scale recovery on the transformer blocks, block
#          output refitting on the Conformer blocks, whose inserted output Linears are refit and then quantized
#          at 4 bits.
#          Evaluation: all seven leaderboard datasets, with the activations fake-quantized.
# Output:  outputs/rotation/evo128_wa_w4a<bits>_<model>.pt, and
#          results/evaluations/wa_w4a4_w4a8_<model>_wa_w4a<bits>_.../{results.csv,config.yaml}, one directory
#          per run. Every run is logged to wandb in the group wa_w4a4_w4a8.
#
# A search is skipped when its rotation file already exists, so the script can be re-run after an interruption.
# Run a subset with MODELS=whisper ABITS=4 bash scripts/wa_w4a4_w4a8.sh
#
# Usage: bash scripts/wa_w4a4_w4a8.sh [conda activate script] [project directory]
CONDA_ACTIVATE=${1:-/home/ubuntu/miniconda3/bin/activate}
PROJECT_DIR=${2:-/home/ubuntu/asrq}
source "$CONDA_ACTIVATE" asrq
cd "$PROJECT_DIR"
declare -A BATCH=( [whisper]=64 [parakeet]=128 [canary_qwen]=128 )
# Only the Conformer models have block-output Linears to quantize; Whisper has no such setting at all, and
# block_output_refit simply finds nothing to refit there.
declare -A BLOCK_OUT=(
  [parakeet]="model.quantize_block_output_linear=true quantizer.block_output_linear_bits=4"
  [canary_qwen]="model.quantize_block_output_linear=true quantizer.block_output_linear_bits=4"
  [whisper]=""
)
stamp() { echo "[$(date +%H:%M:%S)] $*"; }
EXPERIMENT="exp_name=wa_w4a4_w4a8 wandb.enabled=true"
# The search and the quantization must agree on these; asrq/exp.py refuses a rotation searched under other
# activation settings.
ACTIVATIONS="activation_symmetric=True activation_group_size=128"
GRID="quantizer=gptq quantizer.bits=4 quantizer.symmetric=True quantizer.group_size=128"

for model in ${MODELS:-parakeet whisper canary_qwen}; do
  for abits in ${ABITS:-4 8}; do
    rotation=outputs/rotation/evo128_wa_w4a${abits}_${model}.pt

    if [ ! -f $rotation ]; then
      stamp "$model W4A$abits rotation search start"
      python -u asrq/rot-exp.py model=$model $GRID $EXPERIMENT $ACTIVATIONS activation_bits=$abits \
        method=search_w4a$abits calibration.num_samples=128 \
        transform=rotation transform.path=$rotation \
        transform.search=evolution transform.num_samples=128 \
        "transform.evolution.stage_samples=[]" "transform.evolution.survivors=[]" \
        transform.evolution.random_offspring=32 transform.evolution.stages=1 \
        transform.evolution.offspring=32 2>&1 | tr '\r' '\n'
      stamp "$model W4A$abits rotation search end"
    else
      stamp "$model W4A$abits rotation exists, skipping the search"
    fi

    stamp "$model W4A$abits quantize and evaluate start"
    python -u asrq/exp.py model=$model transform=rotation transform.path=$rotation $GRID $EXPERIMENT \
      $ACTIVATIONS activation_bits=$abits \
      quantizer.scale_recovery=true quantizer.block_output_refit=true ${BLOCK_OUT[$model]} \
      calibration.num_samples=2048 quantized_path=null \
      model.eval_batch_size=${BATCH[$model]} create_audio_files=False method=wa_w4a$abits 2>&1 | tr '\r' '\n'
    stamp "$model W4A$abits quantize and evaluate end"
  done
done
stamp "done"
