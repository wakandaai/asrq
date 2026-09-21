#!/bin/bash
#
# Weight-only W4 and W2 on all three models, over the full Open ASR Leaderboard.
#
# Question: how far does weight-only quantization get at 4 and 2 bits, with an evolutionary Hadamard rotation
#           searched against the same weight quantizer, GPTQ, scale recovery, and refit block-output layers?
#
# Varies:  the model (whisper, parakeet, canary_qwen) and the weight width (4, 2).
# Fixed:   groups of 128, activations in full precision (activation_bits=16).
#          Rotation: evolutionary search over the signs of R1 = diag(s1) H diag(s2), weight-only fitness with
#          GPTQ (the quantizer the run uses), 128 calibration utterances -- which are also the Hessians the
#          search's GPTQ uses -- stages 16/64/128 keeping 8 then 4 children, 32 offspring of which the last is
#          drawn at random, 8 generations. That is 32*16 + 8*64 + 4*128 = 1536 samples scored per generation.
#          One rotation per model and width, since the fitness depends on the weight grid.
#          Quantization: GPTQ on 2048 calibration utterances, scale recovery on the transformer blocks, block
#          output refitting on the Conformer blocks, whose inserted output Linears are refit and then quantized
#          at 4 bits whatever the rest of the model uses.
#          Evaluation: all seven leaderboard datasets.
# Output:  outputs/rotation/evo128_wo_w<bits>_<model>.pt, and
#          results/evaluations/wo_w4_w2_<model>_wo_w<bits>_.../{results.csv,config.yaml}, one directory
#          per run. Every run is logged to wandb in the group wo_w4_w2, named
#          wo_w4_w2_<model>_<method>.
#
# A search is skipped when its rotation file already exists, so the script can be re-run after an interruption.
# Roughly 1-3 hours per search and 1-2 hours per evaluation, so about a day for all twelve steps; run a subset
# with MODELS=whisper BITS=2 bash scripts/wo_w4_w2.sh
# Usage: bash scripts/<name>.sh [conda activate script] [project directory]
CONDA_ACTIVATE=${1:-/home/ubuntu/miniconda3/bin/activate}
PROJECT_DIR=${2:-/home/ubuntu/asrq}
source "$CONDA_ACTIVATE" asrq
cd "$PROJECT_DIR"
declare -A BATCH=( [whisper]=64 [parakeet]=128 [canary_qwen]=128 )
# W2 asymmetric and W4 symmetric, as in the earlier runs; the search and the quantization must agree, and
# asrq/exp.py refuses a rotation searched under a different weight grid.
declare -A SYMMETRIC=( [4]=True [2]=False )
# Only the Conformer models have block-output Linears to quantize; Whisper has no such setting at all, and
# block_output_refit simply finds nothing to refit there.
declare -A BLOCK_OUT=(
  [parakeet]="model.quantize_block_output_linear=true quantizer.block_output_linear_bits=4"
  [canary_qwen]="model.quantize_block_output_linear=true quantizer.block_output_linear_bits=4"
  [whisper]=""
)
stamp() { echo "[$(date +%H:%M:%S)] $*"; }
EXPERIMENT="exp_name=wo_w4_w2 wandb.enabled=true"

for model in ${MODELS:-parakeet whisper canary_qwen}; do
  for bits in ${BITS:-4 2}; do
    grid="quantizer=gptq quantizer.bits=$bits quantizer.symmetric=${SYMMETRIC[$bits]} quantizer.group_size=128"
    rotation=outputs/rotation/evo128_wo_w${bits}_${model}.pt

    if [ ! -f $rotation ]; then
      stamp "$model W$bits rotation search start"
      python -u asrq/rot-exp.py model=$model $grid $EXPERIMENT activation_bits=16 method=search_w$bits \
        calibration.num_samples=128 transform=rotation transform.path=$rotation \
        transform.weight_only=true transform.weight_only_quantizer=gptq \
        transform.search=evolution transform.num_samples=128 \
        transform.evolution.stages=1 \
        transform.evolution.offspring=32 \
        transform.evolution.random_offspring=32 \
        transform.evolution.generations=1 2>&1 | tr '\r' '\n'
      stamp "$model W$bits rotation search end"
    else
      stamp "$model W$bits rotation exists, skipping the search"
    fi

    stamp "$model W$bits quantize and evaluate start"
    python -u asrq/exp.py model=$model transform=rotation transform.path=$rotation $grid $EXPERIMENT \
      quantizer.scale_recovery=true quantizer.block_output_refit=true ${BLOCK_OUT[$model]} \
      activation_bits=16 calibration.num_samples=2048 quantized_path=null \
      model.eval_batch_size=${BATCH[$model]} create_audio_files=False method=wo_w$bits 2>&1 | tr '\r' '\n'
    stamp "$model W$bits quantize and evaluate end"
  done
done
stamp "done"
