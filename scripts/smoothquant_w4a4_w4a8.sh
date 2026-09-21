#!/bin/bash
#
# SmoothQuant-style baseline: searched scaling, then round-to-nearest weights, at W4A4 and W4A8.
#
# Question: what do smoothing scales and plain RTN reach, with no rotation, no GPTQ and none of this project's
#           recovery steps? The weakest baseline of the set, and the one that isolates what the rotation buys.
#
# Varies:  the model (whisper, parakeet, canary_qwen) and the activation width (4, 8).
# Fixed:   W4 symmetric weights in groups of 128, rounded to nearest. Activations symmetric, per token, except
#          attn_out and fc2, which keep one scale per group of 128 (activation_groupwise_roles).
#          Transform: smoothquant scales, one per scaled entry, searched inside exp.py against the same weight
#          and activation quantization the run evaluates with, then folded in -- into the previous layer's rows
#          where one exists, and into an InputScale module after the activation where a nonlinearity sits in
#          between. Statistics come from every calibration utterance, the grid search from the first 128.
#          Quantization: RTN alone. No rotation, no GPTQ, no scale recovery, no block output refitting, and no
#          inserted block-output Linears, which only a rotation creates.
#          Evaluation: all seven leaderboard datasets, with the activations fake-quantized.
# Output:  outputs/scaling/smoothquant_w4a<bits>_<model>.pt, and
#          results/evaluations/smoothquant_<model>_smoothquant_w4a<bits>_.../{results.csv,config.yaml}, one
#          directory per run. Every run is logged to wandb in the group smoothquant.
#
# A scale file that already exists is reused rather than searched again, so the script can be re-run after an
# interruption. Run a subset with MODELS=whisper ABITS=4 bash scripts/smoothquant_w4a4_w4a8.sh
#
# Usage: bash scripts/smoothquant_w4a4_w4a8.sh [conda activate script] [project directory]
CONDA_ACTIVATE=${1:-/home/ubuntu/miniconda3/bin/activate}
PROJECT_DIR=${2:-/home/ubuntu/asrq}
source "$CONDA_ACTIVATE" asrq
cd "$PROJECT_DIR"
mkdir -p outputs/scaling
declare -A BATCH=( [whisper]=64 [parakeet]=128 [canary_qwen]=128 )
stamp() { echo "[$(date +%H:%M:%S)] $*"; }
EXPERIMENT="exp_name=smoothquant wandb.enabled=true"
ACTIVATIONS="activation_symmetric=True activation_group_size=128 activation_groupwise_roles=[attn_out,fc2]"
# RTN needs no Hessians, so the calibration set is only what the scale search sees: statistics over all of it,
# and the grid search over the first 128 utterances.
GRID="quantizer=rtn quantizer.bits=4 quantizer.symmetric=True quantizer.group_size=128"

for model in ${MODELS:-parakeet whisper canary_qwen}; do
  for abits in ${ABITS:-4 8}; do
    scales=outputs/scaling/smoothquant_w4a${abits}_${model}.pt
    if [ -f $scales ]; then
      obtain=transform.obtain_scales=false
      stamp "$model W4A$abits scales exist, reusing them"
    else
      obtain=transform.obtain_scales=true
    fi

    stamp "$model W4A$abits scale, quantize and evaluate start"
    python -u asrq/exp.py model=$model transform=scaling transform.path=$scales $obtain transform.type=smoothquant \
      $GRID $EXPERIMENT "$ACTIVATIONS" activation_bits=$abits \
      calibration.num_samples=512 quantized_path=null \
      model.eval_batch_size=${BATCH[$model]} create_audio_files=False method=smoothquant_w4a$abits \
      2>&1 | tr '\r' '\n'
    stamp "$model W4A$abits scale, quantize and evaluate end"
  done
done
stamp "done"
