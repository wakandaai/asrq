#!/bin/bash
#
# Weight-only W4 and W2 on all three models, over the full Open ASR Leaderboard.
#
# Question: how far does weight-only quantization get at 4 and 2 bits, with a Hadamard rotation, GPTQ, scale
#           recovery and per-layer weight recovery?
#
# Varies:  the model (whisper, parakeet, canary_qwen) and the weight width (4, 2).
# Fixed:   groups of 128, activations in full precision (activation_bits=16).
#          Rotation: Cayley SGD on R1 and a head-wise R2 per attention module, from random Hadamards, over 800
#          calibration utterances for one epoch at learning rate 0.5, with transform.objective=auto, which for
#          Cayley is the model's own training loss. Note what that search can and cannot do here: every rewrite
#          a rotation makes is exact, so with the activations in full precision the model's output does not
#          depend on the rotation and the objective is flat -- learn_rotations says so and warns. The pass
#          therefore returns the random Hadamard it starts from, and the rotation still earns its place by
#          making the weights easier to round, not by anything the search found. Set transform.epochs=0 to skip
#          the pass and save the same random Hadamard in a minute. The weight-only search that used to fit a
#          rotation against rounded weights has been removed; see the git history of asrq/transforms/rotation.
#          On Parakeet the loss is the CTC loss, whose backward has no deterministic CUDA kernel, so its search
#          passes deterministic=false.
#          Quantization: GPTQ on 2048 calibration utterances, with
#            - scale_recovery: a per-input-channel scale on every layer a norm feeds, fitted inside GPTQ and
#              folded into that norm. Weight-only runs are the only place it belongs: with quantized
#              activations the activation quantizer would have to round the scaled channels, and
#              asrq/experiment.py turns it off there.
#            - layer_weight_recovery: every quantized weight refit, just before it is quantized, onto the
#              output it had before anything in its block was quantized. It also makes each block quantize its
#              layers in the order they run.
#            - weight_recovery: the Conformer block output refit, which the two models that have an inserted
#              block-output Linear use; their Linears are refit and then quantized at 4 bits whatever the rest
#              of the model uses. Whisper has no such layer and nothing to refit.
#          Evaluation: all seven leaderboard datasets.
# Output:  outputs/rotation/cayley_wo_w<bits>_<model>.pt, and
#          results/evaluations/wo_w4_w2_<model>_wo_w<bits>_.../{results.csv,config.yaml}, one directory
#          per run. Every run is logged to wandb in the group wo_w4_w2, named
#          wo_w4_w2_<model>_<method>.
#
# A search is skipped when its rotation file already exists, so the script can be re-run after an interruption.
# Roughly 1-2 hours per evaluation; run a subset with MODELS=whisper BITS=2 bash scripts/wo_w4_w2.sh
# Usage: bash scripts/<name>.sh [conda activate script] [project directory]
CONDA_ACTIVATE=${1:-/home/ubuntu/miniconda3/bin/activate}
PROJECT_DIR=${2:-/home/ubuntu/asrq}
source "$CONDA_ACTIVATE" asrq
cd "$PROJECT_DIR"
declare -A BATCH=( [whisper]=64 [parakeet]=128 [canary_qwen]=128 )
# W2 asymmetric and W4 symmetric, as in the earlier runs.
declare -A SYMMETRIC=( [4]=True [2]=False )
# Parakeet's CE objective is the CTC loss, and ctc_loss_backward has no deterministic CUDA kernel, so its
# search warns instead of refusing. The candidate draws and the dataloader stay seeded.
declare -A DETERMINISTIC=( [parakeet]="deterministic=false" [whisper]="" [canary_qwen]="" )
# Only the Conformer models have block-output Linears to quantize; Whisper has no such setting at all, and
# weight_recovery simply finds nothing to refit there.
declare -A BLOCK_OUT=(
  [parakeet]="model.quantize_block_output_linear=true quantizer.block_output_linear_bits=4"
  [canary_qwen]="model.quantize_block_output_linear=true quantizer.block_output_linear_bits=4"
  [whisper]=""
)
stamp() { echo "[$(date +%H:%M:%S)] $*"; }
EXPERIMENT="exp_name=wo_w4_w2 wandb.enabled=true"
RECOVERY="quantizer.scale_recovery=true quantizer.layer_weight_recovery=true quantizer.weight_recovery=true"

for model in ${MODELS:-parakeet whisper canary_qwen}; do
  for bits in ${BITS:-4 2}; do
    grid="quantizer=gptq quantizer.bits=$bits quantizer.symmetric=${SYMMETRIC[$bits]} quantizer.group_size=128"
    rotation=outputs/rotation/cayley_wo_w${bits}_${model}.pt

    if [ ! -f $rotation ]; then
      stamp "$model W$bits Cayley rotation start"
      python -u asrq/rot-exp.py model=$model $grid $EXPERIMENT activation_bits=16 \
        ${DETERMINISTIC[$model]} method=cayley_wo_w$bits calibration.num_samples=800 \
        transform=rotation transform.path=$rotation \
        transform.num_samples=800 transform.epochs=1 \
        transform.learning_rate=0.5 2>&1 | tr '\r' '\n'
      stamp "$model W$bits Cayley rotation end"
    else
      stamp "$model W$bits rotation exists, skipping the search"
    fi

    stamp "$model W$bits quantize and evaluate start"
    python -u asrq/exp.py model=$model transform=rotation transform.path=$rotation $grid $EXPERIMENT \
      $RECOVERY ${BLOCK_OUT[$model]} \
      activation_bits=16 calibration.num_samples=2048 quantized_path=null \
      model.eval_batch_size=${BATCH[$model]} create_audio_files=False method=wo_w$bits 2>&1 | tr '\r' '\n'
    stamp "$model W$bits quantize and evaluate end"
  done
done
stamp "done"
