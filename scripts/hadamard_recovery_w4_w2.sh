#!/bin/bash
#
# Weight-only W4 and W2 with an unlearned Hadamard rotation, scale recovery and per-layer weight recovery.
#
# Question: with the rotation taken as given rather than searched, how far do the two fitted recoveries get on
#           their own? Compared line for line with gptq_recovery_w4_w2.sh, which is the same recovery pair
#           without a rotation, and with wo_w4_w2.sh, which adds the Conformer block output refit.
#
# Varies:  the model (whisper, parakeet, canary_qwen) and the weight width (4, 2).
# Fixed:   groups of 128, W4 symmetric and W2 asymmetric as in the other weight-only runs, activations in full
#          precision (activation_bits=16).
#          Rotation: the random Hadamard R1 and head-wise R2s that a search would start from, saved with
#          transform.epochs=0 and no Cayley step. Nothing is learned and nothing could be: every rewrite a
#          rotation makes is exact, so with the activations in full precision the model's output does not
#          depend on the rotation and the search objective is flat -- learn_rotations says as much and warns.
#          The rotation earns its place by making the weights easier to round, which is a property of its
#          structure, not of a search. One rotation per model serves both widths, since nothing about it
#          depends on the weight grid, and it is seeded by cfg.seed. The search's calibration arguments are
#          kept small because no step is taken: the data is loaded only for the objective it prints once.
#          Quantization: GPTQ on 2048 calibration utterances, with
#            - scale_recovery: a per-input-channel scale on every layer a norm feeds, fitted inside GPTQ and
#              folded into that norm. Only weight-only runs get it; asrq/experiment.py turns it off when the
#              activations are quantized.
#            - layer_weight_recovery: every quantized weight refit, just before it is quantized, onto the
#              output it had before anything in its block was quantized. It also makes each block quantize its
#              layers in the order they run.
#          Which blocks each one reaches is not the same, and a rotation decides it:
#            - Whisper: scale recovery on every encoder and decoder norm, layer recovery on every layer.
#            - Canary-Qwen: scale recovery on the LLM's norms only; its Conformer encoder is rotated, so
#              conformer_scale_recovery_targets returns nothing there.
#            - Parakeet: no scale recovery at all. A rotation inserts a Linear after each Conformer block's
#              output norm, and those blocks are excluded on the grounds that the block output refit corrects
#              them instead -- see conformer_scale_recovery_targets. This run does not enable that refit, so
#              Parakeet is a layer-recovery-only run. Add quantizer.weight_recovery=true for the refit, or
#              compare against wo_w4_w2.sh, which has it.
#          The Conformer models' inserted block-output Linears are quantized at the run's own width, leaving
#          block_output_linear_bits null rather than pinning them to 4 bits as wo_w4_w2.sh does, so nothing in
#          the model is wider than the width being measured. With no block output refit enabled, nothing claims
#          those layers, so they are quantized in the block loop and refit by layer recovery like any other
#          layer. At W2 that puts them on the 2-bit grid as well, and their error reaches the residual stream
#          with only that refit between it and the block's output.
#          Evaluation: all seven leaderboard datasets.
# Output:  outputs/rotation/hadamard_<model>.pt, and
#          results/evaluations/hadamard_recovery_<model>_hadamard_recovery_w<bits>_.../{results.csv,config.yaml},
#          one directory per run. Every run is logged to wandb in the group hadamard_recovery.
#
# The rotation is saved once per model and reused, so re-running the script costs nothing extra.
# Roughly 1-2 hours per evaluation; run a subset with MODELS=whisper BITS=2 bash scripts/hadamard_recovery_w4_w2.sh
# Usage: bash scripts/hadamard_recovery_w4_w2.sh [conda activate script] [project directory]
CONDA_ACTIVATE=${1:-/home/ubuntu/miniconda3/bin/activate}
PROJECT_DIR=${2:-/home/ubuntu/asrq}
source "$CONDA_ACTIVATE" asrq
cd "$PROJECT_DIR"
declare -A BATCH=( [whisper]=64 [parakeet]=128 [canary_qwen]=128 )
# W2 asymmetric and W4 symmetric, as in the other weight-only runs.
declare -A SYMMETRIC=( [4]=True [2]=False )
# Parakeet's objective is the CTC loss, whose backward has no deterministic CUDA kernel. No step is taken
# here, but the flag costs nothing and keeps the command the same as the searching scripts'.
declare -A DETERMINISTIC=( [parakeet]="deterministic=false" [whisper]="" [canary_qwen]="" )
# Only the Conformer models have block-output Linears. block_output_linear_bits is left null, which gives them
# the run's own bits; see ModelQ.layer_quant_cfg. Whisper has no such setting.
declare -A BLOCK_OUT=(
  [parakeet]="model.quantize_block_output_linear=true"
  [canary_qwen]="model.quantize_block_output_linear=true"
  [whisper]=""
)
stamp() { echo "[$(date +%H:%M:%S)] $*"; }
EXPERIMENT="exp_name=hadamard_recovery wandb.enabled=true"
RECOVERY="quantizer.scale_recovery=true quantizer.layer_weight_recovery=true"

for model in ${MODELS:-whisper canary_qwen parakeet}; do
  rotation=outputs/rotation/hadamard_${model}.pt

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

  for bits in ${BITS:-4 2}; do
    grid="quantizer=gptq quantizer.bits=$bits quantizer.symmetric=${SYMMETRIC[$bits]} quantizer.group_size=128"
    stamp "$model W$bits quantize and evaluate start"
    python -u asrq/exp.py model=$model transform=rotation transform.path=$rotation $grid $EXPERIMENT \
      $RECOVERY ${BLOCK_OUT[$model]} activation_bits=16 calibration.num_samples=2048 quantized_path=null \
      model.eval_batch_size=${BATCH[$model]} create_audio_files=False \
      method=hadamard_recovery_w$bits 2>&1 | tr '\r' '\n'
    stamp "$model W$bits quantize and evaluate end"
  done
done
stamp "done"
