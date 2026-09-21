#!/bin/bash
#
# W2 weight-only at full LibriSpeech scale, with the recovery recipe.
#
# Question: what does the current recipe reach at 2-bit weights on the whole of test-clean and test-other?
#
# Varies:  the model (whisper, parakeet, canary_qwen).
# Fixed:   W2 asymmetric weights in groups of 128, activations in full precision, the rotation from the
#          GPTQ-fitness weight-only search, GPTQ on 2048 calibration utterances, and the recovery recipe --
#          scale recovery for transformer blocks, block output refitting for Conformer blocks.
# Output:  results/evaluations/<model>_full_w2_recipe_.../results.csv, one directory per run.
#
# Takes about 15 minutes per model for Parakeet and Canary-Qwen and about 40 for Whisper.
# Usage: bash scripts/<name>.sh [conda activate script] [project directory]
CONDA_ACTIVATE=${1:-/home/ubuntu/miniconda3/bin/activate}
PROJECT_DIR=${2:-/home/ubuntu/asrq}
source "$CONDA_ACTIVATE" asrq
cd "$PROJECT_DIR"
SPLITS="eval_datasets=[{dataset:librispeech,split:test.clean},{dataset:librispeech,split:test.other}]"
declare -A BATCH=( [whisper]=64 [parakeet]=128 [canary_qwen]=128 )
stamp() { echo "[$(date +%H:%M:%S)] $*"; }

for model in ${MODELS:-parakeet canary_qwen whisper}; do
  stamp "$model W2 recipe start"
  python -u asrq/exp.py model=$model transform=rotation transform.path=outputs/rotation/wo_w2a_${model}_gptqfit.pt \
    quantizer=gptq quantizer.bits=2 quantizer.symmetric=False quantizer.group_size=128 \
    quantizer.scale_recovery=true quantizer.block_output_refit=true \
    activation_bits=16 calibration.num_samples=2048 quantized_path=null \
    model.eval_batch_size=${BATCH[$model]} "$SPLITS" create_audio_files=False method=full_w2_recipe 2>&1 | tr '\r' '\n'
  stamp "$model W2 recipe end"
done
stamp "done"
