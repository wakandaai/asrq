#!/bin/bash
#
# W2A8 with the recovery recipe, on LibriSpeech test-clean and test-other.
#
# Question: does 2-bit weight quantization survive 8-bit activations? The W2 results so far are weight-only,
#           so this says what the recipe costs once the activations are quantized too.
#
# Varies:  the model (parakeet, canary_qwen, whisper).
# Fixed:   W2 asymmetric weights in groups of 128. Activations 8-bit symmetric, per token, except attn_out and
#          fc2, which keep one scale per group of 128 (activation_groupwise_roles).
#          Rotation: the saved weight-only W2 rotation, reused with transform.check_settings=false --
#          its weight grid is exactly this run's (gptq, 2 bits, groups of 128, asymmetric), but it was searched
#          with activations in full precision, so the check that a rotation matches the run's quantizer has to
#          be turned off. No rotation has been searched against an 8-bit activation quantizer.
#          Quantization: GPTQ on 2048 calibration utterances, scale recovery on the transformer blocks, block
#          output refitting on the Conformer blocks. The inserted Conformer output Linears stay in full
#          precision, as in the weight-only W2 recipe, where quantizing them was measured worse.
#          Evaluation: LibriSpeech test.clean and test.other only, with the activations fake-quantized.
# Output:  results/evaluations/w2a8_<model>_w2a8_recipe_gptq_w2g128asym_a8_rotation_.../{results.csv,config.yaml},
#          one directory per model. Every run is logged to wandb in the group w2a8.
#
# Note: scale recovery fits its scale against the weight error alone, so with quantized activations it also
# moves what the activation quantizer rounds; at 8 bits that shift is small, but it is not accounted for.
#
# Run a subset with MODELS=parakeet bash scripts/w2a8_recipe.sh
#
# Usage: bash scripts/w2a8_recipe.sh [conda activate script] [project directory]
CONDA_ACTIVATE=${1:-/home/ubuntu/miniconda3/bin/activate}
PROJECT_DIR=${2:-/home/ubuntu/asrq}
source "$CONDA_ACTIVATE" asrq
cd "$PROJECT_DIR"
SPLITS="eval_datasets=[{dataset:librispeech,split:test.clean},{dataset:librispeech,split:test.other}]"
declare -A BATCH=( [whisper]=64 [parakeet]=128 [canary_qwen]=128 )
stamp() { echo "[$(date +%H:%M:%S)] $*"; }
EXPERIMENT="exp_name=w2a8 wandb.enabled=true"
ACTIVATIONS="activation_bits=8 activation_symmetric=True activation_group_size=128"

for model in ${MODELS:-parakeet canary_qwen whisper}; do
  stamp "$model W2A8 recipe start"
  python -u asrq/exp.py model=$model \
    transform=rotation transform.path=outputs/rotation/wo_w2a_${model}_gptqfit.pt transform.check_settings=false \
    quantizer=gptq quantizer.bits=2 quantizer.symmetric=False quantizer.group_size=128 \
    quantizer.scale_recovery=true quantizer.weight_recovery=true \
    $EXPERIMENT $ACTIVATIONS calibration.num_samples=2048 quantized_path=null \
    model.eval_batch_size=${BATCH[$model]} "$SPLITS" create_audio_files=False method=w2a8_recipe \
    2>&1 | tr '\r' '\n'
  stamp "$model W2A8 recipe end"
done
stamp "done"
