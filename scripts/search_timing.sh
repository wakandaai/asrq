#!/bin/bash
#
# Cayley SGD against the evolutionary search: how long does each take?
#
# Question: at the settings we would actually use, what does one generation of the evolutionary search cost
#           against one epoch of Cayley SGD, on the same model and the same activation quantization?
#
# Varies:  the search (evolution, cayley) and the model.
# Fixed:   A4 with attn_out and fc2 in groups of 128, 128 calibration utterances, batch size 4.
#          Evolution: stages 32/64/128, 32 offspring, so 1536 scored samples per generation.
#          Cayley: one epoch over the same 128 utterances, so 32 optimiser steps.
# Method:  three runs per model. The first does no search (evolution with generations=0) and measures what
#          loading the model, folding the norms and verifying the rewrite costs. The other two add one
#          generation and one epoch, so the difference is the search's own cost.
# Output:  timestamps in the log; the rotations go to a scratch path and are not meant to be used.
#
# Usage: bash scripts/search_timing.sh [conda activate script] [project directory]
CONDA_ACTIVATE=${1:-/home/ubuntu/miniconda3/bin/activate}
PROJECT_DIR=${2:-/home/ubuntu/asrq}
source "$CONDA_ACTIVATE" asrq
cd "$PROJECT_DIR"
stamp() { echo "[$(date +%s)] $*"; }
SCRATCH=outputs/rotation/timing
mkdir -p $SCRATCH
COMMON="activation_bits=4 calibration.num_samples=128 transform=rotation transform.num_samples=128 transform.batch_size=4"

for model in ${MODELS:-whisper parakeet canary_qwen}; do
  rm -f $SCRATCH/${model}_*.pt

  stamp "$model setup only start"
  python -u asrq/rot-exp.py model=$model $COMMON transform.search=evolution \
    transform.evolution.generations=0 transform.path=$SCRATCH/${model}_setup.pt 2>&1 | tr '\r' '\n'
  stamp "$model setup only end"

  stamp "$model evolution one generation start"
  python -u asrq/rot-exp.py model=$model $COMMON transform.search=evolution \
    transform.evolution.generations=1 transform.evolution.offspring=32 \
    "transform.evolution.stage_samples=[32,64,128]" transform.path=$SCRATCH/${model}_evolution.pt 2>&1 | tr '\r' '\n'
  stamp "$model evolution one generation end"

  stamp "$model cayley one epoch start"
  python -u asrq/rot-exp.py model=$model $COMMON transform.search=cayley transform.epochs=1 \
    transform.learning_rate=0.5 transform.path=$SCRATCH/${model}_cayley.pt 2>&1 | tr '\r' '\n'
  stamp "$model cayley one epoch end"
done
stamp "done"
