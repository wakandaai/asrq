#!/bin/bash
#
# W4A4 speedup over the full-precision model, with real low-bit linear layers.
#
# Question: what does the quantization actually buy in wall-clock terms once the rotated, GPTQ-quantized model
#           runs on humming's low-bit kernels instead of simulating the quantization in fp16?
#
# Varies:  the arm (fp16, then W4A4 on humming layers) and the batch size (1, then 128).
# Fixed:   Parakeet-CTC-1.1b, float16, CUDA-graph transcription (generate_parakeet_cuda_graphs), LibriSpeech
#          test.clean. Both arms transcribe the same utterances at a given batch size, so their RTFx ratio is
#          the speedup. RTFx = audio seconds / the time inside generate_fn, measured after a warmup batch, so
#          it excludes model loading, quantization and data loading.
#          fp16 arm: no transform and quantize=false, so nothing is quantized and the activations stay in 16
#          bits -- the model as it would be served.
#          W4A4 arm: the evolutionary A4 rotation (its recorded activation settings match this run's), GPTQ W4
#          symmetric in groups of 128 on 128 calibration utterances, scale recovery and weight recovery, the
#          inserted Conformer output Linears quantized at 4 bits, activations 4-bit symmetric per token except
#          attn_out and fc2 at one scale per group of 128. inference=humming replaces every
#          activation-quantized layer with a real ASRQLinear; weights do not affect the timing, so the small
#          calibration set costs nothing here, and the WER is reported only to show the model is not broken.
#          quantized_path=auto, so the second W4A4 run reuses the first one's weights instead of re-quantizing.
# Output:  results/evaluations/speed_w4a4_<model>_<arm>_b<batch>_.../{results.csv,config.yaml}. Every run is
#          logged to wandb in the group speed_w4a4. Read the RTFX lines out of the log.
#
# float16 matters: in bfloat16 every ASRQLinear converts its input and output, which cancels the speedup.
# At batch 1 a CUDA graph is captured per length bucket, and only the first bucket is warmed up, so some graph
# capture lands inside the timed region -- in both arms.
#
# Run a subset with ARMS=w4a4 BATCHES=1 bash scripts/speed_w4a4.sh
#
# Usage: bash scripts/speed_w4a4.sh [conda activate script] [project directory]
CONDA_ACTIVATE=${1:-/home/ubuntu/miniconda3/bin/activate}
PROJECT_DIR=${2:-/home/ubuntu/asrq}
source "$CONDA_ACTIVATE" asrq
cd "$PROJECT_DIR"
MODEL=${MODEL:-parakeet}
ROTATION=${ROTATION:-outputs/rotation/evo_parakeet_a4_mixed128.pt}
COMMON="model=$MODEL eval_dtype=float16 model.generate_fn=generate_parakeet_cuda_graphs"
COMMON="$COMMON eval_datasets=[{dataset:librispeech,split:test.clean}] create_audio_files=False"
EXPERIMENT="exp_name=speed_w4a4 wandb.enabled=true"
stamp() { echo "[$(date +%H:%M:%S)] $*"; }
# One batch of 128, or 64 batches of 1, so each arm transcribes a comparable amount of audio.
declare -A BATCHES_OF=( [1]=64 [128]=1 )

for batch in ${BATCHES:-1 128}; do
  for arm in ${ARMS:-fp16 w4a4}; do
    stamp "$MODEL $arm batch $batch start"
    if [ "$arm" = "fp16" ]; then
      python -u asrq/exp.py $COMMON $EXPERIMENT transform=none quantize=false activation_bits=16 \
        calibration.num_samples=16 quantized_path=null \
        model.eval_batch_size=$batch eval_batches=${BATCHES_OF[$batch]} method=fp16_b$batch 2>&1 | tr '\r' '\n'
    else
      python -u asrq/exp.py $COMMON $EXPERIMENT transform=rotation transform.path=$ROTATION \
        quantizer=gptq quantizer.bits=4 quantizer.symmetric=True quantizer.group_size=128 \
        quantizer.scale_recovery=true quantizer.weight_recovery=true \
        model.quantize_block_output_linear=true quantizer.block_output_linear_bits=4 \
        activation_bits=4 activation_symmetric=True activation_group_size=128 \
        inference=humming calibration.num_samples=128 quantized_path=auto \
        model.eval_batch_size=$batch eval_batches=${BATCHES_OF[$batch]} method=w4a4_b$batch 2>&1 | tr '\r' '\n'
    fi
    stamp "$MODEL $arm batch $batch end"
  done
done
stamp "done"
