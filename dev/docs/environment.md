


conda create -n asrq python=3.12
pip install nemo_toolkit[asr]
pip install  torch torchvision --index-url https://download.pytorch.org/whl/cu126

export PYTHONPATH=/home/ubuntu/asrq/third_party/humming:/home/ubuntu/asrq/third_party/open_asr_leaderboard:$PYTHONPATH

pip install peft==0.20.0 evaluate==0.4.6 jiwer==4.0.0 datasets==5.0.1 torchcodec==0.16.0 tqdm wandb num2words
pip install --force-reinstall "torchcodec==0.16.0+cpu" --index-url https://download.pytorch.org/whl/cpu
conda install -n asrq -c conda-forge ffmpeg=7.1.1
pip install ninja

## Generate the Calibration Data
python -m asrq.calibration.build --model openai/whisper-large-v3 \
    --calibration-dir outputs/calibration/librispeech_train_clean_360_2048
python -m asrq.calibration.build --model nvidia/parakeet-ctc-1.1b \
    --calibration-dir outputs/calibration/librispeech_train_clean_360_2048
python -m asrq.calibration.build --model nvidia/canary-qwen-2.5b \
    --calibration-dir outputs/calibration/librispeech_train_clean_360_2048

