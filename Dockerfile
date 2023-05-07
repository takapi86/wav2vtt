FROM nvidia/cuda:11.6.1-devel-ubuntu20.04

RUN apt update && \
    DEBIAN_FRONTEND=noninteractive apt -y install python3 python3-pip curl git ffmpeg && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install git+https://github.com/reazon-research/reazonspeech.git

RUN pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116

RUN pip install packaging wheel pydub

WORKDIR /app

COPY . .

CMD ["python3", "transcribe_wav_to_vtt.py"]
