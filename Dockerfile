# ─────────────────────────────────────────────────────────────
#  Dockerfile  (save as Dockerfile and build with `docker build .`)
# ─────────────────────────────────────────────────────────────
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# — 防御的オプション —
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# -------------------------------------------------------------
#  OS パッケージ類
# -------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv python3.10-distutils \
        build-essential git curl ca-certificates libsndfile1 && \
    ln -s /usr/bin/python3.10 /usr/local/bin/python && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python && \
    rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6

# -------------------------------------------------------------
#  Python パッケージ
#   • PyTorch (CUDA 対応 wheel)
#   • torchaudio
#   • faiss-gpu (公式 wheelページ経由)
# -------------------------------------------------------------
  RUN pip install --no-cache-dir --upgrade pip wheel setuptools \
  && pip install --no-cache-dir \
       torch==2.2.0+cu121 \
       torchaudio==2.2.0+cu121 \
       --index-url https://download.pytorch.org/whl/cu121 \
  && pip install --no-cache-dir \
       faiss-gpu -f https://faiss.ai/whl/ \
       numpy pandas tqdm pytorch_lightning==2.2.0 \
       librosa soundfile
  RUN pip install --no-cache-dir \
      pillow \
      "matplotlib[required]" \
      transformers
  RUN pip install --no-cache-dir ffmpeg-python
# デフォルト作業ディレクトリ
WORKDIR /
#CMD [\"python\", \"-c\", \"print('Docker image ready. Mount your code with -v and run your scripts')\"]
CMD ["bash"]
