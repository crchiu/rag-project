FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-chi-tra \
    tesseract-ocr-eng \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# ✅ 1) 先清掉 conda / pip 既有 transformers (以及常一起帶來的依賴)
#    避免最後 import 的仍是 /opt/conda/... 那套舊版
RUN python -c "import sys; print(sys.executable)" \
 && pip uninstall -y transformers tokenizers safetensors huggingface-hub accelerate peft || true \
 && rm -rf /opt/conda/lib/python3.11/site-packages/transformers* \
           /opt/conda/lib/python3.11/site-packages/tokenizers* \
           /opt/conda/lib/python3.11/site-packages/safetensors* || true

# ✅ 2) 重新安裝（你 requirements 裡的 git transformers 會生效）
RUN pip install --no-cache-dir -r requirements.txt

# ✅ 3) build-time hard check：必須有 SlidingWindowCache，否則直接讓 build 失敗
RUN python -c "import transformers, transformers.cache_utils as c; print('transformers=', transformers.__version__); print('SlidingWindowCache=', hasattr(c,'SlidingWindowCache')); assert hasattr(c,'SlidingWindowCache')"

COPY app ./app

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]