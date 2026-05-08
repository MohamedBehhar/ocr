FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxrender1 libxext6 libgl1 curl \
    poppler-utils libgomp1 ccache \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download models + trigger C++ compilation at build time
RUN python -c "\
import numpy as np; \
from paddleocr import PaddleOCR; \
img = np.ones((100, 300, 3), dtype=np.uint8) * 255; \
ocr_ar = PaddleOCR(use_angle_cls=True, lang='arabic', use_gpu=False, show_log=False); \
ocr_ar.ocr(img, cls=True); \
ocr_fr = PaddleOCR(use_angle_cls=True, lang='french', use_gpu=False, show_log=False); \
ocr_fr.ocr(img, cls=True)"

COPY main.py .

EXPOSE 5000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000"]
