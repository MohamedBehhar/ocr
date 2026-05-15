FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxrender1 libxext6 libgl1 curl \
    poppler-utils libgomp1 ccache \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Install CPU-only PyTorch first to avoid pulling in 2GB of CUDA libraries
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download EasyOCR models at build time so first request is instant
RUN python -c "\
import numpy as np; \
import easyocr; \
reader = easyocr.Reader(['ar', 'en'], gpu=False); \
img = np.ones((100, 300, 3), dtype=np.uint8) * 255; \
reader.readtext(img)"

COPY main.py .

EXPOSE 5000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000"]
