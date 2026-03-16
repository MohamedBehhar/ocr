# ─────────────────────────────────────────────
# OCR App — French & Arabic support
# Base: Python 3.11 slim + Tesseract
# ─────────────────────────────────────────────

FROM python:3.11-slim

# Avoid interactive prompts during apt install
ENV DEBIAN_FRONTEND=noninteractive

# ── System dependencies ──────────────────────
RUN apt-get update && apt-get install -y \
    # Tesseract OCR engine
    tesseract-ocr \
    # Language packs
    tesseract-ocr-fra \
    tesseract-ocr-ara \
    # OpenCV system libs
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    # Cleanup
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ────────────────────────
WORKDIR /app

# ── Python dependencies ──────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── App code ────────────────────────────────
COPY . .

# ── Input/output folders ─────────────────────
RUN mkdir -p /app/input /app/output

# ── Default command ──────────────────────────
CMD ["python", "ocr_test.py"]