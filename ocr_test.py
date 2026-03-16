"""
OCR Test Script — French & Arabic (photos of documents)
Uses: pytesseract + Pillow + OpenCV for preprocessing

Install dependencies:
    pip install pytesseract pillow opencv-python
    sudo apt install tesseract-ocr tesseract-ocr-fra tesseract-ocr-ara  # Linux
    brew install tesseract                                                # Mac (then install lang packs)

NOTE: Tesseract works okay for French but Arabic results can be inconsistent.
      For better Arabic OCR, consider EasyOCR or PaddleOCR in production.
"""

import pytesseract
from PIL import Image
import cv2
import numpy as np
import os
import sys


# ─────────────────────────────────────────────
# 1. IMAGE PREPROCESSING
# Better image quality = better OCR accuracy
# ─────────────────────────────────────────────

def preprocess_image(image_path: str) -> np.ndarray:
    """
    Preprocess a photo of a document to improve OCR accuracy.
    Steps: grayscale → denoise → threshold (binarize) → deskew
    """
    img = cv2.imread(image_path)

    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    # Step 1: Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 2: Denoise (helps with photo noise)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)

    # Step 3: Adaptive thresholding (handles uneven lighting in photos)
    thresh = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31,
        C=10
    )

    # Step 4: Deskew (straighten slightly tilted photos)
    thresh = deskew(thresh)

    return thresh


def deskew(image: np.ndarray) -> np.ndarray:
    """Detect and correct slight rotation in a document photo."""
    coords = np.column_stack(np.where(image < 128))  # find dark pixels
    if len(coords) == 0:
        return image
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    if abs(angle) < 0.5:  # skip if nearly straight
        return image
    h, w = image.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h),
                             flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)
    return rotated


# ─────────────────────────────────────────────
# 2. LANGUAGE CONFIG
# ─────────────────────────────────────────────

LANG_CONFIG = {
    "french":  {"lang": "fra",     "config": "--psm 3"},
    "arabic":  {"lang": "ara",     "config": "--psm 3 --oem 1"},  # oem 1 = LSTM only
    "mixed":   {"lang": "fra+ara", "config": "--psm 3 --oem 1"},
}


# ─────────────────────────────────────────────
# 3. OCR FUNCTIONS
# ─────────────────────────────────────────────

def run_ocr(image_path: str, language: str = "french") -> dict:
    """
    Run OCR on a document photo.

    Args:
        image_path: Path to the image file (jpg, png, etc.)
        language:   "french", "arabic", or "mixed"

    Returns:
        dict with extracted text and metadata
    """
    if language not in LANG_CONFIG:
        raise ValueError(f"Language must be one of: {list(LANG_CONFIG.keys())}")

    cfg = LANG_CONFIG[language]

    # Preprocess
    print(f"  → Preprocessing image...")
    processed = preprocess_image(image_path)
    pil_image = Image.fromarray(processed)

    # Run OCR
    print(f"  → Running Tesseract ({cfg['lang']})...")
    raw_text = pytesseract.image_to_string(pil_image, lang=cfg["lang"], config=cfg["config"])

    # Get confidence data (word-level)
    data = pytesseract.image_to_data(pil_image, lang=cfg["lang"],
                                     config=cfg["config"],
                                     output_type=pytesseract.Output.DICT)

    confidences = [int(c) for c in data["conf"] if str(c).isdigit() and int(c) >= 0]
    avg_conf = round(sum(confidences) / len(confidences), 1) if confidences else 0

    return {
        "text": raw_text.strip(),
        "language": language,
        "avg_confidence": avg_conf,
        "word_count": len(raw_text.split()),
        "image_path": image_path,
    }


def ocr_multiple(image_paths: list, language: str = "french") -> list:
    """Run OCR on a list of images."""
    results = []
    for path in image_paths:
        print(f"\nProcessing: {path}")
        try:
            result = run_ocr(path, language)
            results.append(result)
        except Exception as e:
            results.append({"image_path": path, "error": str(e)})
    return results


def print_result(result: dict):
    """Pretty-print a single OCR result."""
    print("\n" + "=" * 60)
    print(f"  File      : {result.get('image_path', 'N/A')}")
    if "error" in result:
        print(f"  ❌ Error  : {result['error']}")
    else:
        print(f"  Language  : {result['language']}")
        print(f"  Words     : {result['word_count']}")
        print(f"  Confidence: {result['avg_confidence']}%")
        print(f"\n  Extracted Text:\n  {'─'*40}")
        for line in result["text"].splitlines():
            if line.strip():
                print(f"  {line}")
    print("=" * 60)


# ─────────────────────────────────────────────
# 4. DEMO / MAIN
# ─────────────────────────────────────────────

def demo_with_sample():
    """
    Generate a simple test image with text and run OCR on it.
    Useful for verifying your Tesseract installation works.
    """
    from PIL import ImageDraw, ImageFont

    print("\n[DEMO] Generating a sample French test image...")

    # Create a white image with French text
    img = Image.new("RGB", (800, 300), color="white")
    draw = ImageDraw.Draw(img)

    sample_text = (
        "Bonjour! Ceci est un test de reconnaissance optique de caractères.\n"
        "Le document contient du texte en français.\n"
        "Date: 15 mars 2026"
    )

    # Draw text (using default font — install a better font for real use)
    draw.multiline_text((40, 60), sample_text, fill="black", spacing=20)

    sample_path = "/tmp/sample_french.png"
    img.save(sample_path)
    print(f"  Sample image saved to: {sample_path}")

    result = run_ocr(sample_path, language="french")
    print_result(result)


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        # Usage: python ocr_test.py my_image.jpg french
        image_path = sys.argv[1]
        lang = sys.argv[2] if len(sys.argv) >= 3 else "french"
        print(f"\nRunning OCR on: {image_path} (language: {lang})")
        result = run_ocr(image_path, language=lang)
        print_result(result)
    else:
        # Run built-in demo
        demo_with_sample()

        print("\n" + "─" * 60)
        print("To run on your own image:")
        print("  python ocr_test.py path/to/your/image.jpg french")
        print("  python ocr_test.py path/to/your/image.jpg arabic")
        print("  python ocr_test.py path/to/your/image.jpg mixed")
        print("─" * 60)