from fastapi import FastAPI, File, UploadFile, HTTPException
from paddleocr import PaddleOCR
import numpy as np
import cv2
import pdf2image
import re

app = FastAPI()
ocr = PaddleOCR(use_angle_cls=True, lang="arabic", use_gpu=False, show_log=False)

LATIN_KEYWORDS = {
    'ROYAUME', 'MAROC', 'CARTE', 'NATIONALE', 'IDENTITE', 'DIDENTIIE',
    'DIDENTITE', 'DIDENTIEE', 'DU', 'NE', 'NEE', 'AN', 'NO', 'VALABLE',
}


def extract_cin_fields(text: str) -> dict:
    lines = [l.strip() for l in text.split('\n') if l.strip()]

    # Stitch split dates: "31.05" + "96" → "31.05.1996"
    joined: list[str] = []
    i = 0
    while i < len(lines):
        if re.match(r'^\d{2}\.\d{2}$', lines[i]) and i + 1 < len(lines) and re.match(r'^\d{2,4}$', lines[i + 1]):
            yr = lines[i + 1]
            if len(yr) == 2:
                yr = ('19' if int(yr) > 30 else '20') + yr
            joined.append(lines[i] + '.' + yr)
            i += 2
        else:
            joined.append(lines[i])
            i += 1

    full = ' '.join(joined)

    # Valable jusqu'au — date that follows the "Valable" keyword
    vm = re.search(r'[Vv]alable[^\d]*(\d{2}\.\d{2}\.\d{4})', full)
    valable = vm.group(1) if vm else None

    # Date de naissance — first complete date that's not the valable date
    all_dates = re.findall(r'\b(\d{2}\.\d{2}\.\d{4})\b', full)
    naissance = next((d for d in all_dates if d != valable and int(d.split('.')[2]) < 2010), None)

    # Numéro de pièce — letter(s) + 6 digits, fallback to just 6 digits
    cin = re.search(r'\b([A-Z]{1,2}\d{6})\b', full)
    if not cin:
        cin = re.search(r'\b(\d{6})\b', full)
    num_piece = cin.group(1) if cin else None

    # Names — all-caps Latin lines, filtered against known card keywords
    name_lines = [
        l for l in joined
        if re.match(r'^[A-Z][A-Z\s\-]+$', l)
        and l.strip() not in LATIN_KEYWORDS
        and len(l.strip()) > 2
    ]

    return {
        "prenom":          name_lines[0] if len(name_lines) > 0 else None,
        "nom":             name_lines[1] if len(name_lines) > 1 else None,
        "date_naissance":  naissance,
        "ville":           name_lines[2] if len(name_lines) > 2 else None,
        "num_piece":       num_piece,
        "valable_jusqu_au": valable,
    }


def run_ocr_on_image(img: np.ndarray) -> str:
    results = ocr.ocr(img, cls=True)
    lines = []
    if results and results[0]:
        for line in results[0]:
            lines.append(line[1][0])
    return "\n".join(lines)


@app.post("/ocr")
async def run_ocr(file: UploadFile = File(...)):
    data = await file.read()

    if file.content_type == "application/pdf" or (file.filename and file.filename.lower().endswith(".pdf")):
        try:
            images = pdf2image.convert_from_bytes(data, dpi=200)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Could not read PDF: {e}")
        all_text = []
        for pil_img in images:
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            all_text.append(run_ocr_on_image(img))
        extracted_text = "\n".join(all_text)

    elif file.content_type and file.content_type.startswith("image/"):
        arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Could not decode image")
        extracted_text = run_ocr_on_image(img)

    else:
        raise HTTPException(status_code=400, detail="File must be an image or PDF")

    fields = extract_cin_fields(extracted_text)
    return {"extractedText": extracted_text, **fields}


@app.get("/health")
def health():
    return {"status": "ok"}
