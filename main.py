import json
import logging
import os
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone

import boto3
import cv2
import numpy as np
import pdf2image
import pika
import re
from botocore.exceptions import ClientError
from fastapi import FastAPI, File, UploadFile, HTTPException
from paddleocr import PaddleOCR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── RabbitMQ ──────────────────────────────────────────────────────────────────
RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "rabbitmq")
RABBITMQ_PORT = int(os.getenv("RABBITMQ_PORT", "5672"))
RABBITMQ_USER = os.getenv("RABBITMQ_USER", "bankdocs")
RABBITMQ_PASS = os.getenv("RABBITMQ_PASS", "bankdocs123")
BACKEND_QUEUE = os.getenv("BACKEND_QUEUE", "document-events")
LLM_QUEUE = os.getenv("LLM_QUEUE", "llm-events")
RECONNECT_DELAY = int(os.getenv("RECONNECT_DELAY", "5"))

# ── MinIO ─────────────────────────────────────────────────────────────────────
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minio")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minio123")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "documents")

# ── Routing keys (must match backend rabbitmq-contracts.ts) ───────────────────
PATTERN_DOCUMENT_UPLOADED = "document.uploaded"
PATTERN_FOLDER_OCR_COMPLETED = "folder.ocr.completed"
PATTERN_DOCUMENT_OCR_FAILED = "document.ocr.failed"

# ── Folder tracking: wait for all 3 docs before notifying LLM ────────────────
DOCS_PER_FOLDER = int(os.getenv("DOCS_PER_FOLDER", "3"))
_folder_docs: dict[str, list[dict]] = defaultdict(list)
_folder_lock = threading.Lock()

app = FastAPI()
ocr = PaddleOCR(use_angle_cls=True, lang="arabic", use_gpu=False, show_log=False)

s3 = boto3.client(
    "s3",
    endpoint_url=MINIO_ENDPOINT,
    aws_access_key_id=MINIO_ACCESS_KEY,
    aws_secret_access_key=MINIO_SECRET_KEY,
)

LATIN_KEYWORDS = {
    'ROYAUME', 'MAROC', 'CARTE', 'NATIONALE', 'IDENTITE', 'DIDENTIIE',
    'DIDENTITE', 'DIDENTIEE', 'DU', 'NE', 'NEE', 'AN', 'NO', 'VALABLE',
}


# ── OCR helpers ───────────────────────────────────────────────────────────────

def run_ocr_on_image(img: np.ndarray) -> str:
    results = ocr.ocr(img, cls=True)
    lines = []
    if results and results[0]:
        for line in results[0]:
            lines.append(line[1][0])
    return "\n".join(lines)


def run_ocr_on_bytes(data: bytes, mime_type: str = "") -> str:
    if mime_type == "application/pdf" or data[:4] == b"%PDF":
        images = pdf2image.convert_from_bytes(data, dpi=200)
        return "\n".join(
            run_ocr_on_image(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
            for img in images
        )
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image bytes")
    return run_ocr_on_image(img)


def extract_cin_fields(text: str) -> dict:
    lines = [l.strip() for l in text.split('\n') if l.strip()]
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
    vm = re.search(r'[Vv]alable[^\d]*(\d{2}\.\d{2}\.\d{4})', full)
    valable = vm.group(1) if vm else None
    all_dates = re.findall(r'\b(\d{2}\.\d{2}\.\d{4})\b', full)
    naissance = next((d for d in all_dates if d != valable and int(d.split('.')[2]) < 2010), None)
    # Try letter(s) + 6 digits (with optional space, e.g. "Q322874" or "Q 322874")
    cin_m = re.search(r'\b([A-Z]{1,2})\s?(\d{6})\b', full)
    if cin_m:
        num_piece = cin_m.group(1) + cin_m.group(2)
    else:
        # Prefer a standalone 6-digit line (CIN number prints alone on the card)
        standalone = [l for l in joined if re.match(r'^\d{6}$', l.strip())]
        if standalone:
            num_piece = standalone[-1].strip()
        else:
            m = re.search(r'\b(\d{6})\b', full)
            num_piece = m.group(1) if m else None
    name_lines = [
        l for l in joined
        if re.match(r'^[A-Z][A-Z\s\-]+$', l) and l.strip() not in LATIN_KEYWORDS and len(l.strip()) > 2
    ]
    return {
        "prenom":           name_lines[0] if len(name_lines) > 0 else None,
        "nom":              name_lines[1] if len(name_lines) > 1 else None,
        "date_naissance":   naissance,
        "ville":            name_lines[2] if len(name_lines) > 2 else None,
        "num_piece":        num_piece,
        "valable_jusqu_au": valable,
    }


# ── RabbitMQ helpers ──────────────────────────────────────────────────────────

def _publish(channel, queue: str, body: dict):
    channel.queue_declare(queue=queue, durable=True)
    channel.basic_publish(
        exchange="",
        routing_key=queue,
        properties=pika.BasicProperties(content_type="application/json", delivery_mode=2),
        body=json.dumps(body),
    )


def on_message(channel, method, _properties, body):
    try:
        outer = json.loads(body)
        pattern = outer.get("pattern")

        if pattern != PATTERN_DOCUMENT_UPLOADED:
            channel.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
            return

        data = outer.get("data", {})
        document_id = data["documentId"]
        folder_id = data["folderId"]
        document_type = data["documentType"]
        storage_key = data["storageKey"]
        mime_type = data.get("mimeType", "")
        original_file_name = data.get("originalFileName", "")

        logger.info("OCR start: document=%s folder=%s type=%s", document_id, folder_id, document_type)

        try:
            response = s3.get_object(Bucket=MINIO_BUCKET, Key=storage_key)
            file_bytes = response["Body"].read()
        except ClientError as e:
            raise RuntimeError(f"MinIO download failed for key={storage_key}: {e}")

        extracted_text = run_ocr_on_bytes(file_bytes, mime_type)
        logger.info("OCR done: document=%s (%d chars)", document_id, len(extracted_text))

        doc_entry = {
            "documentId": document_id,
            "documentType": document_type,
            "originalFileName": original_file_name,
            "extractedText": extracted_text,
        }

        with _folder_lock:
            _folder_docs[folder_id].append(doc_entry)
            docs_done = len(_folder_docs[folder_id])

        logger.info("Folder %s progress: %d/%d", folder_id, docs_done, DOCS_PER_FOLDER)

        if docs_done >= DOCS_PER_FOLDER:
            with _folder_lock:
                all_docs = _folder_docs.pop(folder_id)

            event = {
                "pattern": PATTERN_FOLDER_OCR_COMPLETED,
                "data": {
                    "folderId": folder_id,
                    "documents": all_docs,
                    "completedAt": datetime.now(timezone.utc).isoformat(),
                },
            }
            _publish(channel, LLM_QUEUE, event)
            logger.info("Folder %s complete — published to [%s]", folder_id, LLM_QUEUE)

    except Exception:
        logger.exception("Error processing document message")
        try:
            data = json.loads(body).get("data", {})
            _publish(channel, BACKEND_QUEUE, {
                "pattern": PATTERN_DOCUMENT_OCR_FAILED,
                "data": {
                    "documentId": data.get("documentId", ""),
                    "folderId": data.get("folderId", ""),
                    "documentType": data.get("documentType", ""),
                    "error": "OCR processing failed",
                    "failedAt": datetime.now(timezone.utc).isoformat(),
                },
            })
        except Exception:
            logger.exception("Failed to publish error event")
    finally:
        try:
            channel.basic_ack(delivery_tag=method.delivery_tag)
        except Exception:
            pass


def _run_consumer():
    while True:
        try:
            credentials = pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASS)
            params = pika.ConnectionParameters(
                host=RABBITMQ_HOST,
                port=RABBITMQ_PORT,
                credentials=credentials,
                heartbeat=600,
                blocked_connection_timeout=300,
            )
            connection = pika.BlockingConnection(params)
            channel = connection.channel()
            channel.queue_declare(queue=BACKEND_QUEUE, durable=True)
            channel.basic_qos(prefetch_count=1)
            channel.basic_consume(queue=BACKEND_QUEUE, on_message_callback=on_message)
            logger.info("OCR consumer ready — listening on [%s]", BACKEND_QUEUE)
            channel.start_consuming()
        except pika.exceptions.AMQPConnectionError as exc:
            logger.error("RabbitMQ connection lost (%s). Retrying in %ds…", exc, RECONNECT_DELAY)
            time.sleep(RECONNECT_DELAY)
        except Exception:
            logger.exception("Unexpected consumer error. Retrying in %ds…", RECONNECT_DELAY)
            time.sleep(RECONNECT_DELAY)


@app.on_event("startup")
async def startup_event():
    t = threading.Thread(target=_run_consumer, daemon=True)
    t.start()
    logger.info("RabbitMQ consumer thread started")


# ── REST endpoints (for direct testing) ───────────────────────────────────────

@app.post("/ocr")
async def run_ocr(file: UploadFile = File(...)):
    data = await file.read()
    try:
        extracted_text = run_ocr_on_bytes(data, file.content_type or "")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not process file: {e}")
    fields = extract_cin_fields(extracted_text)
    return {"extractedText": extracted_text, **fields}


@app.get("/health")
def health():
    return {"status": "ok"}
