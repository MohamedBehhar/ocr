import base64
import json
import logging
import os
import re
import threading
import time
from datetime import datetime, timezone

import cv2
import easyocr
import numpy as np
import pdf2image
import pika
from fastapi import FastAPI

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
RABBITMQ_HOST    = os.getenv("RABBITMQ_HOST", "rabbitmq")
RABBITMQ_PORT    = int(os.getenv("RABBITMQ_PORT", "5672"))
RABBITMQ_USER    = os.getenv("RABBITMQ_USER", "bankdocs")
RABBITMQ_PASS    = os.getenv("RABBITMQ_PASS", "bankdocs123")
BACKEND_QUEUE    = os.getenv("BACKEND_QUEUE", "document-events")
KYC_EXCHANGE     = os.getenv("KYC_EXCHANGE", "kyc-events")
LLM_WORKER_QUEUE = os.getenv("LLM_WORKER_QUEUE", "worker-llm-events")
GATEWAY_QUEUE    = os.getenv("GATEWAY_QUEUE", "gateway-document-events")
RECONNECT_DELAY  = int(os.getenv("RECONNECT_DELAY", "5"))

# ── Routing keys ──────────────────────────────────────────────────────────────
RK_DOCUMENT_UPLOADED      = "document.uploaded"
RK_DOCUMENT_OCR_COMPLETED = "document.ocr.completed"
RK_DOCUMENT_OCR_FAILED    = "document.ocr.failed"

app = FastAPI()

# Lazy singleton — initialized inside consumer thread to avoid blocking uvicorn startup
_reader: easyocr.Reader | None = None
_reader_lock = threading.Lock()

def get_reader() -> easyocr.Reader:
    global _reader
    if _reader is None:
        with _reader_lock:
            if _reader is None:
                logger.info("Initializing EasyOCR (ar, en)…")
                _reader = easyocr.Reader(['ar', 'en'], gpu=False)
                logger.info("EasyOCR ready.")
    return _reader

LATIN_KEYWORDS = {
    'ROYAUME', 'MAROC', 'CARTE', 'NATIONALE', 'IDENTITE', 'DIDENTIIE',
    'DIDENTITE', 'DIDENTIEE', 'DU', 'NE', 'NEE', 'AN', 'NO', 'VALABLE',
}


# ── OCR helpers ───────────────────────────────────────────────────────────────

def run_ocr_on_image(img: np.ndarray) -> str:
    results = get_reader().readtext(img, detail=0)
    return "\n".join(results)


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
    cin_m = re.search(r'\b([A-Z]{1,2})\s?(\d{6})\b', full)
    if cin_m:
        num_piece = cin_m.group(1) + cin_m.group(2)
    else:
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


# ── RabbitMQ topology ─────────────────────────────────────────────────────────

def setup_topology(channel):
    logger.info("Declaring queue: %s", BACKEND_QUEUE)
    channel.queue_declare(queue=BACKEND_QUEUE, durable=True)
    logger.info("Declaring exchange: %s (topic)", KYC_EXCHANGE)
    channel.exchange_declare(exchange=KYC_EXCHANGE, exchange_type="topic", durable=True)
    logger.info("Declaring queue: %s", LLM_WORKER_QUEUE)
    channel.queue_declare(queue=LLM_WORKER_QUEUE, durable=True)
    channel.queue_bind(queue=LLM_WORKER_QUEUE, exchange=KYC_EXCHANGE,
                       routing_key=RK_DOCUMENT_OCR_COMPLETED)
    logger.info("Bound %s -> %s [%s]", LLM_WORKER_QUEUE, KYC_EXCHANGE, RK_DOCUMENT_OCR_COMPLETED)
    logger.info("Declaring queue: %s", GATEWAY_QUEUE)
    channel.queue_declare(queue=GATEWAY_QUEUE, durable=True)
    channel.queue_bind(queue=GATEWAY_QUEUE, exchange=KYC_EXCHANGE,
                       routing_key=RK_DOCUMENT_OCR_COMPLETED)
    channel.queue_bind(queue=GATEWAY_QUEUE, exchange=KYC_EXCHANGE,
                       routing_key=RK_DOCUMENT_OCR_FAILED)
    logger.info("Bound %s -> %s [%s, %s]", GATEWAY_QUEUE, KYC_EXCHANGE,
                RK_DOCUMENT_OCR_COMPLETED, RK_DOCUMENT_OCR_FAILED)
    logger.info("Topology setup complete")


def publish_event(channel, routing_key: str, data: dict):
    body = json.dumps({"pattern": routing_key, "data": data})
    channel.basic_publish(
        exchange=KYC_EXCHANGE,
        routing_key=routing_key,
        properties=pika.BasicProperties(content_type="application/json", delivery_mode=2),
        body=body,
    )


# ── Message handler ───────────────────────────────────────────────────────────

def on_message(channel, method, _properties, body):
    document_id   = ""
    folder_id     = ""
    document_type = ""
    acked         = False

    try:
        outer = json.loads(body)
        pattern = outer.get("pattern")

        if pattern != RK_DOCUMENT_UPLOADED:
            logger.warning("Ignoring unexpected pattern: %s", pattern)
            channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
            acked = True
            return

        data = outer.get("data", {})
        document_id        = data["documentId"]
        folder_id          = data["folderId"]
        document_type      = data["documentType"]
        file_content       = data["fileContent"]
        mime_type          = data.get("mimeType", "")
        original_file_name = data.get("originalFileName", "")

        logger.info("OCR start: document=%s folder=%s type=%s", document_id, folder_id, document_type)

        file_bytes = base64.b64decode(file_content)
        extracted_text = run_ocr_on_bytes(file_bytes, mime_type)
        logger.info("OCR done: document=%s (%d chars)", document_id, len(extracted_text))

        publish_event(channel, RK_DOCUMENT_OCR_COMPLETED, {
            "documentId":       document_id,
            "folderId":         folder_id,
            "documentType":     document_type,
            "originalFileName": original_file_name,
            "extractedText":    extracted_text,
            "completedAt":      datetime.now(timezone.utc).isoformat(),
        })
        logger.info("Published %s for document=%s", RK_DOCUMENT_OCR_COMPLETED, document_id)

    except Exception:
        logger.exception("Error processing document=%s", document_id)
        try:
            publish_event(channel, RK_DOCUMENT_OCR_FAILED, {
                "documentId":   document_id,
                "folderId":     folder_id,
                "documentType": document_type,
                "error":        "OCR processing failed",
                "failedAt":     datetime.now(timezone.utc).isoformat(),
            })
        except Exception:
            logger.exception("Failed to publish OCR failed event")
    finally:
        if not acked:
            try:
                channel.basic_ack(delivery_tag=method.delivery_tag)
            except Exception:
                pass


# ── Consumer loop ─────────────────────────────────────────────────────────────

def _run_consumer():
    logger.info(
        "Consumer config — host=%s port=%d user=%s backend_queue=%s "
        "exchange=%s llm_queue=%s gateway_queue=%s",
        RABBITMQ_HOST, RABBITMQ_PORT, RABBITMQ_USER,
        BACKEND_QUEUE, KYC_EXCHANGE, LLM_WORKER_QUEUE, GATEWAY_QUEUE,
    )
    while True:
        try:
            logger.info("Connecting to RabbitMQ at %s:%d…", RABBITMQ_HOST, RABBITMQ_PORT)
            credentials = pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASS)
            params = pika.ConnectionParameters(
                host=RABBITMQ_HOST,
                port=RABBITMQ_PORT,
                credentials=credentials,
                heartbeat=600,
                blocked_connection_timeout=300,
            )
            connection = pika.BlockingConnection(params)
            logger.info("RabbitMQ connection established")
            channel = connection.channel()
            setup_topology(channel)
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


@app.get("/health")
def health():
    return {"status": "ok"}
