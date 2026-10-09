"""HTTP API for backends.

POST /v1/extract          multipart upload (field 'file')
POST /v1/extract/base64   JSON {"filename": "...", "data": "<base64>"}
GET  /health

Auth: set DOCEXTRACT_API_KEYS="key1,key2" and send 'Authorization: Bearer <key>' or 'X-API-Key: <key>'.
"""

from __future__ import annotations

import base64
import binascii
import hmac
import logging
import os
import threading
from typing import Literal

from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, UploadFile
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from .pipeline import Extractor
from .preprocess import UnsupportedDocument
from .schema import Extraction

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
log = logging.getLogger("docextract.api")

MAX_BYTES = int(os.environ.get("DOCEXTRACT_MAX_MB", "25")) * 1024 * 1024
API_KEYS = [k.strip() for k in os.environ.get("DOCEXTRACT_API_KEYS", "").split(",") if k.strip()]
# The GPU handles one image at a time well; queue the rest instead of thrashing VRAM.
_slots = threading.BoundedSemaphore(int(os.environ.get("DOCEXTRACT_CONCURRENCY", "1")))

app = FastAPI(title="docextract", version="0.1.0", description="Extract line items and fields from any bill/receipt/invoice image or PDF.")
extractor = Extractor()

if not API_KEYS:
    log.warning("DOCEXTRACT_API_KEYS is not set: the API is unauthenticated. Set it before exposing the service.")

Engine = Literal["auto", "vlm", "ocr"]


def auth(authorization: str | None = Header(None), x_api_key: str | None = Header(None)) -> None:
    if not API_KEYS:
        return
    supplied = x_api_key or (authorization[7:] if authorization and authorization.lower().startswith("bearer ") else None)
    if not supplied or not any(hmac.compare_digest(supplied, k) for k in API_KEYS):
        raise HTTPException(401, "invalid or missing API key")


class Base64Request(BaseModel):
    data: str
    filename: str = ""
    engine: Engine = "auto"


def _run(data: bytes, filename: str, engine: str, include_text: bool, fmt: str):
    if len(data) > MAX_BYTES:
        raise HTTPException(413, f"file larger than {MAX_BYTES // (1024 * 1024)} MB")
    if not _slots.acquire(timeout=float(os.environ.get("DOCEXTRACT_QUEUE_TIMEOUT", "600"))):
        raise HTTPException(503, "server busy, retry later")
    try:
        ex = extractor.extract_bytes(data, filename, engine)
    except UnsupportedDocument as e:
        raise HTTPException(415, str(e)) from e
    except RuntimeError as e:
        raise HTTPException(503, str(e)) from e
    finally:
        _slots.release()
    if not include_text:
        ex.raw_text = None
    if fmt == "text":
        return PlainTextResponse(ex.readable())
    return ex


@app.get("/health")
def health() -> dict:
    h = extractor.health()
    h["status"] = "ok" if (h["vlm"] or h["ocr"]) else "down"
    h["degraded"] = not h["vlm"]
    return h


@app.post("/v1/extract", response_model=Extraction, dependencies=[Depends(auth)])
def extract(
    file: UploadFile = File(...),
    engine: Engine = Query("auto"),
    include_text: bool = Query(False, description="include OCR raw text"),
    format: Literal["json", "text"] = Query("json"),
):
    data = file.file.read(MAX_BYTES + 1)
    return _run(data, file.filename or "", engine, include_text, format)


@app.post("/v1/extract/base64", response_model=Extraction, dependencies=[Depends(auth)])
def extract_b64(
    req: Base64Request,
    include_text: bool = Query(False),
    format: Literal["json", "text"] = Query("json"),
):
    payload = req.data.split(",", 1)[1] if req.data.startswith("data:") else req.data
    try:
        data = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as e:
        raise HTTPException(400, "data is not valid base64") from e
    return _run(data, req.filename, req.engine, include_text, format)
