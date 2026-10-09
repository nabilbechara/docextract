# docextract

Extract line items and fields (quantity, item name, unit price, totals, invoice numbers, …) from
photos and scans of receipts, invoices and bills, printed or handwritten, as JSON over an HTTP API.
Runs fully local and free: no paid APIs.

> **Status: work in progress.** The pipeline, API and OCR fallback work and are tested. The vision
> model engine is wired up, but it has not been benchmarked yet, and Docker deployment isn't done yet.

## How it works

1. **Preprocess:** images (JPEG/PNG/WebP/TIFF/HEIC…) or PDFs (multi-page, embedded text layer used
   when present); EXIF rotation; resizing to a pixel budget so long receipts stay legible.
2. **Auto-orient:** sideways and upside-down pages are detected from OCR and rotated upright.
3. **Read:** a local vision-language model (default `qwen3-vl:4b`) returns schema-constrained JSON.
   It works with Ollama or any OpenAI-compatible server (llama.cpp `llama-server`, vLLM, LM Studio),
   with failover across several URLs.
4. **Verify:** arithmetic checks (qty × price = line total, rows add up to subtotal/total), plus a
   cross-check against an independent OCR reading to catch missed rows. Failed checks are fed back to
   the model for up to 3 attempts.
5. **Fallback:** if no model server is reachable, RapidOCR plus rule-based parsing still returns a
   result, flagged with lower confidence.

Every result includes `confidence` (0–1) and `warnings`.

## Run

```bash
python -m venv .venv
.venv/Scripts/pip install fastapi "uvicorn[standard]" python-multipart pydantic httpx pillow pypdfium2 rapidocr-onnxruntime numpy
```

Start a model server, for example llama.cpp with the GGUF weights from `Qwen/Qwen3-VL-4B-Instruct-GGUF`:

```bash
llama-server -m models/Qwen3VL-4B-Instruct-Q4_K_M.gguf --mmproj models/mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf --port 8080 -ngl 99
```

or with Ollama:

```bash
ollama pull qwen3-vl:4b
```

Then start the API:

```bash
DOCEXTRACT_API_KEYS=change-me .venv/Scripts/uvicorn docextract.api:app --port 8000
```

## API

```bash
curl -H "Authorization: Bearer change-me" -F file=@receipt.jpg http://localhost:8000/v1/extract
```

- `POST /v1/extract`: multipart `file`. Query: `engine=auto|vlm|ocr`, `format=json|text`, `include_text=true`.
- `POST /v1/extract/base64`: `{"data": "<base64>", "filename": "x.pdf"}`
- `GET /health`

`format=text` gives readable output:

```
item 1: item quantity 4 | item name Coffee | unit price 2 | item total 8
```

CLI: `python -m docextract.cli receipt.jpg [--json]`

## Configuration

| Env var | Default | |
|---|---|---|
| `DOCEXTRACT_API_KEYS` | (none = no auth) | comma-separated keys |
| `DOCEXTRACT_VLM_URLS` | `http://localhost:11434,http://localhost:8080` | model servers, tried in order |
| `DOCEXTRACT_MODEL` | `qwen3-vl:4b` | |
| `DOCEXTRACT_MAX_ATTEMPTS` | `3` | self-correction attempts per page |
| `DOCEXTRACT_CONCURRENCY` | `1` | parallel extractions (GPU-bound) |
| `DOCEXTRACT_MAX_MB` | `25` | upload limit |

## Tests and evaluation

```bash
.venv/Scripts/python -m pytest -q tests
python scripts/make_synthetic.py data/synthetic 30   # printed / handwritten / photographed bills with ground truth
python scripts/fetch_real.py data/real/cord 25       # real receipt photos (CORD dataset)
python scripts/evaluate.py data/synthetic --engine auto
```

Current numbers (OCR fallback only): synthetic set 77% of documents fully correct, 97% of totals correct.
On real CORD photos the fallback is weak (44% item recall). That's the job of the vision model,
whose benchmark is next.
