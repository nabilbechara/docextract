"""Primary engine: a local vision-language model (free, offline).

Reads printed text, handwriting, stamps, tables and arbitrary layouts, and returns JSON
constrained to our schema. Talks to either Ollama's native API or any OpenAI-compatible
server (llama.cpp's llama-server, vLLM, LM Studio, ...). Several endpoints can be listed
for failover: DOCEXTRACT_VLM_URLS="http://gpu-box:11434,http://localhost:8080".
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time

import httpx
from PIL import Image

from ..preprocess import to_jpeg_bytes
from ..schema import llm_json_schema

log = logging.getLogger(__name__)

MODEL = os.environ.get("DOCEXTRACT_MODEL", "qwen3-vl:4b")
URLS = os.environ.get("DOCEXTRACT_VLM_URLS", os.environ.get("OLLAMA_URL", "http://localhost:11434,http://localhost:8080"))
TIMEOUT = float(os.environ.get("DOCEXTRACT_VLM_TIMEOUT", "300"))
MAX_TOKENS = int(os.environ.get("DOCEXTRACT_MAX_TOKENS", "6144"))

PROMPT = """You are a meticulous document data-entry clerk. Read this document image \
(it may be a printed or HANDWRITTEN receipt, invoice, bill, delivery note, order form, \
statement, table or any other business document, in any language, possibly photographed \
at an angle, crumpled, faded or in poor light) and transcribe ALL of its data into JSON.

Rules:
- line_items: one entry per row of the document's item table / list, in order. Never merge, skip or repeat rows.
  - name: the item description exactly as written (keep the document's language; fix only obvious misreads).
  - quantity: the count/amount for that row as a number ("2x", "2 @", "qty 2", "x2" -> 2; 0.45 kg -> 0.45 with unit "kg"). null if not shown.
  - unit_price: price per unit. null if not shown.
  - total: the row's amount. null if not shown.
  - extra: EVERY other column or detail for that row (SKU, code, discount, VAT %, size, date, notes...) as key/value pairs, keyed by the column header.
- fields: EVERY other piece of information on the document that is not a line item: invoice/receipt number, \
customer, addresses, phone, tax ID, table, cashier, payment method, card last digits, change, tip, discounts, \
service charge, due date, terms, handwritten notes, stamps (e.g. "stamp": "PAID"), signature present, exchange rate, \
totals in a second currency, etc. Use the label printed on the document as key (or a short descriptive key if unlabeled). \
Leave nothing out.
- subtotal, tax, total: numbers as printed. total = the final amount to pay. If there are several taxes, \
tax = their sum and list each one in fields.
- Numbers: plain JSON numbers, no currency symbols or thousands separators, '.' as decimal point \
(read "1.234,50" as 1234.5 and "12,50" as 12.5 when the document uses decimal commas).
- currency: ISO code if you can tell (USD, EUR, LBP, GBP, AED, SAR...), else the symbol.
- date: YYYY-MM-DD if unambiguous, otherwise as written.
- document_type: one short word like receipt, invoice, bill, delivery_note, purchase_order, quote, statement, form, table, other.
- Handwriting: read carefully digit by digit; use the row arithmetic (quantity x unit_price = total) and the \
column sums to resolve ambiguous digits.
- Do not invent rows or values that are not on the document.
"""


class VLMError(RuntimeError):
    pass


class VLMEngine:
    name = "vlm"
    base_confidence = 0.95

    def __init__(self, model: str = MODEL, urls: str | list[str] = URLS, timeout: float = TIMEOUT):
        self.model = model
        self.urls = [u.strip().rstrip("/") for u in (urls.split(",") if isinstance(urls, str) else urls) if u.strip()]
        self.client = httpx.Client(timeout=timeout)
        self._kind: dict[str, str] = {}  # url -> "ollama" | "openai"
        self._checked = 0.0
        self._live: list[str] = []

    # ---- discovery -------------------------------------------------------------------
    def _probe(self, url: str) -> str | None:
        try:
            r = self.client.get(f"{url}/api/tags", timeout=3)
            if r.status_code == 200 and "models" in r.json():
                names = {m["name"] for m in r.json()["models"]}
                if self.model in names or f"{self.model}:latest" in names:
                    return "ollama"
                log.warning("ollama at %s does not have model %s (has %s)", url, self.model, sorted(names))
                return None
        except Exception:  # noqa: BLE001
            pass
        try:
            r = self.client.get(f"{url}/v1/models", timeout=3)
            if r.status_code == 200:
                return "openai"
        except Exception:  # noqa: BLE001
            pass
        return None

    def live_urls(self, max_age: float = 15.0) -> list[str]:
        if time.monotonic() - self._checked > max_age:
            live = []
            for u in self.urls:
                kind = self._probe(u)
                if kind:
                    self._kind[u] = kind
                    live.append(u)
            self._live, self._checked = live, time.monotonic()
        return self._live

    def available(self) -> bool:
        return bool(self.live_urls())

    def describe(self) -> list[str]:
        return [f"{self._kind[u]}@{u}" for u in self.live_urls()]

    # ---- inference -------------------------------------------------------------------
    def extract(self, img: Image.Image, feedback: str | None = None, ocr_hint: str | None = None, hot: bool = False) -> dict:
        prompt = PROMPT
        if ocr_hint:
            prompt += (
                "\nA text-extraction pass produced the text below (reading order may be off, it may contain "
                "errors and it misses handwriting). Trust the image over it, but use it to double-check printed "
                f"text and numbers:\n<text>\n{ocr_hint[:6000]}\n</text>\n"
            )
        if feedback:
            prompt += (
                "\nA previous reading of this document failed these consistency checks. Look again carefully "
                "at the relevant rows and digits and correct the reading (only if the document really says so; "
                "printed totals can legitimately include discounts or fees):\n- " + "\n- ".join(feedback.splitlines()) + "\n"
            )
        b64 = base64.b64encode(to_jpeg_bytes(img, 92)).decode()
        temperature = 0.3 if hot else 0.0
        errors = []
        urls = self.live_urls() or self.live_urls(max_age=0)
        if not urls:
            raise VLMError("no vision model server reachable at " + ", ".join(self.urls))
        for url in urls:
            try:
                if self._kind[url] == "ollama":
                    content, truncated = self._ollama(url, prompt, b64, temperature)
                else:
                    content, truncated = self._openai(url, prompt, b64, temperature)
                if truncated:
                    raise VLMError("model output was cut off (likely a repetition loop)")
                return parse_json_loose(content)
            except Exception as e:  # noqa: BLE001 - try the next endpoint
                errors.append(f"{url}: {e}")
                log.warning("vlm call failed on %s: %s", url, e)
                if isinstance(e, (httpx.ConnectError, httpx.ConnectTimeout)):
                    self._checked = 0  # force re-probe next time
        raise VLMError("; ".join(errors))

    def _ollama(self, url: str, prompt: str, b64: str, temperature: float) -> tuple[str, bool]:
        body = {
            "model": self.model,
            "stream": False,
            "format": llm_json_schema(),
            "keep_alive": os.environ.get("DOCEXTRACT_KEEP_ALIVE", "30m"),
            "options": {
                "temperature": temperature,
                "num_ctx": int(os.environ.get("DOCEXTRACT_NUM_CTX", "12288")),
                "num_predict": MAX_TOKENS,
                "repeat_penalty": 1.05,
            },
            "messages": [{"role": "user", "content": prompt, "images": [b64]}],
        }
        r = self.client.post(f"{url}/api/chat", json=body)
        r.raise_for_status()
        d = r.json()
        return d["message"]["content"], d.get("done_reason") == "length"

    def _openai(self, url: str, prompt: str, b64: str, temperature: float) -> tuple[str, bool]:
        body = {
            "model": self.model,
            "temperature": temperature,
            "max_tokens": MAX_TOKENS,
            "response_format": {"type": "json_schema", "json_schema": {"name": "extraction", "schema": llm_json_schema()}},
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        }
        r = self.client.post(f"{url}/v1/chat/completions", json=body)
        r.raise_for_status()
        ch = r.json()["choices"][0]
        return ch["message"]["content"], ch.get("finish_reason") == "length"


def parse_json_loose(text: str) -> dict:
    """Parse model output, tolerating code fences or trailing junk."""
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{") :]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise
