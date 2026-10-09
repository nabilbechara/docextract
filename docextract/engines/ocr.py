"""Fallback engine: RapidOCR (ONNX, CPU, pip-only) + rule-based receipt/invoice parsing.

Weaker than the VLM (and poor on handwriting) but has no external dependency, so the
service always returns *something* even if the model server is down.
"""

from __future__ import annotations

import re
import threading

import numpy as np
from PIL import Image

from ..validate import parse_number

_PRICE = r"-?\(?[$€£]?\s?\d{1,3}(?:[,.']\d{3})*(?:[.,]\d{1,2})\)?|-?\d+[.,]\d{1,2}|-?\d{1,3}(?:[,.]\d{3})+"
_PRICE_END = re.compile(rf"(?P<price>{_PRICE})\s*[A-Z]?\s*$")
_QTY_PREFIX = re.compile(r"^\s*(?P<q>\d+(?:[.,]\d+)?)\s*(?:x|X|×|\*|pcs?|@)?\s+(?P<rest>.+)$")
_QTY_AT = re.compile(rf"(?P<q>\d+(?:[.,]\d+)?)\s*(?:x|X|×|@)\s*(?P<p>{_PRICE})")
_DATE = re.compile(
    r"\b(\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}|"
    r"\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{2,4})\b",
    re.I,
)
_TOTAL = re.compile(r"\b(grand\s*total|total\s*(?:due|ttc|amount|à payer)?|amount\s*due|balance\s*due|net\s*to\s*pay|montant|sum)\b", re.I)
_SUBTOTAL = re.compile(r"\b(sub\s*-?\s*total|total\s*ht|net\s*amount)\b", re.I)
_TAX = re.compile(r"\b(tax|vat|tva|gst|hst|iva|mwst)\b", re.I)
_SKIP = re.compile(
    r"\b(change|cash|card|visa|master|tender|paid|payment|tip|rounding|balance|thank|receipt|tel|phone|www|invoice\s*no|"
    r"table|server|cashier|order|guest|items?\s*sold|qty\s+item|description|date|no\.)\b",
    re.I,
)
_CURRENCY = {"$": "USD", "€": "EUR", "£": "GBP", "LBP": "LBP", "L.L": "LBP", "USD": "USD", "EUR": "EUR"}


class OCREngine:
    name = "ocr"
    base_confidence = 0.6

    _lock = threading.Lock()
    _ocr = None

    def available(self) -> bool:
        try:
            self._get()
            return True
        except Exception:  # noqa: BLE001
            return False

    @classmethod
    def _get(cls):
        with cls._lock:
            if cls._ocr is None:
                from rapidocr_onnxruntime import RapidOCR

                cls._ocr = RapidOCR()
            return cls._ocr

    def _raw(self, img: Image.Image) -> list:
        # Low detection thresholds: receipts have tiny isolated tokens ("2", "x") that the
        # defaults drop, and those are exactly the quantities we need.
        result, _ = self._get()(np.asarray(img), box_thresh=0.2, unclip_ratio=2.0, text_score=0.3)
        return result or []

    @staticmethod
    def _quality(result: list) -> float:
        """How 'readable' an OCR result is: confident characters, horizontal boxes."""
        good = 0.0
        for box, text, score in result:
            w = abs(box[1][0] - box[0][0])
            h = abs(box[3][1] - box[0][1])
            good += float(score) ** 2 * len(text) * (1.0 if w >= h else 0.3)
        return good

    def _probe(self, img: Image.Image) -> list:
        # No angle classifier here: it silently fixes upside-down crops, hiding the signal.
        result, _ = self._get()(np.asarray(img), use_cls=False, box_thresh=0.2, unclip_ratio=2.0, text_score=0.3)
        return result or []

    def orient(self, img: Image.Image) -> tuple[Image.Image, int]:
        """Rotate sideways / upside-down pages upright. Returns (image, degrees rotated CCW)."""
        probe = img.copy()
        probe.thumbnail((1000, 1000))
        base = self._probe(probe)
        if not base:
            return img, 0
        deg = 0
        vertical = sum(1 for b, t, _ in base if len(t) > 2 and abs(b[3][1] - b[0][1]) > 1.5 * abs(b[1][0] - b[0][0]))
        if vertical > 0.4 * len(base):
            # Sideways: the right quarter-turn gives far better recognition than the wrong one.
            scored = {d: self._quality(self._probe(probe.rotate(d, expand=True))) for d in (90, 270)}
            d = max(scored, key=scored.get)
            if scored[d] > 1.15 * self._quality(base):
                deg = d
                probe = probe.rotate(d, expand=True)
                base = self._probe(probe)
        # Upside-down text is still recognised, but the layout is mirrored: totals and
        # 'thank you' end up at the top, dates / invoice numbers at the bottom.
        if _upside_down(base, probe.height):
            deg = (deg + 180) % 360
        return (img.rotate(deg, expand=True), deg) if deg else (img, 0)

    def lines(self, img: Image.Image) -> list[str]:
        result = self._raw(img)
        # Photos are rarely level: estimate the text skew from the box baselines and
        # rotate box positions back so same-row boxes share a y coordinate.
        angles = [
            np.arctan2(b[1][1] - b[0][1], b[1][0] - b[0][0])
            for b, _, _ in result
            if b[1][0] - b[0][0] > 2 * abs(b[3][1] - b[0][1])
        ]
        theta = float(np.median(angles)) if angles else 0.0
        c, s = np.cos(-theta), np.sin(-theta)
        boxes = []
        for box, text, _ in result:
            pts = np.asarray(box, dtype=float)
            ys = pts[:, 0] * s + pts[:, 1] * c
            xs = pts[:, 0] * c - pts[:, 1] * s
            boxes.append((ys.min(), ys.max(), xs.min(), text))
        return _group_rows(boxes)

    def text(self, img: Image.Image) -> str:
        return "\n".join(self.lines(img))

    def extract(self, img: Image.Image) -> dict:
        return parse_lines(self.lines(img))


_FOOTER = re.compile(r"\b(total|subtotal|thank|merci|shukran|signature|balance|amount\s*due|change|cash|tva|vat|tax)\b", re.I)
_HEADER = re.compile(r"\b(date|invoice|receipt|bill\s*to|no\.|tel|phone|order|table)\b", re.I)


def _upside_down(result: list, height: int) -> bool:
    foot, head = [], []
    for box, text, _ in result:
        y = sum(p[1] for p in box) / 4 / max(1, height)
        if _FOOTER.search(text):
            foot.append(y)
        elif _HEADER.search(text) or _DATE.search(text):
            head.append(y)
    if not foot and not head:
        return False
    f = sum(foot) / len(foot) if foot else None
    h = sum(head) / len(head) if head else None
    if f is not None and h is not None:
        return f < h - 0.1
    return (f is not None and f < 0.3) or (h is not None and h > 0.7)


def _group_rows(boxes: list[tuple[float, float, float, str]]) -> list[str]:
    """Merge OCR boxes that sit on the same visual row (receipts put name and price far apart)."""
    boxes.sort(key=lambda b: (b[0] + b[1]) / 2)
    rows: list[list[tuple[float, float, float, str]]] = []
    for b in boxes:
        cy, h = (b[0] + b[1]) / 2, b[1] - b[0]
        if rows:
            last = rows[-1]
            lcy = sum((x[0] + x[1]) / 2 for x in last) / len(last)
            lh = sum(x[1] - x[0] for x in last) / len(last)
            if abs(cy - lcy) < 0.5 * max(h, lh):
                last.append(b)
                continue
        rows.append([b])
    return [" ".join(t for *_, t in sorted(r, key=lambda x: x[2])) for r in rows]


def parse_lines(lines: list[str]) -> dict:
    out: dict = {"document_type": "receipt", "fields": [], "line_items": []}
    text = "\n".join(lines)
    if lines:
        out["vendor"] = lines[0].strip()
    if m := _DATE.search(text):
        out["date"] = m.group(1)
    for sym, code in _CURRENCY.items():
        if sym in text:
            out["currency"] = code
            break
    if re.search(r"\binvoice\b", text, re.I):
        out["document_type"] = "invoice"

    totals_started = False
    for raw in lines[1:]:
        line = raw.strip()
        pm = _PRICE_END.search(line)
        if not pm:
            continue
        amount = parse_number(pm.group("price"))
        label = line[: pm.start()].strip(" :.-")
        if _SUBTOTAL.search(label):
            out["subtotal"] = amount
            totals_started = True
            continue
        if _TAX.search(label):
            out["tax"] = (out.get("tax") or 0) + (amount or 0)
            totals_started = True
            continue
        if _TOTAL.search(label):
            # Keep the largest "total"-like line: receipts repeat totals (e.g. "total items 3").
            if amount is not None and (out.get("total") is None or amount >= out["total"]):
                out["total"] = amount
            totals_started = True
            continue
        if totals_started or _SKIP.search(label) or not re.search(r"[A-Za-z؀-ۿ]{2,}", label):
            if _SKIP.search(label) and label:
                out["fields"].append({"key": label[:40], "value": pm.group("price")})
            continue

        qty = unit_price = None
        # Table layout "qty | name | unit price | amount": a second trailing number is the unit price.
        if (m2 := _PRICE_END.search(label)) and re.search(r"[A-Za-z؀-ۿ]{2,}", label[: m2.start()]):
            unit_price = parse_number(m2.group("price"))
            label = label[: m2.start()].strip(" :.-@")
        if m := _QTY_AT.search(label):
            qty, unit_price = parse_number(m.group("q")), parse_number(m.group("p"))
            label = (label[: m.start()] + label[m.end() :]).strip()
        if qty is None and (m := _QTY_PREFIX.match(label)):
            q = parse_number(m.group("q"))
            if q is not None and 0 < q < 1000 and q == int(q):
                qty, label = q, m.group("rest").strip()
        label = label.strip(" :.-@")
        out["line_items"].append({"name": label, "quantity": qty, "unit_price": unit_price, "total": amount})
    return out
