"""Orchestration: preprocess -> orient -> OCR shadow read -> VLM with self-correction
-> cross-check against OCR -> validate -> merge pages.

Every step degrades gracefully: if orientation, OCR or the VLM fail, the next layer still
produces a result, and every doubt is surfaced in `warnings` + `confidence`.
"""

from __future__ import annotations

import logging
import os
import re
import time

from .engines import OCREngine, VLMEngine
from .engines.ocr import _zero_decimal_doc, parse_lines
from .preprocess import Page, enhance, load_document
from .schema import Extraction
from .validate import check, coerce

log = logging.getLogger(__name__)

USE_OCR_HINT = os.environ.get("DOCEXTRACT_OCR_HINT", "1") == "1"
MAX_ATTEMPTS = max(1, int(os.environ.get("DOCEXTRACT_MAX_ATTEMPTS", "3")))


class Extractor:
    def __init__(self, vlm: VLMEngine | None = None, ocr: OCREngine | None = None):
        self.vlm = vlm or VLMEngine()
        self.ocr = ocr or OCREngine()

    def health(self) -> dict:
        return {
            "vlm": self.vlm.available(),
            "vlm_model": self.vlm.model,
            "vlm_backends": self.vlm.describe(),
            "ocr": self.ocr.available(),
        }

    def extract_bytes(self, data: bytes, filename: str = "", engine: str = "auto") -> Extraction:
        return self.extract_pages(load_document(data, filename), engine)

    def extract_pages(self, pages: list, engine: str = "auto") -> Extraction:
        pages = [p if isinstance(p, Page) else Page(p) for p in pages]
        t0 = time.monotonic()
        use_vlm = engine in ("auto", "vlm") and self.vlm.available()
        if engine == "vlm" and not use_vlm:
            raise RuntimeError(f"VLM engine requested but no server with model {self.vlm.model} is reachable")
        results = [self._page(p, use_vlm) for p in pages]
        results = [r for r in results if not _blank(r)] or results[:1]
        ex = results[0] if len(results) == 1 else merge(results)
        ex.pages = len(pages)
        log.info("extracted %d page(s) in %.1fs engine=%s conf=%.2f", len(pages), time.monotonic() - t0, ex.engine, ex.confidence)
        return ex

    # ------------------------------------------------------------------------------------
    def _page(self, page: Page, use_vlm: bool) -> Extraction:
        img = page.image
        notes: list[str] = []

        try:
            img, deg = self.ocr.orient(img)
            if deg:
                notes.append(f"page was rotated {deg} degrees; auto-corrected")
        except Exception as e:  # noqa: BLE001
            log.warning("orientation failed: %s", e)

        ocr_lines: list[str] = []
        try:
            ocr_lines = self.ocr.lines(img)
        except Exception as e:  # noqa: BLE001
            log.warning("ocr failed: %s", e)
        text = page.text or "\n".join(ocr_lines)
        shadow_raw = parse_lines(page.text.splitlines() if page.text else ocr_lines)

        if use_vlm:
            ex = self._vlm_page(img, text if USE_OCR_HINT else None, shadow_raw, exact_text=bool(page.text))
            if ex is not None:
                ex.warnings = notes + ex.warnings
                ex.raw_text = text
                return ex
            notes.append("vision model failed; used OCR fallback (lower accuracy, weak on handwriting)")

        ex, issues = check(coerce(shadow_raw), 0.9 if page.text else self.ocr.base_confidence)
        ex.engine = "pdf-text-rules" if page.text else "ocr-rules"
        ex.warnings = notes + issues + ex.warnings
        ex.raw_text = text
        return ex

    def _vlm_page(self, img, hint: str | None, shadow_raw: dict, exact_text: bool) -> Extraction | None:
        best: tuple[Extraction, list[str]] | None = None
        feedback = None
        failed_last = False
        thousands = bool(hint) and _zero_decimal_doc(hint)
        if thousands:
            note = (
                "This document writes amounts with '.' or ',' as THOUSANDS separators and no cents "
                "(e.g. 25.000 or 25,000 means twenty-five thousand): output 25000, not 25."
            )
            hint = f"{note}\n{hint}"
        for attempt in range(MAX_ATTEMPTS):
            # 1: as-is. 2: same image + the failed checks. 3: contrast-enhanced image, slightly
            # warmer sampling, to shake loose a different reading of ambiguous digits. After a
            # failed call (e.g. a repetition loop) always retry warmer: greedy decoding would
            # just loop again.
            last = attempt == 2
            page_img = enhance(img) if last else img
            try:
                raw = self.vlm.extract(page_img, feedback=feedback, ocr_hint=hint, hot=last or failed_last)
                failed_last = False
            except Exception as e:  # noqa: BLE001 - network, timeout, bad JSON, truncation
                log.warning("vlm attempt %d failed: %s", attempt + 1, e)
                failed_last = True
                continue
            ex = coerce(raw)
            if thousands:
                _rescale_thousands(ex)
            ex, issues = check(ex, self.vlm.base_confidence)
            if issues and reconcile(ex, shadow_raw):
                kept = ex.warnings
                ex, issues = check(ex, self.vlm.base_confidence * 0.95)
                ex.warnings = kept + [w for w in ex.warnings if w not in kept]
            issues += cross_check(ex, shadow_raw, hint, exact_text)
            ex.confidence = round(ex.confidence * (0.85 if issues else 1.0), 3)
            if best is None or _better((ex, issues), best):
                best = (ex, issues)
            if not best[1]:
                break
            feedback = "\n".join(best[1])
        if best is None:
            return None
        ex, issues = best
        ex.engine = f"vlm:{self.vlm.model}"
        ex.warnings = issues + ex.warnings
        return ex


_AMT = re.compile(r"\d+(?:[.,]\d{1,2})?")


def cross_check(ex: Extraction, shadow_raw: dict, text: str | None, exact_text: bool) -> list[str]:
    """Compare the VLM reading with the independent OCR/text-layer reading."""
    issues: list[str] = []
    shadow = coerce(shadow_raw)
    # 1. Rows the OCR saw (with an amount) that the VLM has no row for -> possibly missed rows.
    vlm_amounts = {round(v, 2) for it in ex.line_items for v in (it.total, it.unit_price) if v is not None}
    missed = [it for it in shadow.line_items if it.total is not None and round(it.total, 2) not in vlm_amounts and it.total > 0]
    vlm_sum_ok = ex.total is None or not ex.line_items or any(
        abs(sum(i.total or 0 for i in ex.line_items) - t) <= 0.02 * max(1, abs(t)) for t in (ex.total, ex.subtotal or -1)
    )
    if missed and not vlm_sum_ok:
        issues.append(
            "these rows were found by OCR but are missing from the reading: "
            + "; ".join(f"'{m.name}' {m.total}" for m in missed[:8])
        )
    # 2. Total disagreement. With exact PDF text any mismatch is suspect; with OCR only flag the
    # classic slip of reporting the subtotal when the printed TOTAL is larger.
    if shadow.total is not None and ex.total is not None and abs(shadow.total - ex.total) > 0.011:
        if exact_text or (shadow.total > ex.total and ex.subtotal is not None and abs(ex.total - ex.subtotal) < 0.011):
            issues.append(
                f"the document's TOTAL line appears to read {shadow.total} but the reading says {ex.total}; "
                "total must be the final amount to pay including tax, service and fees"
            )
    # 3. With an exact PDF text layer every printed amount must appear in it.
    if exact_text and text:
        present = {_n(m) for m in _AMT.findall(text.replace(",", "."))}
        unseen = [it.total for it in ex.line_items if it.total is not None and round(it.total, 2) not in present]
        if unseen:
            issues.append(f"line totals not present in the document text: {unseen[:6]}")
    return issues


def reconcile(ex: Extraction, shadow_raw: dict) -> bool:
    """If the rows don't add up to the subtotal/total, try the OCR's value for the same row
    (matched by name). Apply it only when that makes the arithmetic close exactly."""
    import difflib

    totals = [it.total for it in ex.line_items]
    if not ex.line_items or any(t is None for t in totals):
        return False
    targets = [t for t in (ex.subtotal, ex.total, (ex.total - ex.tax) if ex.total is not None and ex.tax else None) if t is not None]
    s = sum(totals)
    if not targets or any(abs(s - t) <= 0.011 for t in targets):
        return False
    shadow = coerce(shadow_raw).line_items
    for i, it in enumerate(ex.line_items):
        for sh in shadow:
            if sh.total is None or abs(sh.total - it.total) < 0.011:
                continue
            if difflib.SequenceMatcher(None, it.name.lower(), sh.name.lower()).ratio() < 0.7:
                continue
            new_sum = s - it.total + sh.total
            if any(abs(new_sum - t) <= 0.011 for t in targets):
                ex.warnings.append(f"row {i + 1} '{it.name}': amount {it.total} corrected to {sh.total} (OCR reading; makes the totals add up)")
                if it.unit_price is not None and it.quantity:
                    it.unit_price = round(sh.total / it.quantity, 4)
                it.total = sh.total
                return True
    return False


def _rescale_thousands(ex: Extraction) -> None:
    """Model read '25.000' as 25.0 on a no-cents document: every amount is 1000x too small."""
    amounts = [v for it in ex.line_items for v in (it.unit_price, it.total) if v] + [
        v for v in (ex.subtotal, ex.tax, ex.total) if v
    ]
    if not amounts or max(amounts) >= 1000:
        return
    # On these documents real amounts are whole thousands-ish; tiny values with <=3 decimals
    # are the misread form.
    if all(round(v * 1000) == round(v * 1000, 6) for v in amounts):
        for it in ex.line_items:
            it.unit_price = round(it.unit_price * 1000) if it.unit_price is not None else None
            it.total = round(it.total * 1000) if it.total is not None else None
        for attr in ("subtotal", "tax", "total"):
            v = getattr(ex, attr)
            setattr(ex, attr, round(v * 1000) if v is not None else None)
        ex.warnings.append("amounts used '.' as a thousands separator; rescaled")


def _n(s: str) -> float:
    try:
        return round(float(s), 2)
    except ValueError:
        return -1.0


def _blank(ex: Extraction) -> bool:
    return not ex.line_items and not ex.fields and ex.total is None and not (ex.raw_text or "").strip()


def _better(a: tuple[Extraction, list[str]], b: tuple[Extraction, list[str]]) -> bool:
    return (len(a[1]), -a[0].confidence, -len(a[0].line_items)) < (len(b[1]), -b[0].confidence, -len(b[0].line_items))


def merge(pages: list[Extraction]) -> Extraction:
    """Multi-page: header from the first page that has it, items concatenated, totals from the last."""
    out = Extraction()
    for p in pages:
        for attr in ("vendor", "date", "currency"):
            if getattr(out, attr) is None and getattr(p, attr):
                setattr(out, attr, getattr(p, attr))
        if out.document_type == "unknown":
            out.document_type = p.document_type
        seen = {(f.key, f.value) for f in out.fields}
        out.fields += [f for f in p.fields if (f.key, f.value) not in seen]
        out.line_items += p.line_items
        for attr in ("subtotal", "tax", "total"):
            if getattr(p, attr) is not None:
                setattr(out, attr, getattr(p, attr))
    out.engine = ",".join(sorted({p.engine for p in pages}))
    base = min(p.confidence for p in pages)
    out, issues = check(out, 1.0)
    out.confidence = round(min(base, out.confidence), 3)
    out.warnings = issues + [f"page {i}: {w}" for i, p in enumerate(pages, 1) for w in p.warnings]
    out.raw_text = "\n\n".join(p.raw_text or "" for p in pages)
    return out
