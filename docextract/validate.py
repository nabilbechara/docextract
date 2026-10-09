"""Coercion + arithmetic cross-checks.

Models (and OCR) make mistakes; documents almost always contain redundant arithmetic
(qty x price = line total, sum of lines = subtotal, subtotal + tax = total). We use
that redundancy to detect errors, repair obvious gaps and score confidence.
"""

from __future__ import annotations

import re
from typing import Any

from .schema import Extraction, KeyValue, LineItem

_NUM_RE = re.compile(r"-?\d[\d\s.,']*")
_ZERO_DEC = re.compile(r"LBP|L\.?L\.?|ل\.ل|IDR|Rp|VND|₫|IQD|SYP|IRR|KRW|₩|JPY|¥|CLP|COP|UGX|PYG", re.I)


def parse_number(v: Any) -> float | None:
    """Parse '1,234.50', '1.234,50', '12,5', '$ 4', '4x', '(3.00)' -> float."""
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if not s:
        return None
    neg = s.startswith("(") and s.endswith(")") or s.startswith("-") or s.endswith("-")
    m = _NUM_RE.search(s.replace("O", "0") if re.fullmatch(r"[\dO.,\s]+", s) else s)
    if not m:
        return None
    t = re.sub(r"[\s']", "", m.group(0)).lstrip("-")
    if "," in t and "." in t:
        # Whichever separator comes last is the decimal one.
        if t.rfind(",") > t.rfind("."):
            t = t.replace(".", "").replace(",", ".")
        else:
            t = t.replace(",", "")
    elif "," in t:
        head, _, tail = t.rpartition(",")
        # '1,234' (thousands) vs '12,5' / '12,50' (decimal comma)
        t = t.replace(",", "") if len(tail) == 3 and head.replace(",", "").isdigit() else t.replace(",", ".")
    elif t.count(".") > 1:
        head, _, tail = t.rpartition(".")
        t = head.replace(".", "") + "." + tail if len(tail) != 3 else t.replace(".", "")
    elif "." in t and len(t.rpartition(".")[2]) == 3 and _ZERO_DEC.search(s):
        # 'LBP 150.000' is a hundred and fifty thousand, not 150 (no-cents currencies).
        t = t.replace(".", "")
    t = t.rstrip(".")
    try:
        f = float(t)
    except ValueError:
        return None
    return -f if neg else f


def coerce(raw: dict) -> Extraction:
    """Build an Extraction from loosely-shaped model JSON without ever throwing."""
    raw = raw if isinstance(raw, dict) else {}

    def kvs(x: Any) -> list[KeyValue]:
        out = []
        if isinstance(x, dict):
            x = [{"key": k, "value": v} for k, v in x.items()]
        for e in x or []:
            if isinstance(e, dict) and e.get("key") not in (None, ""):
                val = e.get("value")
                if val in (None, ""):
                    continue
                out.append(KeyValue(key=str(e["key"]).strip(), value=str(val).strip()))
        return out

    def opt_str(x: Any) -> str | None:
        if x is None:
            return None
        s = str(x).strip()
        return s or None

    items = []
    for it in raw.get("line_items") or raw.get("items") or []:
        if not isinstance(it, dict):
            continue
        name = opt_str(it.get("name") or it.get("description") or it.get("item")) or ""
        li = LineItem(
            name=name,
            quantity=parse_number(it.get("quantity", it.get("qty"))),
            unit=opt_str(it.get("unit")),
            unit_price=parse_number(it.get("unit_price", it.get("price"))),
            total=parse_number(it.get("total", it.get("amount"))),
            extra=kvs(it.get("extra")),
        )
        if li.name or li.total is not None or li.unit_price is not None:
            items.append(li)

    return Extraction(
        document_type=(opt_str(raw.get("document_type")) or "unknown").lower().replace(" ", "_"),
        vendor=opt_str(raw.get("vendor")),
        date=opt_str(raw.get("date")),
        currency=opt_str(raw.get("currency")),
        fields=kvs(raw.get("fields")),
        line_items=items,
        subtotal=parse_number(raw.get("subtotal")),
        tax=parse_number(raw.get("tax")),
        total=parse_number(raw.get("total")),
    )


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= max(0.02, 0.01 * max(abs(a), abs(b)))


def check(ex: Extraction, base_confidence: float) -> tuple[Extraction, list[str]]:
    """Repair gaps, run arithmetic checks. Returns (extraction, hard_issues).

    hard_issues are arithmetic contradictions worth re-asking the model about;
    soft notes go straight into ex.warnings.
    """
    issues: list[str] = []
    passed = run = 0
    assumed_qty = 0

    for i, it in enumerate(ex.line_items, 1):
        q, p, t = it.quantity, it.unit_price, it.total
        if q is not None and p is not None and t is not None:
            run += 1
            if _close(q * p, t):
                passed += 1
            elif _close(p, t) and q != 1:
                # Common: the "price" column is already the line total.
                passed += 1
            else:
                issues.append(f"line {i} '{it.name}': {q} x {p} = {q * p:.2f} but line total is {t}")
        elif q is not None and t is not None and p is None and q:
            it.unit_price = round(t / q, 4)
        elif p is not None and t is not None and q is None and p:
            r = t / p
            if abs(r - round(r)) < 0.01 and round(r) > 0:
                it.quantity = float(round(r))
        elif q is None and (t is not None or p is not None):
            it.quantity = 1.0
            assumed_qty += 1
            if t is None:
                it.total = p
            elif p is None:
                it.unit_price = t

    line_totals = [it.total for it in ex.line_items if it.total is not None]
    if line_totals and len(line_totals) == len(ex.line_items):
        s = sum(line_totals)
        target = ex.subtotal
        label = "subtotal"
        if target is None and ex.total is not None:
            target, label = (ex.total - (ex.tax or 0.0)), "total minus tax"
        if target is not None:
            run += 1
            if _close(s, target) or (ex.total is not None and _close(s, ex.total)):
                passed += 1
            else:
                issues.append(f"line totals sum to {s:.2f} but {label} is {target:.2f} (missing or misread items?)")

    if ex.subtotal is not None and ex.tax is not None and ex.total is not None:
        run += 1
        if _close(ex.subtotal + ex.tax, ex.total):
            passed += 1
        else:
            ex.warnings.append(
                f"subtotal {ex.subtotal} + tax {ex.tax} != total {ex.total} (could be discounts/fees/tips)"
            )

    if assumed_qty:
        ex.warnings.append(f"quantity not printed for {assumed_qty} item(s); assumed 1")
    if not ex.line_items and not ex.fields and ex.total is None:
        issues.append("nothing was extracted")

    conf = base_confidence
    if run:
        conf *= 0.6 + 0.4 * (passed / run)
    else:
        conf *= 0.85  # nothing to cross-check against
    if not ex.line_items:
        conf *= 0.9
    ex.confidence = round(max(0.0, min(1.0, conf)), 3)
    return ex, issues
