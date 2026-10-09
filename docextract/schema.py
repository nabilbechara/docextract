"""Output schema shared by every engine, the API and the CLI."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class KeyValue(BaseModel):
    key: str
    value: str


class LineItem(BaseModel):
    name: str = ""
    quantity: float | None = None
    unit: str | None = None
    unit_price: float | None = None
    total: float | None = None
    # Any other column the document has for this row (SKU, discount, VAT %, date, ...).
    extra: list[KeyValue] = Field(default_factory=list)


class Extraction(BaseModel):
    document_type: str = "unknown"  # receipt, invoice, bill, delivery_note, form, ...
    vendor: str | None = None
    date: str | None = None
    currency: str | None = None
    # Header / footer fields that are not line items (invoice no, customer, address, ...).
    fields: list[KeyValue] = Field(default_factory=list)
    line_items: list[LineItem] = Field(default_factory=list)
    subtotal: float | None = None
    tax: float | None = None
    total: float | None = None

    # Filled in by the pipeline, not the model.
    engine: str = ""
    pages: int = 1
    confidence: float = 0.0
    warnings: list[str] = Field(default_factory=list)
    raw_text: str | None = None

    def readable(self) -> str:
        """Human-friendly rendering, e.g. 'item quantity 4 item name Coffee ...'."""
        out: list[str] = [f"document type: {self.document_type}"]
        for label, val in (("vendor", self.vendor), ("date", self.date), ("currency", self.currency)):
            if val:
                out.append(f"{label}: {val}")
        for kv in self.fields:
            out.append(f"{kv.key}: {kv.value}")
        for i, it in enumerate(self.line_items, 1):
            parts: list[str] = []
            if it.quantity is not None:
                parts.append(f"item quantity {_fmt(it.quantity)}")
            parts.append(f"item name {it.name}")
            if it.unit:
                parts.append(f"unit {it.unit}")
            if it.unit_price is not None:
                parts.append(f"unit price {_fmt(it.unit_price)}")
            if it.total is not None:
                parts.append(f"item total {_fmt(it.total)}")
            for kv in it.extra:
                parts.append(f"{kv.key} {kv.value}")
            out.append(f"item {i}: " + " | ".join(parts))
        for label, val in (("subtotal", self.subtotal), ("tax", self.tax), ("total", self.total)):
            if val is not None:
                out.append(f"{label}: {_fmt(val)}")
        out.append(f"confidence: {self.confidence:.2f} (engine: {self.engine})")
        for w in self.warnings:
            out.append(f"warning: {w}")
        return "\n".join(out)


def _fmt(x: Any) -> str:
    if isinstance(x, float) and x.is_integer():
        return str(int(x))
    return str(x)


def llm_json_schema() -> dict:
    """JSON schema handed to the VLM for constrained decoding (model-facing fields only)."""
    kv = {
        "type": "object",
        "properties": {"key": {"type": "string"}, "value": {"type": "string"}},
        "required": ["key", "value"],
    }
    num = {"type": ["number", "null"]}
    s = {"type": ["string", "null"]}
    return {
        "type": "object",
        "properties": {
            "document_type": {"type": "string"},
            "vendor": s,
            "date": s,
            "currency": s,
            "fields": {"type": "array", "items": kv},
            "line_items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "quantity": num,
                        "unit": s,
                        "unit_price": num,
                        "total": num,
                        "extra": {"type": "array", "items": kv},
                    },
                    "required": ["name", "quantity", "unit_price", "total"],
                },
            },
            "subtotal": num,
            "tax": num,
            "total": num,
        },
        "required": ["document_type", "fields", "line_items", "total"],
    }
