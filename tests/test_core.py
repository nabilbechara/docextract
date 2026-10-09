import base64
import io
import json
import random
import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from docextract.engines.ocr import OCREngine, parse_lines  # noqa: E402
from docextract.pipeline import Extractor  # noqa: E402
from docextract.preprocess import UnsupportedDocument, load_document  # noqa: E402
from docextract.schema import Extraction  # noqa: E402
from docextract.validate import check, coerce, parse_number  # noqa: E402
from make_synthetic import make  # noqa: E402


@pytest.mark.parametrize(
    "s,want",
    [
        ("1,234.50", 1234.5), ("1.234,50", 1234.5), ("12,5", 12.5), ("12,50", 12.5), ("1,234", 1234.0),
        ("$ 4", 4.0), ("4x", 4.0), ("(3.00)", -3.0), ("-2.5", -2.5), ("1 234,00", 1234.0), ("abc", None),
        (None, None), (7, 7.0), ("LBP 150.000", 150000.0), ("€9,99", 9.99), ("1.234.567,89", 1234567.89),
    ],
)
def test_parse_number(s, want):
    assert parse_number(s) == want


def test_coerce_never_throws_on_garbage():
    for raw in [None, [], "x", {"line_items": "nope"}, {"line_items": [None, 3, {"qty": "2", "price": "1,50"}]},
                {"fields": {"Invoice": "12", "empty": ""}, "total": "1.234,00"}]:
        ex = coerce(raw)
        assert isinstance(ex, Extraction)
    ex = coerce({"line_items": [{"description": "Tea", "qty": "2", "price": "1,50", "amount": "3,00"}], "fields": {"Inv": "9"}})
    assert ex.line_items[0].name == "Tea" and ex.line_items[0].quantity == 2 and ex.line_items[0].total == 3.0
    assert ex.fields[0].key == "Inv"


def test_check_repairs_and_flags():
    ex = coerce({"line_items": [
        {"name": "A", "quantity": 2, "unit_price": 1.5, "total": 3},
        {"name": "B", "quantity": None, "unit_price": 2, "total": 6},    # qty inferred = 3
        {"name": "C", "quantity": None, "unit_price": None, "total": 4},  # qty assumed 1
    ], "subtotal": 13, "tax": 1.3, "total": 14.3})
    ex, issues = check(ex, 1.0)
    assert not issues
    assert ex.line_items[1].quantity == 3 and ex.line_items[2].quantity == 1
    assert ex.confidence > 0.9

    bad = coerce({"line_items": [{"name": "A", "quantity": 2, "unit_price": 1.5, "total": 30}], "total": 3})
    bad, issues = check(bad, 1.0)
    assert issues and bad.confidence < 0.9


def test_names_are_cleaned_of_qty_and_price():
    ex = coerce({"line_items": [{"name": "3 x Water 500ml @ 1.00", "quantity": 3, "unit_price": 1, "total": 3},
                                {"name": "Batteries AA x4", "quantity": 1, "unit_price": 6, "total": 6}]})
    assert [i.name for i in ex.line_items] == ["Water 500ml", "Batteries AA x4"]


def test_pack_size_misread_as_quantity_is_flagged():
    ex = coerce({"line_items": [{"name": "Batteries AA x4", "quantity": 4, "unit_price": 6, "total": 6}], "total": 6})
    _, issues = check(ex, 1.0)
    assert any("part of the product name" in i for i in issues)


def test_zero_decimal_vs_three_decimal_currencies():
    from docextract.engines.ocr import _zero_decimal_doc

    assert _zero_decimal_doc("TEA 20.000\nSUB-TOTAL 40.000\nCASH 100.000")          # IDR thousands
    assert not _zero_decimal_doc("Lunch 30.060\nTax 2.525\nPayment 34.088")          # Jordanian dinar fils
    assert not _zero_decimal_doc("Coffee 3.50\nTea 2.25\nTOTAL 5.75")


def test_reconcile_uses_ocr_value_only_when_it_closes_the_sum():
    from docextract.pipeline import reconcile

    ex = coerce({"line_items": [{"name": "KUKHURA KO MASHU", "quantity": 1, "total": 695}], "subtotal": 595, "total": 739.59})
    assert reconcile(ex, {"line_items": [{"name": "KURHURA KO MASHU", "total": 595}]})
    assert ex.line_items[0].total == 595 and ex.warnings
    ex = coerce({"line_items": [{"name": "Tea", "total": 5}], "total": 9})
    assert not reconcile(ex, {"line_items": [{"name": "Tea", "total": 7}]})  # 7 doesn't fix it either
    assert ex.line_items[0].total == 5


def test_readable_format():
    ex = coerce({"vendor": "X", "line_items": [{"name": "Coffee", "quantity": 4, "unit_price": 2, "total": 8}], "total": 8})
    ex, _ = check(ex, 1.0)
    txt = ex.readable()
    assert "item quantity 4 | item name Coffee" in txt and "total: 8" in txt


def _png(img: Image.Image, fmt="PNG") -> bytes:
    b = io.BytesIO()
    img.save(b, format=fmt)
    return b.getvalue()


def test_preprocess_formats():
    rgba = Image.new("RGBA", (300, 200), (0, 0, 0, 0))
    pages = load_document(_png(rgba))
    assert pages[0].image.mode == "RGB" and max(pages[0].image.size) >= 800  # upscaled tiny image
    huge = Image.new("RGB", (6000, 8000), "white")
    assert load_document(_png(huge, "JPEG"))[0].image.size[0] * load_document(_png(huge, "JPEG"))[0].image.size[1] <= 2_400_000
    with pytest.raises(UnsupportedDocument):
        load_document(b"definitely not an image")
    with pytest.raises(UnsupportedDocument):
        load_document(b"")


def test_preprocess_pdf_with_text_layer():
    img, gt = make(random.Random(1), "printed")
    pdf = io.BytesIO()
    img.save(pdf, format="PDF")
    pages = load_document(pdf.getvalue(), "x.pdf")
    assert len(pages) == 1 and pages[0].image.size[0] > 100


def test_exif_rotation_is_applied():
    img = Image.new("RGB", (400, 200), "white")
    exif = img.getexif()
    exif[0x0112] = 6  # rotate 90 CW on display
    b = io.BytesIO()
    img.save(b, format="JPEG", exif=exif)
    w, h = load_document(b.getvalue())[0].image.size
    assert h > w


def test_parse_lines_table_and_receipt_layouts():
    out = parse_lines(["Shop", "3 x Water 500ml @ 1.00 3.00", "10 Cappuccino 3.75 37.50", "Bread 2.20", "Subtotal 42.70", "Tax 0.00", "TOTAL $42.70", "Cash 50.00"])
    items = [(i["quantity"], i["name"], i["unit_price"], i["total"]) for i in out["line_items"]]
    assert items == [(3.0, "Water 500ml", 1.0, 3.0), (10.0, "Cappuccino", 3.75, 37.5), (None, "Bread", None, 2.2)]
    assert out["total"] == 42.7 and out["subtotal"] == 42.7


@pytest.fixture(scope="module")
def ocr():
    e = OCREngine()
    if not e.available():
        pytest.skip("rapidocr not installed")
    return e


@pytest.mark.parametrize("style", ["printed", "handwritten", "photo"])
@pytest.mark.parametrize("deg", [0, 90, 180, 270])
def test_orientation_fix(ocr, deg, style):
    img, _ = make(random.Random(7), style)
    fixed, rotated = ocr.orient(img.rotate(deg, expand=True))
    if deg == 180:
        # 180 detection is deliberately conservative: a wrong flip ruins a good page, a missed
        # one still leaves a readable page. So: either fixed, or left alone - never sideways.
        assert rotated in (0, 180), f"rotated {rotated} for input rotated {deg}"
    else:
        assert (rotated + deg) % 360 == 0, f"rotated {rotated} for input rotated {deg}"


class DeadVLM:
    model = "dead"
    base_confidence = 0.95

    def available(self):
        return True

    def describe(self):
        return []

    def extract(self, *a, **k):
        raise ConnectionError("gpu box is down")


class FakeVLM(DeadVLM):
    """Returns a wrong reading first, the right one when given feedback - exercises self-correction."""

    def __init__(self, gt):
        self.gt, self.calls = gt, []

    def extract(self, img, feedback=None, ocr_hint=None, hot=False):
        self.calls.append(feedback)
        items = [dict(i) for i in self.gt["line_items"]]
        if feedback is None:
            items[0]["total"] = items[0]["total"] * 10 + 1  # misread digit
        return {"document_type": "receipt", "line_items": items, "subtotal": self.gt["subtotal"], "tax": self.gt["tax"], "total": self.gt["total"], "fields": []}


def test_pipeline_falls_back_to_ocr_when_vlm_dies(ocr):
    img, gt = make(random.Random(3), "printed")
    ex = Extractor(vlm=DeadVLM(), ocr=ocr).extract_pages([img])
    assert ex.engine == "ocr-rules" and ex.line_items
    assert any("fallback" in w for w in ex.warnings)
    assert ex.total == pytest.approx(gt["total"])


def test_pipeline_self_corrects(ocr):
    img, gt = make(random.Random(4), "printed")
    vlm = FakeVLM(gt)
    ex = Extractor(vlm=vlm, ocr=ocr).extract_pages([img])
    # Fixed either by re-asking the model with the failed checks, or straight away by reconciling
    # with the OCR reading of the same row.
    reconciled = any("corrected to" in w for w in ex.warnings)
    assert reconciled or (len(vlm.calls) == 2 and vlm.calls[1])
    assert ex.line_items[0].total == pytest.approx(gt["line_items"][0]["total"])
    assert not [w for w in ex.warnings if "x" in w and "=" in w]


def test_api(monkeypatch, ocr):
    monkeypatch.setenv("DOCEXTRACT_API_KEYS", "secret")
    import importlib

    import docextract.api as api

    api = importlib.reload(api)
    api.extractor = Extractor(vlm=DeadVLM(), ocr=ocr)
    from fastapi.testclient import TestClient

    c = TestClient(api.app)
    img, gt = make(random.Random(5), "printed")
    data = _png(img, "JPEG")
    assert c.post("/v1/extract", files={"file": ("r.jpg", data)}).status_code == 401
    h = {"Authorization": "Bearer secret"}
    r = c.post("/v1/extract", files={"file": ("r.jpg", data)}, headers=h)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["total"] == pytest.approx(gt["total"]) and body["line_items"]
    r = c.post("/v1/extract/base64?format=text", json={"data": base64.b64encode(data).decode()}, headers={"X-API-Key": "secret"})
    assert r.status_code == 200 and "item name" in r.text
    assert c.post("/v1/extract", files={"file": ("x.bin", b"garbage")}, headers=h).status_code == 415
    assert c.post("/v1/extract/base64", json={"data": "%%%"}, headers=h).status_code == 400
    assert c.get("/health").json()["status"] == "ok"
