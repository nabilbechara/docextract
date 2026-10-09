"""Turn arbitrary uploaded bytes (photo, scan, PDF, ...) into clean RGB page images."""

from __future__ import annotations

import io
from dataclasses import dataclass

from PIL import Image, ImageOps

Image.MAX_IMAGE_PIXELS = 200_000_000  # big scans are legit; still bounded against bombs

MAX_PAGES = 30
# Pixel budget instead of a fixed long side, so a 1:5 till roll keeps legible text while a
# phone photo is shrunk to something an 8 GB GPU encodes quickly.
MAX_PIXELS = 2_400_000
MAX_SIDE = 3600
MIN_SIDE = 1280  # small scans get upscaled: the vision model reads tiny text poorly


class UnsupportedDocument(ValueError):
    pass


@dataclass
class Page:
    image: Image.Image
    text: str | None = None  # embedded text layer (digital PDFs) - exact, better than OCR


def load_pages(data: bytes, filename: str = "") -> list[Image.Image]:
    return [p.image for p in load_document(data, filename)]


def load_document(data: bytes, filename: str = "") -> list[Page]:
    if not data:
        raise UnsupportedDocument("empty file")
    if data[:5] == b"%PDF-" or filename.lower().endswith(".pdf"):
        pages = _pdf_pages(data)
    else:
        pages = [Page(im) for im in _image_pages(data)]
    if not pages:
        raise UnsupportedDocument("no pages found")
    for p in pages:
        p.image = normalize(p.image)
    return pages


def _pdf_pages(data: bytes) -> list[Page]:
    import pypdfium2 as pdfium

    try:
        pdf = pdfium.PdfDocument(data)
    except Exception as e:  # noqa: BLE001 - pdfium raises its own error types
        raise UnsupportedDocument(f"unreadable PDF: {e}") from e
    out = []
    for i in range(min(len(pdf), MAX_PAGES)):
        page = pdf[i]
        w, h = page.get_size()  # points (1/72 in)
        scale = (MAX_PIXELS / max(1.0, w * h)) ** 0.5
        img = page.render(scale=max(1.0, min(scale, 4.0))).to_pil()
        try:
            text = page.get_textpage().get_text_range().strip() or None
        except Exception:  # noqa: BLE001
            text = None
        out.append(Page(img, text))
    return out


def _image_pages(data: bytes) -> list[Image.Image]:
    _register_heif()
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception as e:  # noqa: BLE001
        raise UnsupportedDocument(f"not a supported image or PDF: {e}") from e
    frames = []
    # Multi-frame TIFF scans are common for faxed bills; GIF/APNG: first frame only.
    n = getattr(img, "n_frames", 1) if img.format in ("TIFF", "MPO") else 1
    for i in range(min(n, MAX_PAGES)):
        img.seek(i)
        frames.append(img.copy())
    return frames


def normalize(img: Image.Image) -> Image.Image:
    try:
        img = ImageOps.exif_transpose(img)  # phone photos are often stored sideways
    except Exception:  # noqa: BLE001 - corrupt EXIF must not kill the request
        pass
    if img.mode in ("I;16", "I;16B", "I", "F"):
        img = ImageOps.autocontrast(img.convert("I").point(lambda v: v / 256).convert("L"))
    if img.mode in ("RGBA", "LA", "P", "PA"):
        img = img.convert("RGBA")
        bg = Image.new("RGB", img.size, "white")
        bg.paste(img, mask=img.split()[-1])
        img = bg
    else:
        img = img.convert("RGB")
    w, h = img.size
    r = min(1.0, (MAX_PIXELS / (w * h)) ** 0.5, MAX_SIDE / max(w, h))
    if r < 1.0:
        img = img.resize((max(1, int(w * r)), max(1, int(h * r))), Image.LANCZOS)
    elif max(w, h) < MIN_SIDE:
        r = MIN_SIDE / max(w, h)
        img = img.resize((int(w * r), int(h * r)), Image.BICUBIC)
    return img


def enhance(img: Image.Image) -> Image.Image:
    """Contrast-boosted, sharpened variant used on retry for faded thermal paper / dim photos."""
    from PIL import ImageFilter

    g = ImageOps.grayscale(img)
    g = ImageOps.autocontrast(g, cutoff=1)
    g = g.filter(ImageFilter.UnsharpMask(radius=2, percent=120, threshold=3))
    return g.convert("RGB")


def to_jpeg_bytes(img: Image.Image, quality: int = 90) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


_heif_done = False


def _register_heif() -> None:
    """iPhone photos are HEIC; support them when pillow-heif is installed."""
    global _heif_done
    if _heif_done:
        return
    _heif_done = True
    try:
        from pillow_heif import register_heif_opener

        register_heif_opener()
    except ImportError:
        pass
