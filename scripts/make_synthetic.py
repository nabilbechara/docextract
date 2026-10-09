"""Generate synthetic receipts/bills with exact ground truth: printed (thermal-style),
handwritten-style, and 'photographed' (rotated, blurred, noisy, uneven light).

Usage: python scripts/make_synthetic.py data/synthetic 30
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

FONTS_DIR = Path("C:/Windows/Fonts") if sys.platform == "win32" else Path("/usr/share/fonts")
PRINT_FONTS = ["consola.ttf", "cour.ttf", "arial.ttf", "lucon.ttf"]
HAND_FONTS = ["Inkfree.ttf", "segoepr.ttf", "segoesc.ttf", "comic.ttf"]

VENDORS = ["Cafe Beirut", "Green Market", "Hardware Hub", "Pharma Plus", "Tony's Pizza", "Book Corner", "Al Amal Bakery"]
PRODUCTS = [
    ("Espresso", 2.5), ("Cappuccino", 3.75), ("Croissant", 2.2), ("Water 500ml", 1.0), ("Chicken Wrap", 6.5),
    ("Screws M4 (box)", 4.99), ("Hammer", 12.0), ("Paint 1L", 18.5), ("Paracetamol 500mg", 3.4), ("Bandages", 5.25),
    ("Margherita Pizza", 11.0), ("Coke Can", 1.5), ("Notebook A5", 2.75), ("Pen Blue", 0.8), ("Manakish Zaatar", 1.25),
    ("Tomatoes 1kg", 2.1), ("Olive Oil 1L", 9.9), ("Rice 2kg", 4.6), ("Labneh", 3.3), ("Batteries AA x4", 6.0),
]


def font(name: str, size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(str(FONTS_DIR / name), size)
    except OSError:
        return ImageFont.load_default(size)


def make(rng: random.Random, style: str) -> tuple[Image.Image, dict]:
    vendor = rng.choice(VENDORS)
    n = rng.randint(2, 7)
    items = []
    for name, price in rng.sample(PRODUCTS, n):
        q = rng.choice([1, 1, 1, 2, 2, 3, 4, 5, 10])
        items.append({"name": name, "quantity": q, "unit_price": price, "total": round(q * price, 2)})
    subtotal = round(sum(i["total"] for i in items), 2)
    tax_rate = rng.choice([0, 0.11, 0.1])
    tax = round(subtotal * tax_rate, 2)
    total = round(subtotal + tax, 2)
    date = f"2026-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
    inv = f"{rng.randint(1000, 99999)}"

    hand = style == "handwritten"
    f = font(rng.choice(HAND_FONTS if hand else PRINT_FONTS), 30 if hand else 24)
    fb = font(rng.choice(HAND_FONTS) if hand else "arialbd.ttf", 38 if hand else 32)
    W = 760 if hand else 620
    lh = 46 if hand else 34
    H = 260 + lh * (n + 6)
    img = Image.new("RGB", (W, H), (250, 248, 240) if hand else "white")
    d = ImageDraw.Draw(img)
    ink = rng.choice([(20, 30, 120), (10, 10, 10), (40, 40, 40)]) if hand else (15, 15, 15)
    y = 30
    d.text((W // 2, y), vendor, font=fb, fill=ink, anchor="mt")
    y += 60
    d.text((30, y), f"Date: {date}", font=f, fill=ink)
    d.text((W - 30, y), f"No. {inv}", font=f, fill=ink, anchor="ra")
    y += lh + 10
    if hand:
        # Hand-written bills are usually a ruled table: Qty | Item | Price | Amount
        cols = [30, 110, W - 260, W - 130]
        for x, h in zip(cols, ["Qty", "Item", "Price", "Amount"]):
            d.text((x, y), h, font=f, fill=ink)
        y += lh
        d.line((20, y - 6, W - 20, y - 6), fill=ink, width=2)
        for it in items:
            jitter = rng.randint(-4, 4)
            d.text((cols[0] + jitter, y + jitter), str(it["quantity"]), font=f, fill=ink)
            d.text((cols[1], y), it["name"], font=f, fill=ink)
            d.text((cols[2], y + jitter), f"{it['unit_price']:.2f}", font=f, fill=ink)
            d.text((cols[3], y), f"{it['total']:.2f}", font=f, fill=ink)
            y += lh
            d.line((20, y - 6, W - 20, y - 6), fill=(160, 160, 200), width=1)
    else:
        d.line((20, y, W - 20, y), fill=ink, width=1)
        y += 10
        for it in items:
            if it["quantity"] == 1 and rng.random() < 0.5:
                left = it["name"]
            else:
                left = f"{it['quantity']} x {it['name']} @ {it['unit_price']:.2f}"
            d.text((30, y), left, font=f, fill=ink)
            d.text((W - 30, y), f"{it['total']:.2f}", font=f, fill=ink, anchor="ra")
            y += lh
        d.line((20, y, W - 20, y), fill=ink, width=1)
        y += 10
    for label, val in (("Subtotal", subtotal), ("Tax", tax), ("TOTAL", total)):
        if label == "Tax" and not tax:
            continue
        d.text((W // 2 - 40, y), label, font=fb if label == "TOTAL" else f, fill=ink)
        d.text((W - 30, y), f"${val:.2f}", font=fb if label == "TOTAL" else f, fill=ink, anchor="ra")
        y += lh + 4
    y += 10
    d.text((W // 2, y), "Thank you!", font=f, fill=ink, anchor="mt")
    img = img.crop((0, 0, W, min(H, y + 60)))

    if style == "photo":
        img = photograph(img, rng)

    gt = {
        "vendor": vendor,
        "date": date,
        "line_items": items,
        "subtotal": subtotal,
        "tax": tax,
        "total": total,
        "style": style,
    }
    return img, gt


def photograph(img: Image.Image, rng: random.Random) -> Image.Image:
    """Simulate a phone photo: background, rotation, blur, lighting gradient, noise, JPEG."""
    import numpy as np

    bg = Image.new("RGB", (img.width + 200, img.height + 200), tuple(rng.randint(60, 140) for _ in range(3)))
    bg.paste(img, (100, 100))
    bg = bg.rotate(rng.uniform(-8, 8), resample=Image.BICUBIC, expand=True, fillcolor=(90, 90, 90))
    bg = bg.filter(ImageFilter.GaussianBlur(rng.uniform(0.4, 1.3)))
    a = np.asarray(bg).astype(np.float32)
    grad = np.linspace(rng.uniform(0.65, 0.85), 1.05, a.shape[1])[None, :, None]
    a = a * grad + np.random.default_rng(rng.randint(0, 1 << 30)).normal(0, 7, a.shape)
    return Image.fromarray(np.clip(a, 0, 255).astype("uint8"))


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "data/synthetic")
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)
    styles = ["printed", "handwritten", "photo"]
    for i in range(n):
        style = styles[i % 3]
        img, gt = make(rng, style)
        img.save(out / f"syn_{i:03d}_{style}.jpg", quality=88)
        (out / f"syn_{i:03d}_{style}.json").write_text(json.dumps(gt, indent=2))
    print(f"wrote {n} documents to {out}")


if __name__ == "__main__":
    main()
