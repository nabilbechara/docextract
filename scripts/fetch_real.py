"""Download real-world receipt photos with human-labelled ground truth from public datasets.

- CORD v2 (naver-clova-ix/cord-v2): ~1000 Indonesian receipt photos, item-level labels.

Usage: python scripts/fetch_real.py data/real/cord 40
"""

from __future__ import annotations

import json
import re
import sys
import time
import urllib.request
from pathlib import Path

API = "https://datasets-server.huggingface.co/rows?dataset={ds}&config=default&split={split}&offset={off}&length={n}"


def idr(s) -> float | None:
    """CORD amounts are Indonesian rupiah: '60.000' / '60,000' / '60000' all mean 60000."""
    if s is None:
        return None
    if isinstance(s, list):
        s = s[0] if s else None
        if s is None:
            return None
    neg = "-" in str(s)
    digits = re.sub(r"[^\d]", "", str(s))
    return (-1 if neg else 1) * float(digits) if digits else None


def qty(s) -> float | None:
    if s is None:
        return None
    if isinstance(s, list):
        s = s[0] if s else None
    m = re.search(r"\d+(?:[.,]\d+)?", str(s or ""))
    return float(m.group(0).replace(",", ".")) if m else None


def cord_gt(row: dict) -> dict:
    gp = json.loads(row["ground_truth"])["gt_parse"]
    menu = gp.get("menu", [])
    menu = [menu] if isinstance(menu, dict) else menu
    items = []
    for m in menu:
        name = m.get("nm")
        if isinstance(name, list):
            name = " ".join(map(str, name))
        if not name:
            continue
        items.append({"name": str(name), "quantity": qty(m.get("cnt")), "unit_price": idr(m.get("unitprice")), "total": idr(m.get("price"))})
    tot = gp.get("total", {})
    tot = tot[0] if isinstance(tot, list) else tot
    sub = gp.get("sub_total", {})
    sub = sub[0] if isinstance(sub, list) else sub
    return {
        "line_items": items,
        "total": idr(tot.get("total_price")) if isinstance(tot, dict) else None,
        "subtotal": idr(sub.get("subtotal_price")) if isinstance(sub, dict) else None,
        "style": "real_cord",
    }


def get(url: str, tries: int = 6) -> bytes:
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            print(f"retry {i + 1}: {e}", flush=True)
            time.sleep(3 * (i + 1))
    raise RuntimeError(f"failed: {url[:80]}")


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "data/real/cord")
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    out.mkdir(parents=True, exist_ok=True)
    got = 0
    # Spread the sample across the test split (100 docs) rather than taking the first n.
    step = max(1, 100 // n)
    for off in range(0, 100, step):
        if got >= n:
            break
        dst = out / f"cord_{off:03d}.jpg"
        if dst.exists():
            got += 1
            continue
        rows = json.loads(get(API.format(ds="naver-clova-ix/cord-v2", split="test", off=off, n=1)))["rows"]
        row = rows[0]["row"]
        gt = cord_gt(row)
        img = get(row["image"]["src"])
        dst.write_bytes(img)
        dst.with_suffix(".json").write_text(json.dumps(gt, indent=2, ensure_ascii=False), encoding="utf-8")
        got += 1
        print(f"{dst.name}: {len(gt['line_items'])} items, total {gt['total']}", flush=True)


if __name__ == "__main__":
    main()
