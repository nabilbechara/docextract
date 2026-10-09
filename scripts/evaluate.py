"""Score the extractor against a folder of documents with ground-truth JSON next to them.

Ground truth format (same stem as the image): {"line_items":[{"name","quantity","total"}...], "total": ...}
Images without a .json are still run (and printed) but not scored, e.g. raw internet photos.

Usage: python scripts/evaluate.py data/synthetic [--engine auto|vlm|ocr] [--out results/x.jsonl]
"""

from __future__ import annotations

import argparse
import difflib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docextract.pipeline import Extractor  # noqa: E402

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".pdf", ".tif", ".tiff", ".bmp", ".heic"}


def norm(s: str) -> str:
    return " ".join("".join(c.lower() if c.isalnum() else " " for c in (s or "")).split())


def close(a, b, tol=0.015) -> bool:
    return a is not None and b is not None and abs(float(a) - float(b)) <= max(tol, 0.005 * abs(float(b)))


def score(pred: dict, gt: dict) -> dict:
    gi, pi = gt.get("line_items", []), pred.get("line_items", [])
    used: set[int] = set()
    matched = qty_ok = tot_ok = 0
    for g in gi:
        best, bj = 0.0, -1
        for j, p in enumerate(pi):
            if j in used:
                continue
            r = difflib.SequenceMatcher(None, norm(g["name"]), norm(p.get("name", ""))).ratio()
            if r > best:
                best, bj = r, j
        if bj >= 0 and best >= 0.6:
            used.add(bj)
            matched += 1
            p = pi[bj]
            qty_ok += close(p.get("quantity"), g.get("quantity"), 0.001) if g.get("quantity") is not None else 1
            tot_ok += close(p.get("total"), g.get("total")) if g.get("total") is not None else 1
    total_ok = close(pred.get("total"), gt.get("total")) if gt.get("total") is not None else None
    perfect = matched == len(gi) == len(pi) and qty_ok == tot_ok == matched and total_ok is not False
    return {
        "gt_items": len(gi),
        "pred_items": len(pi),
        "matched": matched,
        "qty_ok": qty_ok,
        "item_total_ok": tot_ok,
        "total_ok": total_ok,
        "perfect": perfect,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=Path)
    ap.add_argument("--engine", default="auto")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    ex = Extractor()
    files = sorted(p for p in args.folder.rglob("*") if p.suffix.lower() in EXTS)
    if args.limit:
        files = files[: args.limit]
    agg: dict[str, dict] = defaultdict(lambda: defaultdict(float))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
    outf = args.out.open("w", encoding="utf-8") if args.out else None

    for f in files:
        t0 = time.monotonic()
        try:
            res = ex.extract_bytes(f.read_bytes(), f.name, args.engine)
            pred = res.model_dump()
            pred["raw_text"] = None
            err = None
        except Exception as e:  # noqa: BLE001
            pred, err = {"line_items": []}, str(e)
        dt = time.monotonic() - t0
        gtp = f.with_suffix(".json")
        rec = {"file": str(f), "seconds": round(dt, 1), "error": err, "pred": pred}
        if gtp.exists():
            gt = json.loads(gtp.read_text(encoding="utf-8"))
            s = score(pred, gt)
            rec["score"] = s
            for bucket in ("ALL", gt.get("style", "real")):
                a = agg[bucket]
                a["docs"] += 1
                a["seconds"] += dt
                for k in ("gt_items", "pred_items", "matched", "qty_ok", "item_total_ok"):
                    a[k] += s[k]
                a["total_ok"] += bool(s["total_ok"])
                a["total_n"] += s["total_ok"] is not None
                a["perfect"] += s["perfect"]
                a["errors"] += err is not None
            tag = "PERFECT" if s["perfect"] else f"items {s['matched']}/{s['gt_items']} (pred {s['pred_items']}) qty {s['qty_ok']} tot {s['item_total_ok']} total {s['total_ok']}"
        else:
            tag = "(no ground truth)"
        if not args.quiet:
            print(f"{f.name:40s} {dt:5.1f}s conf={pred.get('confidence', 0):.2f} {pred.get('engine', '')} {tag} {err or ''}", flush=True)
        if outf:
            outf.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if agg:
        print("\n%-12s %5s %8s %8s %8s %8s %8s %8s %6s" % ("bucket", "docs", "recall", "precis", "qty", "itemtot", "total", "perfect", "s/doc"))
        for b, a in sorted(agg.items()):
            print(
                "%-12s %5d %7.1f%% %7.1f%% %7.1f%% %7.1f%% %7.1f%% %7.1f%% %6.1f"
                % (
                    b, a["docs"],
                    100 * a["matched"] / max(1, a["gt_items"]),
                    100 * a["matched"] / max(1, a["pred_items"]),
                    100 * a["qty_ok"] / max(1, a["matched"]),
                    100 * a["item_total_ok"] / max(1, a["matched"]),
                    100 * a["total_ok"] / max(1, a["total_n"]),
                    100 * a["perfect"] / a["docs"],
                    a["seconds"] / a["docs"],
                )
            )


if __name__ == "__main__":
    main()
