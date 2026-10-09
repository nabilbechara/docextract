"""CLI: python -m docextract.cli receipt.jpg [more files] [--json] [--engine auto|vlm|ocr]"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .pipeline import Extractor


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="docextract", description=__doc__)
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--json", action="store_true", help="print JSON instead of readable text")
    ap.add_argument("--engine", default="auto", choices=["auto", "vlm", "ocr"])
    args = ap.parse_args(argv)

    ex = Extractor()
    rc = 0
    for f in args.files:
        try:
            res = ex.extract_bytes(f.read_bytes(), f.name, args.engine)
        except Exception as e:  # noqa: BLE001
            print(f"{f}: ERROR {e}", file=sys.stderr)
            rc = 1
            continue
        res.raw_text = None
        if args.json:
            print(json.dumps({"file": str(f), **res.model_dump()}, ensure_ascii=False, indent=2))
        else:
            print(f"=== {f} ===\n{res.readable()}\n")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
