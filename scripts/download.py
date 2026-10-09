"""Resumable multi-connection downloader for slow/flaky links (no extra deps).

Usage: python scripts/download.py URL DEST [--parts 6]
"""

from __future__ import annotations

import argparse
import threading
import time
import urllib.request
from pathlib import Path


def size_of(url: str) -> tuple[str, int]:
    req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        final = r.geturl()
        cr = r.headers.get("Content-Range", "")
        return final, int(cr.rsplit("/", 1)[1]) if "/" in cr else int(r.headers["Content-Length"])


def worker(url: str, part: Path, start: int, end: int, prog: list[int], i: int) -> None:
    while True:
        have = part.stat().st_size if part.exists() else 0
        prog[i] = have
        if start + have > end:
            return
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={start + have}-{end}"})
            with urllib.request.urlopen(req, timeout=60) as r, open(part, "ab") as f:
                while chunk := r.read(1 << 18):
                    f.write(chunk)
                    prog[i] += len(chunk)
        except Exception:  # noqa: BLE001
            time.sleep(3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("dest", type=Path)
    ap.add_argument("--parts", type=int, default=6)
    a = ap.parse_args()
    if a.dest.exists():
        print(f"DONE {a.dest} (exists)", flush=True)
        return
    url, total = size_of(a.url)
    a.dest.parent.mkdir(parents=True, exist_ok=True)
    step = -(-total // a.parts)
    ranges = [(i * step, min(total, (i + 1) * step) - 1) for i in range(a.parts)]
    parts = [a.dest.with_name(f"{a.dest.name}.part{i}") for i in range(a.parts)]
    prog = [0] * a.parts
    ts = [threading.Thread(target=worker, args=(url, parts[i], s, e, prog, i), daemon=True) for i, (s, e) in enumerate(ranges)]
    for t in ts:
        t.start()
    t0, last = time.time(), 0
    while any(t.is_alive() for t in ts):
        time.sleep(30)
        done = sum(prog)
        print(f"{a.dest.name}: {done / total:6.1%} {done / 1e6:.0f}/{total / 1e6:.0f} MB  {(done - last) / 30e3:.0f} KB/s", flush=True)
        last = done
    with open(a.dest.with_suffix(a.dest.suffix + ".tmp"), "wb") as out:
        for p in parts:
            out.write(p.read_bytes())
    tmp = a.dest.with_suffix(a.dest.suffix + ".tmp")
    if tmp.stat().st_size != total:
        raise SystemExit(f"size mismatch {tmp.stat().st_size} != {total}; rerun to resume")
    tmp.rename(a.dest)
    for p in parts:
        p.unlink()
    print(f"DONE {a.dest} in {(time.time() - t0) / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
