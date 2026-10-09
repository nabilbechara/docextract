"""Download an Ollama model straight from the registry (resumable, retrying) into ~/.ollama/models.

Useful on slow/flaky connections where `ollama pull` keeps restarting.
Usage: python scripts/fetch_ollama_model.py qwen3-vl:4b
"""
import hashlib
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

REG = "https://registry.ollama.ai/v2/library"
ROOT = Path(os.environ.get("OLLAMA_MODELS", Path.home() / ".ollama" / "models"))


def get(url, headers=None):
    req = urllib.request.Request(url, headers=headers or {})
    return urllib.request.urlopen(req, timeout=60)


def fetch_blob(digest, size):
    dst = ROOT / "blobs" / digest.replace(":", "-")
    part = dst.with_suffix(".partial")
    if dst.exists() and dst.stat().st_size == size:
        print(f"have {digest[:19]}", flush=True)
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    while True:
        have = part.stat().st_size if part.exists() else 0
        if have >= size:
            break
        try:
            with get(f"{REG}/{NAME}/blobs/{digest}", {"Range": f"bytes={have}-"}) as r, open(part, "ab") as f:
                last = time.time()
                while chunk := r.read(1 << 20):
                    f.write(chunk)
                    have += len(chunk)
                    if time.time() - last > 30:
                        print(f"{digest[:19]} {have / size:6.1%} of {size / 1e9:.2f} GB", flush=True)
                        last = time.time()
        except Exception as e:  # noqa: BLE001
            print(f"retry after error: {e}", flush=True)
            time.sleep(5)
    h = hashlib.sha256()
    with open(part, "rb") as f:
        while b := f.read(1 << 24):
            h.update(b)
    if "sha256:" + h.hexdigest() != digest:
        part.unlink()
        raise SystemExit(f"checksum mismatch for {digest}, deleted; rerun")
    part.rename(dst)
    print(f"done {digest[:19]}", flush=True)


if __name__ == "__main__":
    NAME, TAG = (sys.argv[1].split(":") + ["latest"])[:2]
    accept = {"Accept": "application/vnd.docker.distribution.manifest.v2+json"}
    manifest = get(f"{REG}/{NAME}/manifests/{TAG}", accept).read()
    m = json.loads(manifest)
    for layer in sorted(m["layers"] + [m["config"]], key=lambda l: l["size"]):
        fetch_blob(layer["digest"], layer["size"])
    mp = ROOT / "manifests" / "registry.ollama.ai" / "library" / NAME / TAG
    mp.parent.mkdir(parents=True, exist_ok=True)
    mp.write_bytes(manifest)
    print(f"MODEL READY {NAME}:{TAG}", flush=True)
