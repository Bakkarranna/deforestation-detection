"""Chunked parallel downloader for the OSCD benchmark from Hugging Face.

OSCD (Onera Satellite Change Detection) ships as three zip archives. Hugging Face
allows anonymous range requests, so we slice each archive into ~8 MB chunks and
fetch them all in one thread pool: a single CDN connection tops out around
1.5 MB/s here, while 16 parallel connections saturate it.

Chunks are written as `<archive>.part<nnn>` next to the destination, so an
interrupted run resumes without re-downloading anything. Returns the archive
names that could not be completed, so callers fail loudly instead of training
on truncated archives.
"""

from __future__ import annotations

import shutil
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterator

BASE_URL = "https://huggingface.co/datasets/hkristen/oscd/resolve/main"
# Order matters: labels are small, images are the bulk of the transfer.
ARCHIVES = {
    "train_labels": "Onera Satellite Change Detection dataset - Train Labels.zip",
    "test_labels": "Onera Satellite Change Detection dataset - Test Labels.zip",
    "images": "Onera Satellite Change Detection dataset - Images.zip",
}
WORKERS = 16
CHUNK = 8 * 1024 * 1024
RETRIES = 10


def _open(url: str, headers: dict):
    req = urllib.request.Request(url, headers=headers)
    return urllib.request.urlopen(req, timeout=900)


def remote_total(url: str) -> int | None:
    """Total size in bytes, read from a 1-byte range request."""
    try:
        with _open(url, {"Range": "bytes=0-0"}) as r:
            content_range = r.headers.get("Content-Range", "")
            if "/" in content_range:
                return int(content_range.split("/")[-1])
            size = int(r.headers.get("Content-Length", 0))
            return size or None
    except (urllib.error.URLError, ValueError, OSError):
        return None


def _unlink(path: Path) -> None:
    """Remove a chunk file, tolerating a handle Windows has not released yet."""
    for delay in (0.0, 0.2, 1.0, 3.0):
        try:
            path.unlink(missing_ok=True)
            return
        except PermissionError:
            if delay:
                time.sleep(delay)


def fetch_range(url: str, dest: Path, start: int, end: int, retries: int = RETRIES) -> bool:
    """Fetch bytes [start, end) into dest; retries on a dropped connection."""
    if dest.exists() and dest.stat().st_size == end - start:
        return True
    for attempt in range(retries):
        try:
            with _open(url, {"Range": f"bytes={start}-{end - 1}"}) as resp, \
                    open(dest, "wb") as fh:
                shutil.copyfileobj(resp, fh, length=1 << 20)
            return dest.stat().st_size == end - start
        except (urllib.error.URLError, OSError, ValueError):
            if attempt == retries - 1:
                return False
            time.sleep(min(2 ** attempt, 15))


def _chunks(total: int, chunk: int = None) -> Iterator[tuple[int, int]]:
    """Split [0, total) into ranges of at most `chunk` bytes."""
    chunk = chunk or CHUNK
    start = 0
    while start < total:
        end = min(start + chunk, total)
        yield start, end
        start = end


def download_file(key: str, dest_dir: Path, workers: int = WORKERS) -> bool:
    """Download one archive in parallel chunks. True when it is complete."""
    name = ARCHIVES[key]
    dest = dest_dir / name
    url = f"{BASE_URL}/{urllib.request.quote(name)}"
    total = remote_total(url)
    if total is None:
        print(f"  {key}: could not reach {url}")
        return False
    if dest.exists() and dest.stat().st_size == total:
        print(f"  {key}: already complete ({total / 1e6:.0f} MB)")
        return True

    tasks = []
    for i, (s, e) in enumerate(_chunks(total)):
        part = dest.with_name(f"{dest.name}.part{i:04d}")
        if part.exists() and part.stat().st_size == e - s:
            continue
        tasks.append((url, part, s, e))
    print(f"  {key}: {total / 1e6:.0f} MB, {len(tasks)} chunks to fetch")

    if tasks:
        done = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(fetch_range, u, p, s, e): p for u, p, s, e in tasks}
            for fut in as_completed(futures):
                part = futures[fut]
                done += 1
                if not fut.result():
                    print(f"    WARNING chunk failed: {part.name}")
                if done % 40 == 0 or done == len(tasks):
                    print(f"    {done}/{len(tasks)} chunks", flush=True)

    parts = sorted(dest.parent.glob(f"{dest.name}.part*"),
                   key=lambda p: int(p.name.rsplit("part", 1)[1]))
    if parts:
        with open(dest, "wb") as out:
            for part in parts:
                with open(part, "rb") as fh:
                    shutil.copyfileobj(fh, out)
                _unlink(part)
    ok = dest.exists() and dest.stat().st_size == total
    print(f"  {key}: {'complete' if ok else 'INCOMPLETE'}")
    return ok


def extract(archive: Path, dest_dir: Path) -> Path:
    """Extract an OSCD zip and return the single top-level directory it made."""
    marker = dest_dir / f".{archive.stem}.done"
    if marker.exists():
        return dest_dir
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(dest_dir)
    marker.write_text("ok")
    return dest_dir


def download_all(dest_dir: Path) -> dict:
    """Download every archive and extract it. Returns per-archive status."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    status = {}
    for key in ARCHIVES:
        ok = download_file(key, dest_dir)
        status[key] = ok
        if ok:
            extract(dest_dir / ARCHIVES[key], dest_dir)
    return status


if __name__ == "__main__":
    import sys

    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/raw")
    print(download_all(target))
