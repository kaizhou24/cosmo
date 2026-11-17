"""Stage 1: turn a ZIP (or a folder) into a clean list of RGB images.

Steps: safe extraction -> open as image -> fix rotation, RGB, downscale
-> drop near-duplicates by perceptual hash.
"""
from __future__ import annotations

import hashlib
import io
import os
import stat
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import imagehash
from PIL import Image, ImageOps

from . import config

try:  # HEIC (iPhone photos) support is optional
    import pillow_heif

    pillow_heif.register_heif_opener()
    HEIC_SUPPORTED = True
except Exception:  # pragma: no cover - depends on platform wheels
    HEIC_SUPPORTED = False

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"} | ({".heic", ".heif"} if HEIC_SUPPORTED else set())


@dataclass
class Item:
    """One kept image."""
    name: str           # path relative to the ZIP root, used for display/eval
    path: Path          # where the normalized copy lives on disk
    sha1: str           # hash of the file's bytes, used as cache key
    image: Image.Image  # RGB, long side <= MAX_SIDE_PX


@dataclass
class IngestResult:
    items: list[Item] = field(default_factory=list)
    duplicates: int = 0
    unreadable: int = 0          # files with an image extension that failed to open
    skipped_heic: int = 0        # HEIC files skipped because support is missing
    skipped_other: int = 0       # non-image files, dotfiles, __MACOSX, symlinks
    rejected_unsafe: int = 0     # entries that tried to escape the target dir
    truncated: bool = False      # hit the file-count or size cap


def _is_junk(name: str) -> bool:
    parts = Path(name).parts
    return any(p == "__MACOSX" or p.startswith(".") for p in parts)


def safe_extract(zip_source, dest: Path) -> IngestResult:
    """Extract a ZIP into dest, refusing anything dangerous.

    zip_source: a path, or raw bytes (from a Streamlit upload).
    """
    res = IngestResult()
    if isinstance(zip_source, (bytes, bytearray)):
        zip_source = io.BytesIO(zip_source)
    dest = dest.resolve()
    total = 0
    count = 0
    with zipfile.ZipFile(zip_source) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = info.filename
            # Symlinks are stored with the unix S_IFLNK bit in external_attr.
            target = (dest / name).resolve()
            # "Zip slip": an entry like ../../etc/passwd resolves outside dest.
            if not target.is_relative_to(dest):
                res.rejected_unsafe += 1
                continue
            if stat.S_ISLNK(info.external_attr >> 16) or _is_junk(name):
                res.skipped_other += 1
                continue
            ext = target.suffix.lower()
            if ext in {".heic", ".heif"} and not HEIC_SUPPORTED:
                res.skipped_heic += 1
                continue
            if ext not in IMAGE_EXTS:
                res.skipped_other += 1
                continue
            if count >= config.MAX_FILES or total + info.file_size > config.MAX_UNCOMPRESSED_BYTES:
                res.truncated = True
                break
            target.parent.mkdir(parents=True, exist_ok=True)
            # Copy in chunks and count real bytes: file_size in the header can lie.
            written = 0
            with zf.open(info) as src, open(target, "wb") as out:
                while chunk := src.read(1 << 20):
                    written += len(chunk)
                    if total + written > config.MAX_UNCOMPRESSED_BYTES:
                        break
                    out.write(chunk)
            if total + written > config.MAX_UNCOMPRESSED_BYTES:
                target.unlink(missing_ok=True)
                res.truncated = True
                break
            total += written
            count += 1
    return res


def _load(path: Path) -> Image.Image | None:
    try:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im)  # honor phone rotation metadata
            im = im.convert("RGB")
            im.thumbnail((config.MAX_SIDE_PX, config.MAX_SIDE_PX))  # keeps aspect ratio
            return im.copy()
    except Exception:
        return None


def load_images(folder: Path, res: IngestResult | None = None) -> IngestResult:
    """Open every image under folder, normalize it, and drop near-duplicates."""
    res = res or IngestResult()
    folder = folder.resolve()
    files = sorted(
        p for p in folder.rglob("*")
        if p.is_file() and not p.is_symlink() and not _is_junk(str(p.relative_to(folder)))
    )
    seen_hashes: list[imagehash.ImageHash] = []
    for p in files:
        ext = p.suffix.lower()
        if ext in {".heic", ".heif"} and not HEIC_SUPPORTED:
            res.skipped_heic += 1
            continue
        if ext not in IMAGE_EXTS:
            res.skipped_other += 1
            continue
        if len(res.items) + res.duplicates >= config.MAX_FILES:
            res.truncated = True
            break
        im = _load(p)
        if im is None:
            res.unreadable += 1
            continue
        # Perceptual hash: similar-looking images get hashes that differ in few bits.
        h = imagehash.phash(im)
        if any(h - s <= config.DUPLICATE_HAMMING for s in seen_hashes):
            res.duplicates += 1
            continue
        seen_hashes.append(h)
        # Hash the file's bytes, not the decoded pixels: JPEG decoders differ
        # slightly between machines, and the cache key must not.
        sha1 = hashlib.sha1(p.read_bytes()).hexdigest()
        res.items.append(Item(name=str(p.relative_to(folder)), path=p, sha1=sha1, image=im))
    return res


def ingest_zip(zip_source, workdir: Path | None = None) -> tuple[IngestResult, tempfile.TemporaryDirectory | None]:
    """Extract and load. Returns the temp dir handle so the caller controls cleanup."""
    tmp = None
    if workdir is None:
        tmp = tempfile.TemporaryDirectory(prefix="cosmo_")
        workdir = Path(tmp.name)
    res = safe_extract(zip_source, workdir)
    try:
        load_images(workdir, res)
    except Exception:
        if tmp:
            tmp.cleanup()
        raise
    return res, tmp


def ingest_folder(folder: Path) -> IngestResult:
    return load_images(Path(folder))
