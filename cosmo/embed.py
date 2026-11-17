"""Stage 2: turn each image into a vector (an "embedding").

Similar-looking clothes get vectors that point in similar directions.
We L2-normalize every vector to length 1, so the dot product of two
vectors equals the cosine of the angle between them (1 = same direction).
"""
from __future__ import annotations

import functools
import gc
import os

import numpy as np
import torch

from . import config


def pick_device() -> str:
    if os.environ.get("COSMO_DEVICE"):  # manual override, e.g. COSMO_DEVICE=cpu
        return os.environ["COSMO_DEVICE"]
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():  # Apple Silicon GPU
        return "mps"
    return "cpu"


def low_memory() -> bool:
    """True on machines without a GPU (e.g. a free cloud host). In this mode
    only one of the two models is kept in memory at a time: together they
    peak near 3 GB, which is more than free hosts allow."""
    default = "1" if pick_device() == "cpu" else "0"
    return os.environ.get("COSMO_LOW_MEMORY", default) == "1"


def unload() -> None:
    """Drop the embedding model so its memory can be reused."""
    if load_model.cache_info().currsize:
        load_model.cache_clear()
        gc.collect()


@functools.lru_cache(maxsize=1)
def load_model():
    """Load once per process. Returns (model, preprocess, tokenizer, device)."""
    import open_clip

    if low_memory():
        from . import detect
        detect.unload()
    device = pick_device()
    model, _, preprocess = open_clip.create_model_and_transforms(
        config.MODEL_NAME, pretrained=config.PRETRAINED
    )
    tokenizer = open_clip.get_tokenizer(config.MODEL_NAME)
    model = model.to(device).eval()
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    config.ready_marker(config.MODEL_NAME).touch()
    return model, preprocess, tokenizer, device


def _cache_path(sha1: str, root=None):
    # Key by model too, so switching models never reuses stale vectors.
    model_key = config.MODEL_NAME.replace("/", "_").replace(":", "_")
    return (root or config.CACHE_DIR) / "embeddings" / model_key / f"{sha1}.npy"


def _cached(sha1: str):
    """Look in the local cache, then in the read-only cache shipped with the repo."""
    for root in (config.CACHE_DIR, config.SAMPLE_CACHE_DIR):
        p = _cache_path(sha1, root)
        if p.exists():
            return np.load(p)
    return None


def embed_images(items, progress=None) -> np.ndarray:
    """Return an (N, D) float32 array of unit vectors, one row per item.

    progress: optional callback(done, total) for the UI.
    """
    vecs: dict[int, np.ndarray] = {}
    todo = []
    for i, it in enumerate(items):
        v = _cached(it.sha1)
        if v is not None:
            vecs[i] = v
        else:
            todo.append(i)

    if todo:
        model, preprocess, _, device = load_model()
        for start in range(0, len(todo), config.BATCH_SIZE):
            idx = todo[start:start + config.BATCH_SIZE]
            batch = torch.stack([preprocess(items[i].image) for i in idx]).to(device)
            with torch.no_grad():
                feats = model.encode_image(batch).float()
            feats = torch.nn.functional.normalize(feats, dim=-1).cpu().numpy()
            for i, f in zip(idx, feats):
                vecs[i] = f
                p = _cache_path(items[i].sha1)
                p.parent.mkdir(parents=True, exist_ok=True)
                np.save(p, f)
            if progress:
                progress(min(start + config.BATCH_SIZE, len(todo)), len(todo))

    if not items:
        return np.zeros((0, 1), dtype=np.float32)
    return np.stack([vecs[i] for i in range(len(items))]).astype(np.float32)


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed text prompts into the same space as images (used for labeling)."""
    model, _, tokenizer, device = load_model()
    with torch.no_grad():
        feats = model.encode_text(tokenizer(texts).to(device)).float()
    return torch.nn.functional.normalize(feats, dim=-1).cpu().numpy()
