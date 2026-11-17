"""Stage 1b: find each piece of clothing in an image and crop it out.

We use OWLv2, an "open-vocabulary" detector: you give it text queries like
"a photo of jeans" and it returns boxes for things matching them, without
any training on our side. The raw boxes are cached per image; the cleanup
rules below run on every load, so they can be tuned without re-detecting.
"""
from __future__ import annotations

import functools
import gc
import hashlib
import json
from dataclasses import dataclass

from PIL import Image

from . import config, labels


@dataclass
class Piece:
    image_idx: int                 # which ingest Item this came from
    box: tuple[int, int, int, int]  # x0, y0, x1, y1 in the normalized image
    image: Image.Image             # the crop (embed.py reads .image and .sha1)
    sha1: str                      # cache key: source image hash + box
    det_label: str                 # what the detector thought it was
    det_score: float
    whole: bool = False            # True if no boxes were found and we used the whole image


@functools.lru_cache(maxsize=1)
def _load():
    import torch
    from transformers import AutoProcessor, Owlv2ForObjectDetection

    from . import embed

    if embed.low_memory():
        embed.unload()  # one model in memory at a time
    device = embed.pick_device()
    proc = AutoProcessor.from_pretrained(config.DETECTOR_NAME)
    model = Owlv2ForObjectDetection.from_pretrained(config.DETECTOR_NAME).to(device).eval()
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    config.ready_marker(config.DETECTOR_NAME).touch()
    return proc, model, device, torch


def unload() -> None:
    """Drop the detector so its memory can be reused."""
    if _load.cache_info().currsize:
        _load.cache_clear()
        gc.collect()


def _cache_path(sha1: str, root=None):
    key = config.DETECTOR_NAME.replace("/", "_")
    return (root or config.CACHE_DIR) / "detections" / key / f"{sha1}.json"


def raw_detections(im: Image.Image, sha1: str) -> list[list]:
    """Return [[x0, y0, x1, y1, label, score], ...] above the score floor."""
    for root in (config.CACHE_DIR, config.SAMPLE_CACHE_DIR):  # local, then shipped with the repo
        if _cache_path(sha1, root).exists():
            return json.loads(_cache_path(sha1, root).read_text())
    p = _cache_path(sha1)
    proc, model, device, torch = _load()
    queries = [f"a photo of {c}" for c in labels.CATEGORIES]
    inputs = proc(text=[queries], images=im, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs)
    # OWLv2 pads the image to a square (image in the top-left corner), so
    # boxes come back in the coordinates of that square.
    side = max(im.size)
    res = proc.image_processor.post_process_object_detection(
        out, threshold=config.DET_SCORE_FLOOR,
        target_sizes=torch.tensor([[side, side]], device=device),
    )[0]
    dets = [
        [*(round(float(v), 1) for v in b), labels.CATEGORIES[int(l)], round(float(s), 4)]
        for b, s, l in zip(res["boxes"], res["scores"], res["labels"])
    ]
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(dets))
    return dets


# --- cleanup rules ----------------------------------------------------------

def _area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _inter(a, b):
    return _area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))


def _union(a, b):
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def clean_boxes(dets: list[list], width: int, height: int) -> list[dict]:
    """Turn raw detections into one box per real piece.

    1. Drop weak boxes and tiny boxes.
    2. If one box mostly sits inside another, keep one. Prefer the larger box
       unless it is much less confident (fixes "half a pair of trousers").
    3. Merge footwear boxes of the same kind that sit next to each other
       (one pair of shoes is one item, not two).
    """
    img_area = width * height
    boxes = []
    for x0, y0, x1, y1, lab, score in dets:
        b = (max(0, x0), max(0, y0), min(width, x1), min(height, y1))
        if score < config.DET_SCORE and lab not in labels.FOOTWEAR:
            continue
        if score < config.DET_SCORE_FOOTWEAR:
            continue
        if _area(b) < config.DET_MIN_AREA * img_area:
            continue
        boxes.append({"box": b, "label": lab, "score": score})

    boxes.sort(key=lambda d: -d["score"])
    kept: list[dict] = []
    for d in boxes:
        clash = None
        for k in kept:
            small = min(_area(d["box"]), _area(k["box"]))
            if small and _inter(d["box"], k["box"]) / small > config.DET_CONTAIN:
                clash = k
                break
        if clash is None:
            kept.append(d)
        elif (d["label"] == clash["label"] and _area(d["box"]) > _area(clash["box"])
              and d["score"] >= 0.6 * clash["score"]):
            # Same kind of piece: keep the bigger box (fixes "half a pair of
            # trousers"). Different kinds: the higher score already won.
            clash["box"] = d["box"]

    merged = True
    while merged:
        merged = False
        for i in range(len(kept)):
            for j in range(i + 1, len(kept)):
                a, b = kept[i], kept[j]
                if a["label"] not in labels.FOOTWEAR or b["label"] not in labels.FOOTWEAR:
                    continue
                if a["label"] != b["label"]:
                    continue
                ab, bb = a["box"], b["box"]
                # Close in any direction: side by side, stacked, or overlapping.
                gap_x = max(bb[0] - ab[2], ab[0] - bb[2], 0)
                gap_y = max(bb[1] - ab[3], ab[1] - bb[3], 0)
                if gap_x < 0.06 * width and gap_y < 0.06 * height:
                    a["box"] = _union(ab, bb)
                    a["score"] = max(a["score"], b["score"])
                    kept.pop(j)
                    merged = True
                    break
            if merged:
                break
    return kept


def _crop(im: Image.Image, box) -> tuple[Image.Image, tuple[int, int, int, int]]:
    w, h = im.size
    pad_x = (box[2] - box[0]) * 0.04
    pad_y = (box[3] - box[1]) * 0.04
    b = (int(max(0, box[0] - pad_x)), int(max(0, box[1] - pad_y)),
         int(min(w, box[2] + pad_x)), int(min(h, box[3] + pad_y)))
    return im.crop(b), b


def detect_pieces(items, progress=None) -> list[Piece]:
    pieces: list[Piece] = []
    for idx, it in enumerate(items):
        dets = clean_boxes(raw_detections(it.image, it.sha1), *it.image.size)
        if not dets:
            w, h = it.image.size
            pieces.append(Piece(idx, (0, 0, w, h), it.image, it.sha1, "(none found)", 0.0, whole=True))
        for d in sorted(dets, key=lambda d: (d["box"][1], d["box"][0])):  # top-to-bottom reading order
            crop, b = _crop(it.image, d["box"])
            key = hashlib.sha1(f"{it.sha1}:{b}".encode()).hexdigest()
            pieces.append(Piece(idx, b, crop, key, d["label"], d["score"]))
        if progress:
            progress(idx + 1, len(items))
    return pieces
