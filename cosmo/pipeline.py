"""Run every stage in order. Shared by app.py and eval.py."""
from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np

from . import config, detect, embed, group, label, labels


@dataclass
class Analysis:
    items: list                 # ingest.Item, one per kept image
    pieces: list                # detect.Piece, one per detected garment
    piece_vecs: np.ndarray      # (P, D) embeddings of crops
    clothing: np.ndarray        # (P,) bool: crop passed the clothing check
    category: list              # label.Label per piece
    color: list
    color_probs: np.ndarray     # (P, n_colors) full distribution, used when grouping
    material: list              # label.Label or None (watches, glasses, jewelry)
    aesthetic: list             # label.Label per *image* (style is an outfit-level idea)

    @property
    def kept(self) -> list[int]:
        """Indices of pieces that are clothing."""
        return [i for i in range(len(self.pieces)) if self.clothing[i]]


# Only one analysis runs at a time. On a shared host two visitors could
# otherwise load both models at once and run out of memory.
_LOCK = threading.Lock()


def analyze(items, step=lambda msg: None, progress=lambda d, t: None) -> Analysis:
    if _LOCK.locked():
        step("Waiting for another visitor's upload to finish")
    with _LOCK:
        return _analyze(items, step, progress)


def _analyze(items, step, progress) -> Analysis:
    step("Finding each piece of clothing")
    pieces = detect.detect_pieces(items, progress=progress)

    if embed.low_memory():
        detect.unload()
    step(f"Embedding {len(pieces)} pieces")
    piece_vecs = embed.embed_images(pieces, progress=progress)
    image_vecs = embed.embed_images(items)

    step("Labeling")
    clothing = label.is_clothing(piece_vecs)
    category = label.classify(piece_vecs, labels.CATEGORIES, "category")
    color = label.classify(piece_vecs, labels.COLORS, "color")
    material_all = label.classify(piece_vecs, labels.MATERIALS, "material")
    color_probs = label.probabilities(piece_vecs, labels.COLORS, "color")
    material = [None if c.top in labels.NO_MATERIAL else m for c, m in zip(category, material_all)]
    aesthetic = label.classify(image_vecs, labels.AESTHETICS, "aesthetic")
    return Analysis(items, pieces, piece_vecs, clothing, category, color, color_probs, material, aesthetic)


def make_groups(a: Analysis, threshold: float = config.DEFAULT_DISTANCE_THRESHOLD) -> list[list[int]]:
    """Groups of piece indices (into a.pieces), largest first. Clothing only."""
    kept = a.kept
    local = group.group_by_category(
        a.piece_vecs[kept], [a.category[i].top for i in kept], threshold, a.color_probs[kept]
    )
    return [[kept[j] for j in g] for g in local]
