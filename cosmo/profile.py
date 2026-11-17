"""Counts, group names, pairings and plain-English summary sentences.

No language model here: sentences are filled-in templates over counts.
"""
from __future__ import annotations

from collections import Counter
from itertools import combinations

import numpy as np

from . import config, labels


def plural(word: str) -> str:
    if word.endswith(("ss", "ch", "sh", "x")):
        return word + "es"
    if word.endswith("s") or word in {"activewear", "jewelry"}:
        return word
    return word + "s"


def material_word(m) -> str | None:
    """The material to state for one piece, or None if we should stay quiet."""
    if m is None or not m.confident or m.top in labels.PLAIN_MATERIALS:
        return None
    return m.top


def group_name(members: list[int], a) -> str:
    """e.g. "navy knit sweaters". An adjective appears only when the model is
    confident about it for enough of the group (config.NAME_AGREEMENT);
    otherwise the name says less rather than guessing."""
    def agreed(values):
        v, n = Counter(values).most_common(1)[0]
        return v if v and n / len(values) >= config.NAME_AGREEMENT else None

    cat = Counter(a.category[i].top for i in members).most_common(1)[0][0]
    color = agreed([a.color[i].top if a.color[i].confident else None for i in members])
    material = agreed([material_word(a.material[i]) for i in members])
    if material == cat or (material == "denim" and cat == "jeans"):
        material = None  # "denim jeans" says nothing
    return " ".join(w for w in [color, material, plural(cat)] if w)


def representative(members: list[int], vecs: np.ndarray) -> int:
    """The member closest to the group's average vector."""
    centre = vecs[members].mean(axis=0)
    return members[int(np.argmax(vecs[members] @ centre))]


def pairings(image_of: list[int], group_of: dict[int, int], cat_of: list[str]) -> list[tuple[tuple[int, int], int]]:
    """Count, for each pair of groups, how many images contain both.

    Only pairs of different categories count (a sweater and a sweater in the
    same flat-lay is not a pairing).
    """
    per_image: dict[int, set[int]] = {}
    for piece, g in group_of.items():
        per_image.setdefault(image_of[piece], set()).add(g)
    group_cat = {g: cat_of[p] for p, g in group_of.items()}
    counts: Counter = Counter()
    for groups in per_image.values():
        for a, b in combinations(sorted(groups), 2):
            if group_cat[a] != group_cat[b]:
                counts[(a, b)] += 1
    return counts.most_common()


def sentences(cats: list[str], colors: list[str], biggest: tuple[str, int] | None,
              top_pair: tuple[str, str, int] | None) -> list[str]:
    out = []
    n = len(cats)
    if n:
        kind, k = Counter(labels.KIND.get(c, c) for c in cats).most_common(1)[0]
        out.append(f"Most of what you save is {kind} ({k / n:.0%} of {n} pieces).")
        color, c = Counter(colors).most_common(1)[0]
        out.append(f"Your most saved color is {color} ({c / n:.0%}).")
    if biggest and biggest[1] > 1:
        out.append(f"Your largest group is {biggest[1]} {biggest[0]}.")
    if top_pair and top_pair[2] > 1:
        out.append(f"You keep pairing {top_pair[0]} with {top_pair[1]} ({top_pair[2]} saved outfits).")
    return out
