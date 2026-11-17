"""Stage 3: zero-shot labels.

The model puts images and text in the same vector space. To label a crop,
we embed each candidate label as text ("a photo of jeans, a type of
clothing") and pick the label whose vector is closest to the crop's vector.
No training data is involved.

Turning similarities into probabilities: multiply cosine similarities by the
model's learned temperature ("logit scale") and apply softmax. The gap
between the top two probabilities tells us how sure the model is.
"""
from __future__ import annotations

import functools
import hashlib
from dataclasses import dataclass

import numpy as np

from . import config, embed, labels


@dataclass
class Label:
    top: str
    score: float      # probability of the top label
    runner_up: str
    runner_score: float

    @property
    def low_confidence(self) -> bool:
        return self.score - self.runner_score < config.LOW_CONF_MARGIN

    @property
    def confident(self) -> bool:
        return self.score >= config.CONFIDENT_SCORE and not self.low_confidence

    def __str__(self):
        flag = " (low confidence)" if self.low_confidence else ""
        return f"{self.top} {self.score:.0%}{flag}"


def _text_cache_path(key: str, root):
    model_key = config.MODEL_NAME.replace("/", "_").replace(":", "_")
    return root / "text" / model_key / key


def _disk_cached(key: str, compute):
    """Return a small array from the disk cache (local, then the one shipped
    with the repo), computing and saving it on a miss. With these cached, a
    run whose images are also cached never has to load the model."""
    for root in (config.CACHE_DIR, config.SAMPLE_CACHE_DIR):
        p = _text_cache_path(key, root)
        if p.exists():
            return np.load(p)
    value = compute()
    p = _text_cache_path(key, config.CACHE_DIR)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.save(p, value)
    return value


@functools.lru_cache(maxsize=None)
def _text_matrix(names: tuple[str, ...], templates: tuple[str, ...]) -> np.ndarray:
    """One unit vector per label, averaged over the prompt templates."""
    def compute():
        rows = []
        for n in names:
            v = embed.embed_texts([t.format(n) for t in templates]).mean(axis=0)
            rows.append(v / np.linalg.norm(v))
        return np.stack(rows)

    # The key changes whenever a label or template is edited.
    key = hashlib.sha1(repr((names, templates)).encode()).hexdigest() + ".npy"
    return _disk_cached(key, compute)


@functools.lru_cache(maxsize=1)
def _logit_scale() -> float:
    compute = lambda: np.array(float(embed.load_model()[0].logit_scale.exp().item()))
    return float(_disk_cached("logit_scale.npy", compute))


def _probs(vecs: np.ndarray, text: np.ndarray) -> np.ndarray:
    logits = _logit_scale() * vecs @ text.T
    logits -= logits.max(axis=1, keepdims=True)  # numerical stability
    e = np.exp(logits)
    return e / e.sum(axis=1, keepdims=True)


def probabilities(vecs: np.ndarray, names: list[str], kind: str) -> np.ndarray:
    """(N, len(names)) matrix: each row is a probability distribution over labels."""
    if len(vecs) == 0:
        return np.zeros((0, len(names)), dtype=np.float32)
    return _probs(vecs, _text_matrix(tuple(names), tuple(labels.TEMPLATES[kind])))


def classify(vecs: np.ndarray, names: list[str], kind: str) -> list[Label]:
    p = probabilities(vecs, names, kind)
    order = np.argsort(-p, axis=1)
    return [
        Label(names[o[0]], float(row[o[0]]), names[o[1]], float(row[o[1]]))
        for row, o in zip(p, order)
    ]


def is_clothing(vecs: np.ndarray) -> np.ndarray:
    """True where the best-matching prompt is one of the clothing prompts."""
    if len(vecs) == 0:
        return np.zeros(0, dtype=bool)
    prompts = labels.CLOTHING_PROMPTS + labels.NOT_CLOTHING_PROMPTS
    p = _probs(vecs, _text_matrix(tuple(prompts), ("{}",)))
    return p.argmax(axis=1) < len(labels.CLOTHING_PROMPTS)
