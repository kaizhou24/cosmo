"""All tunable settings in one place.

To switch models, change MODEL_NAME (and PRETRAINED for non-hub models).
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# --- Model ---------------------------------------------------------------
# Primary: fashion-tuned SigLIP from Marqo, loaded from the Hugging Face hub.
MODEL_NAME = "hf-hub:Marqo/marqo-fashionSigLIP"
PRETRAINED = None
# Fallback (one-line switch): general-purpose CLIP.
# MODEL_NAME = "ViT-B-32"
# PRETRAINED = "laion2b_s34b_b79k"

BATCH_SIZE = 16

# Garment detector (open-vocabulary, text queries come from labels.CATEGORIES).
DETECTOR_NAME = "google/owlv2-base-patch16-ensemble"
DET_SCORE_FLOOR = 0.1      # cached raw boxes keep everything above this
DET_SCORE = 0.22           # boxes below this are dropped...
DET_SCORE_FOOTWEAR = 0.15  # ...except shoes, which score lower but are usually right
DET_MIN_AREA = 0.006       # drop boxes smaller than 0.6% of the image
DET_CONTAIN = 0.6          # a box >60% inside another is the same piece

# --- Labels ---------------------------------------------------------------
LOW_CONF_MARGIN = 0.15     # top-1 minus top-2 probability below this = low confidence
CONFIDENT_SCORE = 0.60     # a label is only used in a group name above this probability
NAME_AGREEMENT = 0.60      # ...and only if this share of the group agrees on it

# --- Ingest limits --------------------------------------------------------
MAX_FILES = 500
MAX_FILES_LOW_MEMORY = 60   # cap on machines without a GPU (the free hosted demo)
MAX_UNCOMPRESSED_BYTES = 300 * 1024 * 1024  # 300 MB
MAX_SIDE_PX = 1024
DUPLICATE_HAMMING = 5  # phash distance <= this counts as a duplicate

# --- Grouping -------------------------------------------------------------
DEFAULT_DISTANCE_THRESHOLD = 0.24  # embedding distance + color term; lower = tighter groups

# How much a color mismatch pushes two pieces apart. Color difference runs from
# 0 (same color) to 1 (completely different) and is multiplied by this before
# being added to the embedding distance.
COLOR_WEIGHT = 0.15

# --- Paths ----------------------------------------------------------------
CACHE_DIR = Path(os.environ.get("COSMO_CACHE_DIR", ROOT / ".cache"))
SAMPLE_DIR = ROOT / "sample_data"
# Precomputed boxes and vectors for the sample set, committed to the repo so
# the hosted demo's sample button does not have to run the detector.
SAMPLE_CACHE_DIR = ROOT / "sample_cache"

# --- Offline once cached ----------------------------------------------------
# After both models have loaded successfully once, stop contacting the
# Hugging Face hub (this also silences its "unauthenticated requests" warning).
# The marker files are written by embed.py and detect.py after a good load,
# so an interrupted download can never leave the app stuck offline.
def ready_marker(name: str) -> Path:
    return CACHE_DIR / ("ready-" + name.replace("/", "_").replace(":", "_"))


if "HF_HUB_OFFLINE" not in os.environ and all(
    ready_marker(n).exists() for n in (MODEL_NAME, DETECTOR_NAME)
):
    os.environ["HF_HUB_OFFLINE"] = "1"
