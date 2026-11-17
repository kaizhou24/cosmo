"""Measure label and grouping quality against hand labels.

Step 1, make a blank template (one row per detected piece):
    python eval.py --make-template data/
  This writes labels.csv and saves each crop to eval_crops/ so you can see
  which piece each row refers to. Fill in `category` and `group_id` by hand.

Step 2, score:
    python eval.py data/
  Prints the results and saves them to eval_results.md.

What the two numbers mean:
  Category accuracy: share of pieces whose predicted category equals yours.
  Cluster purity: for each predicted group, the share of its pieces that
    belong to that group's most common hand group_id; averaged over groups,
    weighted by group size. 1.0 means no group mixes two of your groups.
    Purity rewards splitting (a group of one is always pure), so the script
    also reports purity over groups of 2+ and the number of groups.

Not measured: garments the detector missed. Only detected pieces are scored.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

from cosmo import config, ingest, labels, pipeline

FIELDS = ["filename", "piece", "crop_file", "category", "group_id"]


def run(folder: Path):
    res = ingest.ingest_folder(folder)
    if not res.items:
        sys.exit(f"No images found in {folder}")
    a = pipeline.analyze(res.items, step=lambda m: print(m, file=sys.stderr))
    # Piece number = position within its image, in detection order (1-based).
    # Detections are cached and sorted, so the numbering is stable across runs.
    keys, counter = {}, Counter()
    for pi, p in enumerate(a.pieces):
        counter[p.image_idx] += 1
        keys[pi] = (res.items[p.image_idx].name, counter[p.image_idx])
    return a, keys


def make_template(folder: Path, out: Path, crops: Path, force: bool) -> None:
    if out.exists() and not force:
        sys.exit(f"{out} already exists. Refusing to overwrite hand labels (use --force to replace it).")
    a, keys = run(folder)
    crops.mkdir(parents=True, exist_ok=True)
    rows = []
    for pi in a.kept:
        name, n = keys[pi]
        crop_file = f"{Path(name).stem}__piece{n}.jpg"
        a.pieces[pi].image.save(crops / crop_file, quality=90)
        rows.append({"filename": name, "piece": n, "crop_file": crop_file, "category": "", "group_id": ""})
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out} with {len(rows)} rows and saved crops to {crops}/")
    print("Fill in `category` using exactly one of:")
    print("  " + ", ".join(labels.CATEGORIES))
    print("and `group_id` with any name you like; pieces you consider the same kind of item share one.")


def purity(groups: list[list[int]], hand: dict[int, str]) -> tuple[float | None, int]:
    """Weighted purity over the given groups, counting only hand-labeled pieces."""
    hit = total = 0
    for g in groups:
        ids = [hand[i] for i in g if i in hand]
        if ids:
            hit += Counter(ids).most_common(1)[0][1]
            total += len(ids)
    return (hit / total if total else None), total


def evaluate(folder: Path, labels_csv: Path, out: Path, threshold: float) -> None:
    if not labels_csv.exists():
        sys.exit(f"{labels_csv} not found. Run: python eval.py --make-template {folder}")
    a, keys = run(folder)
    by_key = {k: pi for pi, k in keys.items()}
    hand_cat: dict[int, str] = {}
    hand_group: dict[int, str] = {}
    unmatched = 0
    with open(labels_csv, newline="") as f:
        for row in csv.DictReader(f):
            pi = by_key.get((row["filename"], int(row["piece"])))
            if pi is None or not a.clothing[pi]:
                unmatched += 1
                continue
            if row.get("category", "").strip():
                hand_cat[pi] = row["category"].strip().lower()
            if row.get("group_id", "").strip():
                hand_group[pi] = row["group_id"].strip()
    if not hand_cat and not hand_group:
        sys.exit(f"{labels_csv} has no filled-in rows yet. Fill in `category` and `group_id` first.")

    lines = ["# Cosmo evaluation", ""]
    lines += [f"- Date: {date.today().isoformat()}",
              f"- Embedding model: `{config.MODEL_NAME}`",
              f"- Detector: `{config.DETECTOR_NAME}`",
              f"- Distance threshold: {threshold} (color weight {config.COLOR_WEIGHT})",
              f"- Images: {len(a.items)}; detected clothing pieces: {len(a.kept)}",
              "- Scope: only detected pieces are scored. Garments the detector missed are not counted.", ""]

    # --- Category accuracy ---
    wrong = []
    if hand_cat:
        unknown = sorted({c for c in hand_cat.values() if c not in labels.CATEGORIES})
        correct = sum(a.category[pi].top == c for pi, c in hand_cat.items())
        lines.append(f"**Category accuracy: {correct}/{len(hand_cat)} = {correct / len(hand_cat):.1%}**")
        if unknown:
            lines.append(f"  - Warning: hand labels not in the category list always count as wrong: {', '.join(unknown)}")
        wrong = sorted(((a.category[pi].score, pi) for pi, c in hand_cat.items() if a.category[pi].top != c), reverse=True)
    else:
        lines.append("Category accuracy: not measured (no `category` values filled in).")
    lines.append("")

    # --- Cluster purity ---
    groups = pipeline.make_groups(a, threshold)
    minority = []
    if hand_group:
        scored = [[i for i in g if i in hand_group] for g in groups]
        scored = [g for g in scored if g]
        p_all, n_all = purity(scored, hand_group)
        multi = [g for g in scored if len(g) > 1]
        p_multi, n_multi = purity(multi, hand_group)
        lines.append(f"**Cluster purity: {p_all:.1%}** ({n_all} pieces in {len(scored)} predicted groups; "
                     f"you labeled {len(set(hand_group.values()))} distinct groups)")
        if multi:
            lines.append(f"  - Purity over groups of 2+ only: {p_multi:.1%} ({n_multi} pieces in {len(multi)} groups)")
        else:
            lines.append("  - No predicted group has 2+ labeled pieces, so purity is trivially 100%.")
        for g in multi:
            top = Counter(hand_group[i] for i in g).most_common(1)[0][0]
            minority += [(len(g), i, top) for i in g if hand_group[i] != top]
    else:
        lines.append("Cluster purity: not measured (no `group_id` values filled in).")
    lines.append("")

    # --- Worst mistakes ---
    lines.append("## Ten worst mistakes")
    lines.append("")
    shown = 0
    for score, pi in wrong[:10]:
        name, n = keys[pi]
        lines.append(f"- `{name}` piece {n}: predicted **{a.category[pi].top}** ({score:.0%} sure), you said **{hand_cat[pi]}**")
        shown += 1
    for size, pi, top in sorted(minority, reverse=True)[: 10 - shown]:
        name, n = keys[pi]
        lines.append(f"- `{name}` piece {n}: grouped with {size - 1} others that are mostly `{top}`, you said `{hand_group[pi]}`")
        shown += 1
    if not shown:
        lines.append("- None found.")
    if unmatched:
        lines += ["", f"Note: {unmatched} rows in the labels file matched no detected clothing piece and were skipped."]

    text = "\n".join(lines) + "\n"
    print(text)
    out.write_text(text)
    print(f"Saved to {out}", file=sys.stderr)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Evaluate Cosmo against hand labels.")
    ap.add_argument("folder", nargs="?", type=Path, help="folder of images to score (e.g. data/)")
    ap.add_argument("--make-template", type=Path, metavar="FOLDER", help="write a blank labels.csv for this folder")
    ap.add_argument("--labels", type=Path, default=Path("labels.csv"))
    ap.add_argument("--out", type=Path, default=Path("eval_results.md"))
    ap.add_argument("--crops", type=Path, default=Path("eval_crops"))
    ap.add_argument("--threshold", type=float, default=config.DEFAULT_DISTANCE_THRESHOLD)
    ap.add_argument("--force", action="store_true", help="overwrite an existing labels.csv")
    args = ap.parse_args()
    if args.make_template:
        make_template(args.make_template, args.labels, args.crops, args.force)
    elif args.folder:
        evaluate(args.folder, args.labels, args.out, args.threshold)
    else:
        ap.print_help()
