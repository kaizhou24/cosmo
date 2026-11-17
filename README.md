# Cosmo

Cosmo turns the clothing screenshots you already save into a picture of your style. Upload a ZIP of screenshots and it groups near-identical items, so "14 cropped black jackets" becomes visible at a glance.

## Try it

**Hosted demo:** https://cosmo-style.streamlit.app/

Click **Try the sample set** to see results on 20 outfit images without uploading anything. The hosted demo runs on a small free server: the sample set is instant, while your own ZIP takes several seconds per image and is capped at 60 images there. For larger sets, run it locally.

## Run it locally

Requires Python 3.10+ (tested on 3.11).

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The first upload downloads two models (about 1.5 GB in total): `Marqo/marqo-fashionSigLIP` for labels and similarity, and `google/owlv2-base-patch16-ensemble` for finding garments. Everything runs locally: no paid APIs, no accounts. Uploaded images are held in memory for the session and never written to disk; only numeric vectors and box coordinates are cached in `.cache/`.

## What works today

- Safe ZIP upload (path-traversal, symlink, size and file-count guards), with corrupt and non-image files skipped
- Near-duplicate removal with perceptual hashing
- **One item per garment, not per image:** an open-vocabulary detector (OWLv2) finds each piece in a screenshot or outfit flat-lay and crops it
- Zero-shot labels per piece (category, color, material/texture) and per image (aesthetic). Low-confidence labels are flagged, and group names only state what the model is confident about
- Grouping of similar pieces within each category, using visual similarity plus a color-difference penalty, with a strictness slider
- **Style map:** one bubble per group, laid out like an outfit, with lines between groups saved in the same image (what you save together, not a judgment of what looks good)
- A "How each image was labeled" tab showing every box, label, probability and runner-up
- `eval.py`, which scores category accuracy and cluster purity against hand labels

## Evaluation

**The eval has not been run yet; no accuracy numbers exist.**

```
python eval.py --make-template data/   # writes labels.csv (one row per piece) and eval_crops/
# fill in the category and group_id columns by hand
python eval.py data/                   # prints results, saves eval_results.md
```

## Known limits

- Labels come from zero-shot text matching, not a trained classifier. Aesthetic and material labels are rough; category and color are more dependable.
- The detector can miss small accessories or draw one box around two touching pieces. The eval only scores pieces it found.
- Pairing lines count co-occurrence in your saves. With few images most pairs appear once and are hidden.

## Tests

```
pytest
```

## Sample images

The images in `sample_data/` are outfit inspiration images saved from Pinterest. They belong to their original creators and are included only to demonstrate the tool. If one is yours and you would like it removed, open an issue.
