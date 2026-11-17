"""Cosmo Streamlit page. Run with: streamlit run app.py"""
from __future__ import annotations

import tempfile
from collections import Counter
from pathlib import Path

import sys

import pandas as pd
import streamlit as st
from PIL import ImageDraw

# Streamlit re-reads this file when it changes but keeps the `cosmo` package
# it imported at startup. If any file in cosmo/ is newer than the copy in
# memory, drop the package so the import below loads the current code.
CODE_STAMP = max(p.stat().st_mtime for p in (Path(__file__).parent / "cosmo").glob("*.py"))
_loaded = sys.modules.get("cosmo")
if _loaded is not None and getattr(_loaded, "_code_stamp", None) != CODE_STAMP:
    for _name in [m for m in sys.modules if m == "cosmo" or m.startswith("cosmo.")]:
        del sys.modules[_name]

import cosmo
from cosmo import config, ingest, labels, pipeline, profile, viz

cosmo._code_stamp = CODE_STAMP
# Results computed by older code may not match the current page: recompute
# them from the upload that is still attached.
if st.session_state.get("code_stamp") != CODE_STAMP:
    st.session_state["code_stamp"] = CODE_STAMP
    for _key in ("result", "upload_id", "map_pick", "style_map"):
        st.session_state.pop(_key, None)

st.set_page_config(page_title="Cosmo", layout="wide")
st.title("Cosmo")
st.write("Upload the clothing screenshots you've saved. Cosmo finds each piece, groups the near-identical ones, and shows what you keep reaching for.")

from cosmo import embed  # noqa: E402  (after the reload check above)

HOSTED = embed.low_memory()  # no GPU: this is the free hosted demo (or a CPU-only laptop)
if HOSTED:
    config.MAX_FILES = config.MAX_FILES_LOW_MEMORY

THUMB_PX = 256
GRID_COLS = 6
BOX_COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4", "#f032e6", "#9a6324"]


def annotate(item_image, pieces_here):
    """Copy of the image with a numbered box per detected piece."""
    im = item_image.copy()
    d = ImageDraw.Draw(im)
    w = max(2, im.width // 250)
    for n, (_, p) in enumerate(pieces_here, 1):
        c = BOX_COLORS[(n - 1) % len(BOX_COLORS)]
        d.rectangle(p.box, outline=c, width=w)
        tag = f" {n} "
        tb = d.textbbox((p.box[0], p.box[1]), tag, font_size=max(14, im.width // 40))
        d.rectangle(tb, fill=c)
        d.text((p.box[0], p.box[1]), tag, fill="white", font_size=max(14, im.width // 40))
    im.thumbnail((700, 700))
    return im


def run_pipeline(source: str, payload) -> None:
    """Uploads are extracted to a temp dir that is deleted as soon as the
    images are in memory; only embedding vectors and boxes are cached."""
    with st.status("Processing images...", expanded=True) as status:
        st.write("Reading images")
        if source == "zip":
            with tempfile.TemporaryDirectory(prefix="cosmo_") as tmp:
                res = ingest.safe_extract(payload, Path(tmp))
                ingest.load_images(Path(tmp), res)
        else:
            res = ingest.ingest_folder(payload)
        if not res.items:
            status.update(label="No usable images found", state="error")
            st.session_state.pop("result", None)
            return
        bar = st.progress(0.0)
        a = pipeline.analyze(res.items, step=st.write, progress=lambda d, t: bar.progress(d / t))
        thumbs = []
        for p in a.pieces:
            t = p.image.copy()
            t.thumbnail((THUMB_PX, THUMB_PX))
            thumbs.append(t)
        status.update(label="Done", state="complete", expanded=False)
        by_image: dict[int, list] = {}
        for pi, p in enumerate(a.pieces):
            by_image.setdefault(p.image_idx, []).append((pi, p))
        annotated = [annotate(it.image, by_image.get(i, [])) for i, it in enumerate(res.items)]
        # Everything the page shows is now a thumbnail; release the full-size
        # images so a session holds a few MB rather than hundreds.
        for obj in [*res.items, *a.pieces]:
            obj.image = None
    st.session_state["result"] = {"res": res, "a": a, "thumbs": thumbs, "by_image": by_image,
                                  "annotated": annotated, "discs": {}}
    # A click on the old map means nothing for new data: forget both our own
    # record of it and the chart widget's remembered selection.
    st.session_state.pop("map_pick", None)
    st.session_state.pop("style_map", None)


def show_grid(indices, thumbs, captions=None):
    cols = st.columns(GRID_COLS)
    for k, i in enumerate(indices):
        cols[k % GRID_COLS].image(thumbs[i], width="stretch", caption=captions[k] if captions else None)


# --- Input ------------------------------------------------------------------
if HOSTED:
    st.info(f"This is running on a small free server. **Try the sample set** is instant. Your own ZIP "
            f"works too but takes several seconds per image, and only the first {config.MAX_FILES} images are used.")
upload = st.file_uploader("ZIP of images", type=["zip"])
sample_available = config.SAMPLE_DIR.exists() and any(config.SAMPLE_DIR.iterdir())
if st.button("Try the sample set", disabled=not sample_available):
    run_pipeline("folder", config.SAMPLE_DIR)
if upload is not None and st.session_state.get("upload_id") != upload.file_id:
    st.session_state["upload_id"] = upload.file_id
    try:
        run_pipeline("zip", upload.getvalue())
    except Exception as e:  # a bad ZIP should never crash the page
        st.error(f"Could not read that ZIP: {type(e).__name__}")

result = st.session_state.get("result")
if not result:
    st.stop()

res, a, thumbs = result["res"], result["a"], result["thumbs"]
kept = a.kept
cats = [c.top for c in a.category]
colors = [c.top for c in a.color]

threshold = st.slider(
    "Grouping strictness (lower = tighter, more numerous groups)",
    0.05, 0.6, config.DEFAULT_DISTANCE_THRESHOLD, 0.01,
    help="Two pieces join a group when their distance is below this. Distance = how "
         "different the crops look to the model, plus a penalty when their colors differ.",
)
groups = pipeline.make_groups(a, threshold)  # fast: re-clusters without re-embedding
names = [profile.group_name(g, a) for g in groups]
group_of = {p: gi for gi, g in enumerate(groups) for p in g}
pairs = profile.pairings([p.image_idx for p in a.pieces], group_of, cats)
reps = {gi: profile.representative(g, a.piece_vecs) for gi, g in enumerate(groups)}

# --- Summary line -------------------------------------------------------------
not_clothing = [i for i in range(len(a.pieces)) if not a.clothing[i]]
notes = [f"{len(res.items) + res.duplicates} images"]
if res.duplicates:
    notes.append(f"{res.duplicates} duplicates removed")
if res.unreadable:
    notes.append(f"{res.unreadable} unreadable files skipped")
if res.skipped_heic:
    notes.append(f"{res.skipped_heic} HEIC files skipped (install pillow-heif to read them)")
if res.rejected_unsafe:
    notes.append(f"{res.rejected_unsafe} unsafe ZIP entries rejected")
if res.truncated:
    notes.append(f"stopped at the {config.MAX_FILES}-image / 300 MB limit")
notes.append(f"{len(not_clothing)} not clothing")
st.subheader(", ".join(notes) + f", {len(kept)} pieces in {len(groups)} groups")

tab_map, tab_style, tab_groups, tab_labels = st.tabs(
    ["Style map", "Style profile", "All groups", "How each image was labeled"]
)


def describe(pi: int) -> str:
    """Caption for one piece: only what the model is confident about."""
    words = [a.color[pi].top if a.color[pi].confident else None,
             profile.material_word(a.material[pi]), a.category[pi].top]
    return " ".join(w for w in words if w)


# --- Style map ----------------------------------------------------------------
@st.fragment
def style_map_section():
    """A fragment: clicking a bubble or a control in here re-runs only this
    function, not the whole page, so the map responds quickly and the other
    tabs are left alone."""
    st.write("Each bubble is a group of similar pieces; bigger means you saved more of them. "
             "A line joins groups that appear in the same saved image (thicker = more images). "
             "Click a bubble to open it.")
    c1, c2 = st.columns(2)
    one_offs = c1.toggle("Include one-offs", value=False)
    min_pair = c2.select_slider("Draw a line when saved together in at least", [1, 2, 3, 4, 5], value=2,
                                format_func=lambda n: f"{n} image{'s' if n > 1 else ''}")
    show = [gi for gi, g in enumerate(groups) if one_offs or len(g) > 1]
    if not show:
        st.info("No group has more than one piece yet. Turn on one-offs or loosen the strictness slider.")
        return
    fig = viz.style_map(groups, names, cats, thumbs, reps, pairs, show, min_pair, result["discs"])
    event = st.plotly_chart(fig, on_select="rerun", selection_mode="points", key="style_map",
                            config={"displayModeBar": False, "doubleClick": False})
    # Remember the last bubble clicked. Clicking the open bubble again sends an
    # empty selection; we ignore that so the group stays open.
    for pt in event.get("selection", {}).get("points", []):
        cd = pt.get("customdata")
        cd = cd[0] if isinstance(cd, (list, tuple)) else cd
        if cd is not None:
            st.session_state["map_pick"] = int(cd)
    g = group_of.get(st.session_state.get("map_pick"))
    if g is None or g not in show:
        st.caption("Lines show what you save together, not a judgment of what looks good.")
        return
    st.markdown(f"### {names[g]} ({len(groups[g])})")
    show_grid(groups[g], thumbs)
    partners = [(g2 if g1 == g else g1, n) for (g1, g2), n in pairs if g in (g1, g2)]
    if partners:
        st.markdown("**Saved together with**")
        cols = st.columns(GRID_COLS)
        for k, (other, n) in enumerate(partners[:GRID_COLS]):
            cols[k].image(thumbs[reps[other]], width="stretch",
                          caption=f"{names[other]}: {n} image{'s' if n > 1 else ''}")
    st.caption("Lines show what you save together, not a judgment of what looks good.")


with tab_map:
    style_map_section()

# --- Style profile ------------------------------------------------------------
with tab_style:
    top_pair = None
    if pairs:
        (g1, g2), n = pairs[0]
        top_pair = (names[g1], names[g2], n)
    biggest = (names[0], len(groups[0])) if groups else None
    for s in profile.sentences([cats[i] for i in kept], [colors[i] for i in kept], biggest, top_pair):
        st.markdown(f"- {s}")
    c1, c2 = st.columns(2)
    c1.markdown("**Category** (pieces)")
    c1.bar_chart(pd.Series(Counter(cats[i] for i in kept)).sort_values(ascending=False), horizontal=True)
    c2.markdown("**Color** (pieces)")
    c2.bar_chart(pd.Series(Counter(colors[i] for i in kept)).sort_values(ascending=False), horizontal=True)
    c3, c4 = st.columns(2)
    mat_counts = Counter(w for w in (profile.material_word(a.material[i]) for i in kept) if w)
    c3.markdown("**Material / texture** (pieces where the model is confident)")
    if mat_counts:
        c3.bar_chart(pd.Series(mat_counts).sort_values(ascending=False), horizontal=True)
        c3.caption(f"{len(kept) - sum(mat_counts.values())} of {len(kept)} pieces have no confident material and are left out.")
    c4.markdown("**Aesthetic** (whole images)")
    c4.bar_chart(pd.Series(Counter(x.top for x in a.aesthetic)).sort_values(ascending=False), horizontal=True)
    c4.caption("Aesthetic labels are rough; category and color are more dependable.")

# --- All groups ---------------------------------------------------------------
with tab_groups:
    multi = [(n, g) for n, g in zip(names, groups) if len(g) > 1]
    singles = [g[0] for g in groups if len(g) == 1]
    for name, g in multi:
        st.markdown(f"**{name} ({len(g)})**")
        show_grid(g, thumbs)
    if singles:
        with st.expander(f"One-offs ({len(singles)})"):
            show_grid(singles, thumbs, [describe(i) for i in singles])
    if not_clothing:
        with st.expander(f"Not clothing ({len(not_clothing)})"):
            show_grid(not_clothing, thumbs)
    low = [i for i in kept if a.category[i].low_confidence]
    if low:
        with st.expander(f"Low-confidence category ({len(low)})"):
            show_grid(low, thumbs, [f"{a.category[i].top} or {a.category[i].runner_up}?" for i in low])

# --- Labels inspector ----------------------------------------------------------
with tab_labels:
    st.write("Every detected piece and the label the model chose, with its probability "
             "and runner-up. Use this to see where the model goes wrong.")
    only_low = st.toggle("Only show images with a low-confidence label")
    by_image = result["by_image"]
    for idx, item in enumerate(res.items):
        here = by_image.get(idx, [])
        if only_low and not any(a.category[pi].low_confidence or a.color[pi].low_confidence for pi, _ in here):
            continue
        st.markdown(f"**{item.name}** · aesthetic: {a.aesthetic[idx]} (runner-up {a.aesthetic[idx].runner_up})")
        left, right = st.columns([1, 2])
        left.image(result["annotated"][idx], width="stretch")
        rows = []
        for n, (pi, p) in enumerate(here, 1):
            m = a.material[pi]
            if m is None:
                mat = "n/a"
            elif profile.material_word(m):
                mat = str(m)
            else:
                mat = f"not sure (best guess: {m.top} {m.score:.0%})"
            rows.append({
                "#": n,
                "category": str(a.category[pi]),
                "runner-up": f"{a.category[pi].runner_up} {a.category[pi].runner_score:.0%}",
                "color": str(a.color[pi]),
                "material": mat,
                "detector said": "whole image (no boxes)" if p.whole else f"{p.det_label} {p.det_score:.2f}",
                "clothing?": "yes" if a.clothing[pi] else "no",
                "group": names[group_of[pi]] if pi in group_of else "n/a",
            })
        right.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.divider()
