"""The style map: one bubble per group, laid out like an outfit.

Rows run top to bottom in the order you wear things (outerwear, tops,
bottoms, shoes, accessories). Bubble area is proportional to the number of
pieces in the group. A line joins two groups that appear in the same saved
image; thicker lines mean more images.
"""
from __future__ import annotations

import base64
import io
import math

import plotly.graph_objects as go
from PIL import Image, ImageDraw

from . import labels

ROWS = ["outerwear", "tops", "dresses and skirts", "bottoms", "shoes", "accessories"]
MAX_ROW_WIDTH = 21.0   # layout units before a row wraps; keeps the map under ~1200 px wide
PX_PER_UNIT = 46       # used to turn layout units into chart height
LINE = "rgba(128,128,128,0.55)"
ACCENT = "#e4572e"     # ring around the bubble that is open


def _disc(img: Image.Image, px: int = 180) -> str:
    """Crop on a white disc with a grey ring, as a PNG data URI."""
    big = px * 2  # draw at 2x then shrink, for smooth edges
    canvas = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    ImageDraw.Draw(canvas).ellipse([4, 4, big - 5, big - 5], fill="white", outline="#8a8a8a", width=6)
    t = img.convert("RGB")
    side = int(big * 0.66)  # fits inside the circle's inscribed square
    t.thumbnail((side, side))
    canvas.paste(t, ((big - t.width) // 2, (big - t.height) // 2))
    canvas = canvas.resize((px, px), Image.LANCZOS)
    buf = io.BytesIO()
    canvas.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _wrap(name: str) -> str:
    if len(name) <= 10 or " " not in name:
        return name
    words = name.split(" ")
    best = min(range(1, len(words)), key=lambda k: abs(len(" ".join(words[:k])) - len(name) / 2))
    return " ".join(words[:best]) + "<br>" + " ".join(words[best:])


def layout(sizes: dict[int, int], kinds: dict[int, str]) -> tuple[dict[int, tuple[float, float, float]], list[tuple[str, float]]]:
    """Place bubbles. Returns {group: (x, y, radius)} and [(row label, y)]."""
    pos: dict[int, tuple[float, float, float]] = {}
    row_labels: list[tuple[str, float]] = []
    y = 0.0
    for kind in ROWS:
        members = sorted((g for g in sizes if kinds[g] == kind), key=lambda g: -sizes[g])
        if not members:
            continue
        lines: list[list[tuple[int, float, float]]] = [[]]
        x = 0.0
        for g in members:
            r = 0.62 * math.sqrt(sizes[g])
            if lines[-1]:
                prev_r = lines[-1][-1][2]
                x += max(prev_r + r + 0.7, 3.0)   # leave room for the text label
                if x + r > MAX_ROW_WIDTH:
                    lines.append([])
                    x = 0.0
            lines[-1].append((g, x, r))
        first_y = None
        for line in lines:
            top_r = max(r for _, _, r in line)
            y -= top_r
            width = line[-1][1]
            for g, gx, r in line:
                pos[g] = (gx - width / 2, y, r)
            first_y = y if first_y is None else first_y
            y -= top_r + 1.9                      # label space under the bubbles
        row_labels.append((kind, first_y))
    return pos, row_labels


def style_map(groups: list[list[int]], names: list[str], cats: list[str], thumbs: list,
              reps: dict[int, int], pairs: list[tuple[tuple[int, int], int]],
              show: list[int], min_pair: int = 2, disc_cache: dict | None = None) -> go.Figure:
    """groups/names are indexed by group id; `show` is the group ids to draw.

    The figure does not depend on which bubble is open. That matters: if a
    click changed the figure, the browser would rebuild the chart, and a
    second click arriving during the rebuild would be lost. The highlight
    ring is instead drawn by Plotly itself (see the marker trace below).
    `disc_cache` (piece index -> image) avoids re-encoding thumbnails."""
    disc_cache = {} if disc_cache is None else disc_cache
    sizes = {g: len(groups[g]) for g in show}
    kinds = {g: labels.KIND.get(cats[groups[g][0]], "accessories") for g in show}
    pos, row_labels = layout(sizes, kinds)
    fig = go.Figure()
    if not pos:
        return fig

    # Lines first (drawn below the bubbles). They are plain shapes with no
    # hover or click points of their own, so a click can only ever land on a bubble.
    for (g1, g2), n in pairs:
        if n < min_pair or g1 not in pos or g2 not in pos:
            continue
        (x1, y1, _), (x2, y2, _) = pos[g1], pos[g2]
        fig.add_shape(type="line", x0=x1, y0=y1, x1=x2, y1=y2, layer="below",
                      line=dict(color=LINE, width=1 + 1.6 * n))
    ids = list(pos)
    for g in ids:
        x, y, r = pos[g]
        if reps[g] not in disc_cache:
            disc_cache[reps[g]] = _disc(thumbs[reps[g]])
        fig.add_layout_image(source=disc_cache[reps[g]], x=x, y=y, sizex=2 * r, sizey=2 * r,
                             xref="x", yref="y", xanchor="center", yanchor="middle", layer="above")
        n = sizes[g]
        fig.add_annotation(x=x, y=y - r - 0.15, yanchor="top", showarrow=False, align="center",
                           text=f"<b>{_wrap(names[g])}</b><br>{n} piece{'s' if n != 1 else ''}",
                           font=dict(size=12))
    # One marker per bubble: it carries the hover text and receives clicks.
    # The marker is an open circle slightly larger than the bubble, invisible
    # (opacity 0) until Plotly marks it as selected, when it shows as the
    # accent-colored ring. Each marker carries its group's representative
    # piece index, which stays valid when the slider regroups things.
    # Sizes are in pixels and match the bubbles when the chart is drawn at
    # PX_PER_UNIT; MAX_ROW_WIDTH keeps the chart narrow enough for that.
    fig.add_trace(go.Scatter(
        x=[pos[g][0] for g in ids], y=[pos[g][1] for g in ids], mode="markers",
        marker=dict(symbol="circle-open", color=ACCENT, opacity=0, line=dict(width=4),
                    size=[2 * pos[g][2] * PX_PER_UNIT + 8 for g in ids]),
        selected=dict(marker=dict(opacity=1)), unselected=dict(marker=dict(opacity=0)),
        customdata=[reps[g] for g in ids], hoverinfo="text", showlegend=False,
        hovertext=[f"{names[g]}: {sizes[g]} piece{'s' if sizes[g] != 1 else ''} (click to open)" for g in ids],
    ))

    left = min(x - r for x, _, r in pos.values()) - 4.2
    right = max(x + r for x, _, r in pos.values()) + 1.5
    bottom = min(y - r for _, y, r in pos.values()) - 2.0
    for kind, ry in row_labels:
        fig.add_annotation(x=left + 0.2, y=ry, xanchor="left", showarrow=False,
                           text=kind.upper(), font=dict(size=11), opacity=0.7)
    fig.update_xaxes(visible=False, range=[left, right], fixedrange=True)
    fig.update_yaxes(visible=False, range=[bottom, 0.6], fixedrange=True, scaleanchor="x")
    fig.update_layout(height=int((0.6 - bottom) * PX_PER_UNIT), margin=dict(l=0, r=0, t=0, b=0),
                      dragmode=False, clickmode="event+select", hovermode="closest", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                      hoverlabel=dict(font_size=13))
    return fig
