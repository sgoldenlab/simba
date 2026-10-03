# -*- coding: utf-8 -*-
"""
Lay out the published-studies concept word cloud inside a coronal mouse-brain section.

The section has five areas -- cortex, hippocampus, thalamus, hypothalamus, amygdala. Each word is placed
entirely inside one area and is colored by it, so the layout traces the anatomy. Brain-region terms are
placed in their own area (e.g. "Amygdala" in the amygdala); structures outside this section (e.g. "Cerebellum")
only go in the cortex filler, never inside another structure.

The packing needs numpy / scipy / matplotlib / Pillow, which the docs CI job (stdlib only) does not have.
So the layout is computed locally, together with the corpus refresh (misc/extract_corpus_stats.py), and stored
as plain coordinates under "brain_cloud" in misc/corpus_stats.json; misc/usecase_map_stats.py only draws it.

Text is measured with the docs' own Poppins Bold (docs/_static/fonts), the font the page renders the words in.

Run:  python misc/brain_cloud_layout.py   (re-lays out the cloud in the existing misc/corpus_stats.json)
"""
import json
import os
import sys

import numpy as np
from matplotlib.path import Path
from PIL import ImageFont
from scipy.ndimage import binary_erosion

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from usecase_map_stats import CLOUD_EXCLUDE

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CORPUS_PATH = os.path.join(ROOT, "misc", "corpus_stats.json")
FONT_PATH = os.path.join(ROOT, "docs", "_static", "fonts", "Poppins-Bold.ttf")

W, H = 1000, 700      # layout canvas
CELL = 2              # occupancy grid resolution (px per cell)
PAD = 1               # free space kept around each word (px)
MID = W / 2
MIN_SIZE = 8.0        # smallest font size; the cloud renders at roughly 1:1, so this is about 8 px on the page

AREAS = ("Cortex", "Hippocampus", "Thalamus", "Hypothalamus", "Amygdala")

# Corpus terms that name a structure in this section are placed in that structure.
REGION_TERMS = {"Cortex": ["Prefrontal cortex", "Cingulate", "Insula", "Prelimbic / infralimbic", "Orbitofrontal cortex", "Entorhinal cortex"],
                "Hippocampus": ["Hippocampus", "Dentate gyrus"],
                "Thalamus": ["Thalamus", "Habenula"],
                "Hypothalamus": ["Hypothalamus", "Paraventricular nucleus", "Preoptic area"],
                "Amygdala": ["Amygdala", "Basolateral amygdala"]}
OFF_SECTION_TERMS = {"Striatum", "Nucleus accumbens", "VTA", "BNST", "Cerebellum", "Periaqueductal gray", "Dorsal raphe",
                     "Substantia nigra", "Locus coeruleus", "Lateral septum"}

# Right half of the section outline, dorsal midline to ventral midline; the left half is its mirror.
OUTLINE_RIGHT = [(510, 92),
                 ((560, 58), (790, 70), (880, 210)),      # dorsal cortex
                 ((935, 300), (942, 420), (900, 500)),    # lateral cortex
                 ((862, 568), (790, 616), (705, 612)),    # piriform / ventrolateral
                 ((640, 610), (596, 592), (566, 612)),    # ventral surface
                 ((548, 640), (528, 664), (500, 664))]    # hypothalamus, ventral midline


def _bezier(start, segments, n=40):
    pts, p0 = [start], np.array(start, dtype=float)
    for c1, c2, p3 in segments:
        t = np.linspace(0, 1, n)[1:, None]
        c1, c2, p3 = map(np.array, (c1, c2, p3))
        pts += list((1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * c1 + 3 * (1 - t) * t ** 2 * c2 + t ** 3 * p3)
        p0 = p3
    return np.array(pts)


def _mirror(pts):
    return np.column_stack([2 * MID - pts[:, 0], pts[:, 1]])


def _ellipse(cx, cy, rx, ry, n=80):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([cx + rx * np.cos(t), cy + ry * np.sin(t)])


def _hippocampus(n=60):
    """Right hippocampus: a dorsal arc curving laterally and down, as in a coronal section."""
    t = np.linspace(0, 1, n)
    outer = np.column_stack([530 + 270 * np.sin(t * 2.0), 300 - 95 * np.cos(t * 2.0) + 20 * t])
    inner = np.column_stack([535 + 205 * np.sin(t * 2.0), 300 - 42 * np.cos(t * 2.0) + 30 * t])
    return np.vstack([outer, inner[::-1]])


def _shapes():
    right = _bezier(OUTLINE_RIGHT[0], OUTLINE_RIGHT[1:])
    outline = np.vstack([right, _mirror(right)[::-1][1:]])
    hippo, thal, amyg = _hippocampus(), _ellipse(590, 420, 82, 78), _ellipse(775, 540, 78, 46)
    areas = {"Hippocampus": [hippo, _mirror(hippo)],
             "Thalamus": [thal, _mirror(thal)],
             "Amygdala": [amyg, _mirror(amyg)],
             "Hypothalamus": [_ellipse(MID, 585, 72, 66)]}
    return outline, areas


def _rasterize(polys):
    ys, xs = np.mgrid[0:H:CELL, 0:W:CELL]
    centres = np.column_stack([xs.ravel() + CELL / 2, ys.ravel() + CELL / 2])
    mask = np.zeros(ys.shape, dtype=bool)
    for poly in polys:
        mask |= Path(poly).contains_points(centres).reshape(ys.shape)
    return mask


def _area_masks(outline, areas):
    """One mask per area (cortex = rest of the section), inset so words never straddle a boundary."""
    brain = _rasterize([outline])
    raw = {name: _rasterize(polys) & brain for name, polys in areas.items()}
    raw["Cortex"] = brain & ~np.any(list(raw.values()), axis=0)
    k = np.ones((2 * int(np.ceil(5 / CELL)) + 1,) * 2, dtype=bool)
    return {name: binary_erosion(m, structure=k) for name, m in raw.items()}


def _free_positions(blocked, cw, ch):
    S = np.pad(blocked.astype(np.int32).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    rect = S[ch:, cw:] - S[:-ch, cw:] - S[ch:, :-cw] + S[:-ch, :-cw]   # blocked cells under each placement
    return np.nonzero(rect == 0)


def _svg_path(pts):
    """Polygon as SVG path data -- one JSON string instead of hundreds of indented numbers."""
    return "M" + "L".join(f"{x:.1f},{y:.1f}" for x, y in pts) + "Z"


def _word_hash(word):
    return sum((i + 1) * ord(ch) for i, ch in enumerate(word))  # stable across runs and Python versions


def compute_layout(wordcloud):
    """
    :param wordcloud: [[term, n_papers], ...] as stored under "wordcloud" in corpus_stats.json.
    :return: dict with the section outline, the area polygons and the placed words, as plain JSON-able coordinates.
    """
    words = [(w, n) for w, n in wordcloud if w.lower() not in CLOUD_EXCLUDE]
    outline, areas = _shapes()
    masks = _area_masks(outline, areas)
    occupied = np.zeros(next(iter(masks.values())).shape, dtype=bool)
    term_area = {t: a for a, terms in REGION_TERMS.items() for t in terms}
    centroids = {a: tuple(np.mean(np.nonzero(m), axis=1)) for a, m in masks.items()}
    cy, cx = np.mean(np.nonzero(np.any(list(masks.values()), axis=0)), axis=1)
    lo, hi = min(n for _, n in words), max(n for _, n in words)
    placed = []
    # region terms first so their own area still has room, then everything else largest-first
    for word, n in sorted(words, key=lambda kv: (kv[0] not in term_area, -kv[1])):
        norm = ((n - lo) / (hi - lo)) ** 0.6 if hi > lo else 0.5
        size = MIN_SIZE + 2 + 28 * norm
        vertical = word not in term_area and norm < 0.4 and _word_hash(word) % 3 == 0  # some small words fill gaps vertically
        candidates = [term_area[word]] if word in term_area else ["Cortex"] if word in OFF_SECTION_TERMS else list(masks)
        while size >= MIN_SIZE:
            x0, y0, x1, y1 = ImageFont.truetype(FONT_PATH, int(round(size))).getbbox(word)
            bw, bh = (y1 - y0, x1 - x0) if vertical else (x1 - x0, y1 - y0)
            cw, ch = int(np.ceil((bw + 2 * PAD) / CELL)), int(np.ceil((bh + 2 * PAD) / CELL))
            best = None
            for area in candidates:
                fy, fx = _free_positions(~masks[area] | occupied, cw, ch)
                if not fy.size:
                    continue
                ty, tx = centroids[area] if word in term_area else (cy, cx)
                dist = ((fx + cw / 2 - tx) * 0.6) ** 2 + (fy + ch / 2 - ty) ** 2   # the section is wide: spread along x
                i = int(np.argmin(dist))
                if best is None or dist[i] < best[0]:
                    best = (dist[i], area, fy[i], fx[i])
            if best is not None:
                _, area, gy, gx = best
                occupied[gy:gy + ch, gx:gx + cw] = True
                bx, by = gx * CELL + PAD, gy * CELL + PAD
                # text anchor so the glyph box lands on (bx, by) with dominant-baseline: text-before-edge;
                # vertical words are rotated -90 deg about the anchor and read bottom-to-top
                ax, ay = (bx - y0, by + bh + x0) if vertical else (bx - x0, by - y0)
                placed.append([word, n, AREAS.index(area), int(round(size)), int(vertical), round(float(ax), 1), round(float(ay), 1)])
                break
            size *= 0.9
    xs, ys = outline[:, 0], outline[:, 1]
    return {"viewbox": [round(float(xs.min() - 10)), round(float(ys.min() - 10)), round(float(np.ptp(xs) + 20)), round(float(np.ptp(ys) + 20))],
            "outline": _svg_path(outline),
            "areas": {name: " ".join(_svg_path(p) for p in polys) for name, polys in areas.items()},
            "area_names": list(AREAS),
            "words": placed,   # [term, n_papers, area index, font size, vertical (0/1), anchor x, anchor y]
            "n_words": len(words)}


def main():
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        corpus = json.load(f)
    corpus["brain_cloud"] = compute_layout(corpus["wordcloud"])
    with open(CORPUS_PATH, "w", encoding="utf-8") as f:
        json.dump(corpus, f, indent=1)
    bc = corpus["brain_cloud"]
    print(f"[brain cloud] placed {len(bc['words'])} of {bc['n_words']} words -> {CORPUS_PATH}")


if __name__ == "__main__":
    main()
