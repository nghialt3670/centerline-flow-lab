"""Extract an ordered open curve from a single Gradio ImageEditor brush stroke.

Gradio exports a raster alpha layer, not stroke timestamps. A simple connected
open stroke can be recovered by skeletonizing and tracing its longest path.
Branches / disconnected multi-strokes cannot be faithfully ordered; warn or fail.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
from PIL import Image


def resample_polyline(points: np.ndarray, count: int = 64) -> np.ndarray:
    p = np.asarray(points, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 2 or len(p) < 2:
        raise ValueError("Expected at least 2 ordered (x,y) points")
    delta = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(delta)])
    keep = np.concatenate([[True], np.diff(s) > 1e-7])
    p, s = p[keep], s[keep]
    if len(p) < 2 or s[-1] < 3:
        raise ValueError("Scribble is too short to form an open contour")
    u = np.linspace(0, s[-1], count)
    return np.column_stack([np.interp(u, s, p[:, 0]), np.interp(u, s, p[:, 1])])


def ordered_stroke(stroke_mask: np.ndarray, count: int = 64) -> tuple[np.ndarray, str]:
    """Return resampled x,y points and warnings; supports one long open curve."""
    mask = np.asarray(stroke_mask, dtype=bool)
    if mask.sum() < 8:
        raise ValueError("No drawn scribble found. Draw a visible open stroke over the image.")
    skel = skeletonize(mask)
    lbl, n = ndi.label(skel, structure=np.ones((3, 3), np.uint8))
    if n == 0:
        raise ValueError("Could not obtain a scribble skeleton")
    lengths = np.bincount(lbl.ravel())[1:]
    component = int(np.argmax(lengths) + 1)
    ys, xs = np.where(lbl == component)
    if len(xs) < 8:
        raise ValueError("Scribble is too short. Draw a longer open curve.")
    g = nx.Graph()
    occupied = set(zip(ys.tolist(), xs.tolist()))
    for y, x in occupied:
        g.add_node((y, x))
        for dy, dx in ((-1, -1), (-1, 0), (-1, 1), (0, -1)):
            q = (y + dy, x + dx)
            if q in occupied:
                g.add_edge((y, x), q, weight=float(np.hypot(dx, dy)))
    ends = [v for v, d in g.degree if d == 1]
    warning = ""
    if n > 1:
        warning += f"{n} disconnected strokes detected; using largest component. "
    if len(ends) != 2:
        warning += f"Found {len(ends)} endpoints (branches/cycle possible); using longest skeleton path. "
    # Endpoints are typically just two. Cap pathological branch counts.
    candidates = ends[:32] if len(ends) >= 2 else list(g.nodes)[::max(1, len(g) // 24)]
    best_len, best_path = -1.0, None
    for start in candidates:
        distances, paths = nx.single_source_dijkstra(g, start, weight="weight")
        targets = ends if ends else distances.keys()
        for target in targets:
            if target in distances and distances[target] > best_len:
                best_len, best_path = distances[target], paths[target]
    if best_path is None:
        raise ValueError("Unable to trace the scribble")
    xy = np.asarray([(float(x), float(y)) for y, x in best_path])
    return resample_polyline(xy, count=count), warning.strip()


def mask_from_editor(editor: dict) -> tuple[Image.Image, np.ndarray]:
    """Extract background + raster brush alpha. Never use composite as the source."""
    if not editor or editor.get("background") is None:
        raise ValueError("Upload a source image to the editor first")
    background = editor["background"]
    if not isinstance(background, Image.Image):
        background = Image.fromarray(np.asarray(background))
    background = background.convert("RGB")
    w, h = background.size
    stroke = np.zeros((h, w), bool)
    for layer in (editor.get("layers") or []):
        if layer is None:
            continue
        if not isinstance(layer, Image.Image):
            layer = Image.fromarray(np.asarray(layer))
        if layer.size != (w, h):
            layer = layer.resize((w, h), Image.Resampling.NEAREST)
        rgba = np.asarray(layer.convert("RGBA"))
        a = rgba[:, :, 3]
        # Gradio brush layers have a transparent background. Avoid an opaque
        # uploaded image layer being accidentally interpreted as a scribble.
        if float((a > 16).mean()) > 0.5:
            continue
        stroke |= a > 16
    if not stroke.any():
        raise ValueError("No brush stroke found. Draw on top of the uploaded image using the brush tool.")
    return background, stroke


def resize_input(image: Image.Image, stroke: np.ndarray, max_edge: int) -> tuple[Image.Image, np.ndarray]:
    w, h = image.size
    scale = min(1., float(max_edge) / max(w, h))
    # Diffusers FLUX.2 requires both dimensions divisible by 32.
    nw = max(64, int(round(w * scale / 32)) * 32)
    nh = max(64, int(round(h * scale / 32)) * 32)
    out = image.resize((nw, nh), Image.Resampling.LANCZOS)
    mask = Image.fromarray(stroke.astype(np.uint8) * 255).resize((nw, nh), Image.Resampling.NEAREST)
    return out, (np.asarray(mask) > 0)


def insertion_mask(stroke: np.ndarray, padding_px: int) -> Image.Image:
    """Broad edit rectangle around stroke; NOT the target object silhouette."""
    ys, xs = np.where(stroke)
    if len(xs) == 0:
        raise ValueError("No stroke available for edit region")
    h, w = stroke.shape
    x0, x1 = max(0, int(xs.min()) - padding_px), min(w, int(xs.max()) + padding_px + 1)
    y0, y1 = max(0, int(ys.min()) - padding_px), min(h, int(ys.max()) + padding_px + 1)
    out = np.zeros((h, w), np.uint8)
    out[y0:y1, x0:x1] = 255
    return Image.fromarray(out, mode="L")
