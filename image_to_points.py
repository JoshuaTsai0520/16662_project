"""
image_to_points.py
Convert an input image into draw-point data for two robot drawing styles.

  image_to_strokes()   – lines mode
      Traces connected edge-pixels into ordered strokes using 8-connectivity.
      Each stroke is a continuous pen-down movement.

  image_to_dot_points() – dots mode
      Samples dark pixels spread across the image and orders them with a
      greedy nearest-neighbor pass to minimise total pen travel.
"""

import numpy as np
from PIL import Image, ImageFilter

MAX_IMG_DIM = 150   # resize input before processing to keep edge counts manageable


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _load_binary(image_path: str, threshold: int = 128, invert: bool = False) -> np.ndarray:
    """Load image, resize if large, return boolean mask (True = draw here)."""
    img = Image.open(image_path).convert("L")
    w, h = img.size
    if max(w, h) > MAX_IMG_DIM:
        scale = MAX_IMG_DIM / max(w, h)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        img = img.resize((nw, nh), Image.LANCZOS)
    arr = np.array(img, dtype=np.uint8)
    binary = arr < threshold
    if invert:
        binary = ~binary
    return binary


def _edge_mask(binary: np.ndarray) -> np.ndarray:
    """Thin binary mask to edge pixels only (PIL FIND_EDGES)."""
    pil = Image.fromarray(binary.astype(np.uint8) * 255, mode="L")
    edges = pil.filter(ImageFilter.FIND_EDGES)
    return np.array(edges, dtype=np.uint8) > 32


# ---------------------------------------------------------------------------
# Lines-mode helpers
# ---------------------------------------------------------------------------

def _trace_contours(mask: np.ndarray) -> list[np.ndarray]:
    """
    Trace connected components in *mask* into ordered stroke pixel sequences.

    Uses 8-connectivity.  At each step the tracer picks the unvisited neighbor
    that best continues the current travel direction, keeping curves smooth.

    Returns a list of (K, 2) float arrays, each row (x, y) in pixels.
    """
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return []

    remaining: set[tuple[int, int]] = set(zip(xs.tolist(), ys.tolist()))
    strokes: list[np.ndarray] = []

    while remaining:
        start = next(iter(remaining))
        remaining.discard(start)
        stroke: list[tuple[int, int]] = [start]

        while True:
            cx, cy = stroke[-1]
            nbrs = [
                (cx + dx, cy + dy)
                for dx in (-1, 0, 1)
                for dy in (-1, 0, 1)
                if (dx != 0 or dy != 0) and (cx + dx, cy + dy) in remaining
            ]
            if not nbrs:
                break
            if len(stroke) >= 2 and len(nbrs) > 1:
                px, py = stroke[-2]
                dir_x, dir_y = cx - px, cy - py
                nbrs.sort(
                    key=lambda nb: (nb[0] - cx - dir_x) ** 2 + (nb[1] - cy - dir_y) ** 2
                )
            nxt = nbrs[0]
            remaining.discard(nxt)
            stroke.append(nxt)

        strokes.append(np.array(stroke, dtype=float))

    return strokes


def _subsample_strokes(strokes: list[np.ndarray], max_points: int) -> list[np.ndarray]:
    """Uniformly thin each stroke so total points ≤ max_points."""
    total = sum(len(s) for s in strokes)
    if total <= max_points:
        return strokes
    ratio = max_points / total
    result = []
    for s in strokes:
        k = max(2, int(len(s) * ratio))
        if k >= len(s):
            result.append(s)
        else:
            idx = np.round(np.linspace(0, len(s) - 1, k)).astype(int)
            result.append(s[idx])
    return result


# ---------------------------------------------------------------------------
# Dots-mode helpers
# ---------------------------------------------------------------------------

def _nearest_neighbor_order(points: np.ndarray) -> np.ndarray:
    """Greedy nearest-neighbor reordering — O(n²), fine for n ≤ 1000."""
    n = len(points)
    if n <= 1:
        return points
    visited = np.zeros(n, dtype=bool)
    order   = np.empty(n, dtype=int)
    order[0] = 0
    visited[0] = True
    for i in range(1, n):
        last = points[order[i - 1]]
        d = np.sum((points - last) ** 2, axis=1)
        d[visited] = np.inf
        order[i] = int(np.argmin(d))
        visited[order[i]] = True
    return points[order]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def image_to_strokes(
    image_path: str,
    max_points: int = 500,
    threshold: int = 128,
    invert: bool = False,
) -> tuple[list[np.ndarray], tuple[int, int]]:
    """
    Lines mode: image → list of normalized contour strokes.

    Each stroke is a (K, 2) ndarray of (x_norm, y_norm) ∈ [0, 1]² that
    should be drawn as one continuous pen-down movement.
    x_norm=0 → left edge; y_norm=0 → top edge.

    Returns (strokes, (height, width)) of the resized working image.
    """
    binary    = _load_binary(image_path, threshold, invert)
    h, w      = binary.shape
    stroke_px = _trace_contours(_edge_mask(binary))

    if not stroke_px:
        return [], (h, w)

    stroke_px = _subsample_strokes(stroke_px, max_points)

    norm: list[np.ndarray] = []
    for s in stroke_px:
        ns = s.copy()
        ns[:, 0] /= max(w - 1, 1)
        ns[:, 1] /= max(h - 1, 1)
        norm.append(ns)

    return norm, (h, w)


def image_to_dot_points(
    image_path: str,
    max_points: int = 400,
    threshold: int = 128,
    invert: bool = False,
) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Dots mode: image → flat (N, 2) array of normalized draw-points.

    Samples dark pixels spread across the full image (not just edges),
    sub-samples to max_points, then reorders with nearest-neighbor to
    minimise pen travel between dots.

    Returns (points, (height, width)) of the resized working image.
    """
    binary = _load_binary(image_path, threshold, invert)
    h, w   = binary.shape

    ys, xs = np.where(binary)
    if len(xs) == 0:
        return np.zeros((0, 2), dtype=float), (h, w)

    points = np.column_stack([xs.astype(float), ys.astype(float)])

    if len(points) > max_points:
        idx    = np.round(np.linspace(0, len(points) - 1, max_points)).astype(int)
        points = points[idx]

    points = _nearest_neighbor_order(points)

    points[:, 0] /= max(w - 1, 1)
    points[:, 1] /= max(h - 1, 1)
    return points, (h, w)


def image_to_draw_points(
    image_path: str,
    max_points: int = 500,
    threshold: int = 128,
    invert: bool = False,
    **_kwargs,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Compatibility shim — returns flat point array using strokes pipeline."""
    strokes, shape = image_to_strokes(image_path, max_points, threshold, invert)
    if not strokes:
        return np.zeros((0, 2), dtype=float), shape
    return np.vstack(strokes), shape
