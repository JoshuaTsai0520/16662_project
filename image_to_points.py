"""
image_to_points.py
Convert an input image into draw-point data for two robot drawing styles,
and render a preview PNG showing exactly what will be drawn.

Public API
----------
image_to_strokes(...)     – lines mode: connected contour strokes
image_to_dot_points(...)  – dots mode: spread-out sampled dark pixels
save_preview(...)         – render draw-points to a PNG for inspection
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

DEFAULT_RESOLUTION = 300   # default max image dimension before processing


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _load_binary(
    image_path: str,
    threshold: int = 128,
    invert: bool = False,
    resolution: int = DEFAULT_RESOLUTION,
) -> np.ndarray:
    """Load image, resize so the longer side ≤ resolution, return bool mask."""
    img = Image.open(image_path).convert("L")
    w, h = img.size
    if resolution and max(w, h) > resolution:
        scale = resolution / max(w, h)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        img = img.resize((nw, nh), Image.LANCZOS)
    arr = np.array(img, dtype=np.uint8)
    binary = arr < threshold
    if invert:
        binary = ~binary
    return binary


def _edge_mask(binary: np.ndarray) -> np.ndarray:
    pil = Image.fromarray(binary.astype(np.uint8) * 255, mode="L")
    return np.array(pil.filter(ImageFilter.FIND_EDGES), dtype=np.uint8) > 32


# ---------------------------------------------------------------------------
# Lines-mode helpers
# ---------------------------------------------------------------------------

def _trace_contours(mask: np.ndarray) -> list[np.ndarray]:
    """
    Trace connected edge-pixels into ordered stroke sequences (8-connectivity).
    At each step picks the neighbor that best continues the current direction.
    Returns list of (K, 2) float arrays with (x, y) pixel coords.
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
    out = []
    for s in strokes:
        k = max(2, int(len(s) * ratio))
        if k >= len(s):
            out.append(s)
        else:
            idx = np.round(np.linspace(0, len(s) - 1, k)).astype(int)
            out.append(s[idx])
    return out


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
# Public API — image → draw-points
# ---------------------------------------------------------------------------

def image_to_strokes(
    image_path: str,
    max_points: int = 500,
    threshold: int = 128,
    invert: bool = False,
    resolution: int = DEFAULT_RESOLUTION,
) -> tuple[list[np.ndarray], tuple[int, int]]:
    """
    Lines mode: image → list of normalised contour strokes.

    Parameters
    ----------
    resolution : int
        Resize the input image so its longer side is at most this many pixels
        before edge-detection.  Higher values → finer strokes, more points,
        slower IK.  Default 150.

    Returns
    -------
    strokes   : list of (K, 2) float arrays, each row (x_norm, y_norm) ∈ [0,1]²
    img_shape : (height, width) of the resized working image
    """
    binary    = _load_binary(image_path, threshold, invert, resolution)
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
    resolution: int = DEFAULT_RESOLUTION,
) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Dots mode: image → flat (N, 2) array of normalised draw-points.

    Parameters
    ----------
    resolution : int
        Resize the input image so its longer side is at most this many pixels
        before sampling.  Higher values → denser/finer dot grid.  Default 150.

    Returns
    -------
    points    : (N, 2) float array, each row (x_norm, y_norm) ∈ [0,1]²
    img_shape : (height, width) of the resized working image
    """
    binary = _load_binary(image_path, threshold, invert, resolution)
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


# ---------------------------------------------------------------------------
# Preview renderer
# ---------------------------------------------------------------------------

def save_preview(
    draw_data,              # list[ndarray] for lines, ndarray for dots
    style: str,             # "lines" or "dots"
    output_path: str,
    canvas_px: int = 512,   # preview image side length in pixels
    line_width: int = 2,    # pixel width of strokes (lines mode)
    dot_radius: int = 4,    # pixel radius of dots   (dots mode)
) -> None:
    """
    Render the draw-points onto a white PNG canvas and save it.

    The canvas uses the same (x_norm, y_norm) coordinate system as the
    normalised draw-points: x left→right, y top→bottom.  The result shows
    exactly what the robot arm will draw on the paper.

    Parameters
    ----------
    draw_data   : strokes (list of arrays) for lines mode,
                  flat point array for dots mode.
    style       : "lines" or "dots"
    output_path : where to write the PNG
    canvas_px   : side length of the square preview image (default 512)
    line_width  : stroke width in pixels, lines mode only
    dot_radius  : dot radius in pixels, dots mode only
    """
    img  = Image.new("RGB", (canvas_px, canvas_px), "white")
    draw = ImageDraw.Draw(img)
    N    = canvas_px - 1

    def to_px(nx: float, ny: float) -> tuple[int, int]:
        return (int(nx * N), int(ny * N))

    if style == "lines":
        for stroke in draw_data:
            if len(stroke) == 0:
                continue
            if len(stroke) == 1:
                px, py = to_px(stroke[0][0], stroke[0][1])
                r = max(1, line_width)
                draw.ellipse([px - r, py - r, px + r, py + r], fill="black")
            else:
                pts = [to_px(pt[0], pt[1]) for pt in stroke]
                draw.line(pts, fill="black", width=max(1, line_width))
    else:  # dots
        for pt in draw_data:
            px, py = to_px(float(pt[0]), float(pt[1]))
            r = max(1, dot_radius)
            draw.ellipse([px - r, py - r, px + r, py + r], fill="black")

    img.save(output_path)
    print(f"[preview] saved → {output_path}  ({canvas_px}×{canvas_px} px)")


# ---------------------------------------------------------------------------
# Compatibility shim
# ---------------------------------------------------------------------------

def image_to_draw_points(
    image_path: str,
    max_points: int = 800,
    threshold: int = 128,
    invert: bool = False,
    resolution: int = DEFAULT_RESOLUTION,
    **_kwargs,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Compatibility shim — returns flat point array using strokes pipeline."""
    strokes, shape = image_to_strokes(
        image_path, max_points, threshold, invert, resolution)
    if not strokes:
        return np.zeros((0, 2), dtype=float), shape
    return np.vstack(strokes), shape
