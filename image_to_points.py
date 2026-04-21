"""
image_to_points.py
Convert an input image into draw-point data for the robot dots drawing style,
and render a preview PNG showing exactly what will be drawn.

Public API
----------
image_to_dot_points(...)  – dots mode: spread-out sampled dark pixels
save_preview(...)         – render draw-points to a PNG for inspection
"""

import numpy as np
from PIL import Image, ImageDraw

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
    dot_points: np.ndarray,
    output_path: str,
    canvas_px: int = 512,
    dot_radius: int = 4,
) -> None:
    """
    Render the dot draw-points onto a white PNG canvas and save it.

    Parameters
    ----------
    dot_points  : (N, 2) float array of normalised draw-points
    output_path : where to write the PNG
    canvas_px   : side length of the square preview image (default 512)
    dot_radius  : dot radius in pixels (default 4)
    """
    img  = Image.new("RGB", (canvas_px, canvas_px), "white")
    draw = ImageDraw.Draw(img)
    N    = canvas_px - 1

    for pt in dot_points:
        px = int(float(pt[0]) * N)
        py = int(float(pt[1]) * N)
        r  = max(1, dot_radius)
        draw.ellipse([px - r, py - r, px + r, py + r], fill="black")

    img.save(output_path)
    print(f"[preview] saved → {output_path}  ({canvas_px}×{canvas_px} px)")
