"""
drawing_real.py
Franka Emika Panda arm draws a picture on a real robot using frankapy.

The robot draws in dots style: it stamps the pen at each sampled dark-pixel
position on the paper.

Before the robot moves the processed draw-points are rendered to a PNG preview
file (<input_stem>_preview.png) so you can verify the layout.

CALIBRATION REQUIRED before first use
--------------------------------------
Set the constants in the "Real-world configuration" section below to match
your physical setup:
  DRAW_Z    – measured Z height of the paper surface in robot world frame (m)
  PAPER_CENTER_XY – (x, y) of the paper center in robot world frame (m)
  DRAW_AREA – side length of the square drawing region on paper (m)
"""

import argparse
import time
from pathlib import Path

import numpy as np
from autolab_core import RigidTransform
from frankapy import FrankaArm

from image_to_points import image_to_dot_points, save_preview


# ---------------------------------------------------------------------------
# Real-world configuration  ← CALIBRATE THESE FOR YOUR SETUP
# ---------------------------------------------------------------------------

# Paper surface height in robot world frame (meters).
# Measure with the pen tip touching the paper.
DRAW_Z = 0.102

# Height to lift pen between dot groups (meters above DRAW_Z).
LIFT_OFFSET = 0.08
LIFT_Z = DRAW_Z + LIFT_OFFSET

# Center of the drawing area (x, y) in robot world frame (meters).
PAPER_CENTER_XY = np.array([0.55, 0.0])

# Side length of the square drawing region on the paper (meters).
DRAW_AREA = 0.22

# Gripper width to hold the pen (meters).
PEN_GRIP_WIDTH = 0.012

# Distance threshold for splitting dot sequence into separate lift groups (m).
PEN_UP_DIST = 0.07

# ---------------------------------------------------------------------------
# Derived draw-area origin
# ---------------------------------------------------------------------------
DRAW_ORIG_X = PAPER_CENTER_XY[0] - DRAW_AREA / 2
DRAW_ORIG_Y = PAPER_CENTER_XY[1] - DRAW_AREA / 2

# ---------------------------------------------------------------------------
# Pen-pointing-down rotation (end-effector frame → world frame).
# Adjust if your robot's neutral orientation differs.
# ---------------------------------------------------------------------------
DRAW_ROT = np.array([
    [1.0,  0.0,  0.0],
    [0.0, -1.0,  0.0],
    [0.0,  0.0, -1.0],
], dtype=float)

# ---------------------------------------------------------------------------
# Motion timing (seconds)
# ---------------------------------------------------------------------------
TRAVEL_DURATION = 2.0   # time to move between dot groups
LOWER_DURATION  = 1.0   # time to lower pen to paper
STAMP_HOLD      = 0.15  # seconds to hold pen on paper
LIFT_DURATION   = 0.8   # time to lift pen after stamping


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def norm_to_world(nx: float, ny: float, z: float) -> np.ndarray:
    """Convert normalised draw-point (nx, ny) ∈ [0,1]² to robot world (x,y,z)."""
    wx = DRAW_ORIG_X + (1.0 - ny) * DRAW_AREA
    wy = DRAW_ORIG_Y + nx * DRAW_AREA
    return np.array([wx, wy, z])


def _make_pose(position: np.ndarray, rotation: np.ndarray = DRAW_ROT) -> RigidTransform:
    return RigidTransform(
        rotation=rotation,
        translation=position,
        from_frame="franka_tool",
        to_frame="world",
    )


def _group_points(points: np.ndarray) -> list[np.ndarray]:
    """Split flat point array into groups separated by large jumps."""
    if len(points) == 0:
        return []
    groups: list[list] = [[points[0]]]
    for pt in points[1:]:
        if np.linalg.norm(pt - groups[-1][-1]) > PEN_UP_DIST:
            groups.append([pt])
        else:
            groups[-1].append(pt)
    return [np.array(g) for g in groups]


# ---------------------------------------------------------------------------
# Drawing routine
# ---------------------------------------------------------------------------

def _draw_dots_real(fa: FrankaArm, dot_points: np.ndarray) -> None:
    """
    Execute the dots drawing sequence on the real Franka arm.

    For each dot:
      1. Move (in Cartesian space) to hover position above the dot.
      2. Lower pen to paper surface.
      3. Hold briefly to stamp ink.
      4. Lift pen back to hover height.
    """
    groups = _group_points(dot_points)
    total_dots = sum(len(g) for g in groups)
    done = 0

    for gi, group in enumerate(groups):
        print(f"[robot] dot group {gi+1}/{len(groups)}  ({len(group)} dots)", flush=True)

        for pi, pt in enumerate(group):
            world_xy  = norm_to_world(pt[0], pt[1], DRAW_Z)
            hover_pos = norm_to_world(pt[0], pt[1], LIFT_Z)

            hover_pose = _make_pose(hover_pos)
            stamp_pose = _make_pose(world_xy)

            if pi == 0:
                # Travel to hover above first dot in this group
                fa.goto_pose(hover_pose, duration=TRAVEL_DURATION, use_impedance=False)

            # Lower to paper
            fa.goto_pose(stamp_pose, duration=LOWER_DURATION, use_impedance=False)
            time.sleep(STAMP_HOLD)

            # Lift back to hover
            fa.goto_pose(hover_pose, duration=LIFT_DURATION, use_impedance=False)

            done += 1
            if done % 10 == 0 or done == total_dots:
                print(f"[robot] {done}/{total_dots} dots complete", flush=True)


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def run_drawing_real(
    image_path: str,
    max_points: int = 400,
    threshold: int = 128,
    invert: bool = False,
    resolution: int = 150,
    dry_run: bool = False,
) -> None:
    """
    Parameters
    ----------
    image_path  : path to input image
    max_points  : max number of dots to extract from the image
    threshold   : grayscale binarisation threshold (0–255); pixels darker
                  than this are drawn
    invert      : draw light pixels instead of dark ones
    resolution  : resize image so its longer side ≤ this before sampling
    dry_run     : if True, only generate the preview PNG — do not move the robot
    """
    # ---- image → draw-points -------------------------------------------
    print(f"[image] loading '{image_path}'  resolution={resolution}")
    dot_points, img_shape = image_to_dot_points(
        image_path, max_points=max_points, threshold=threshold,
        invert=invert, resolution=resolution)
    print(f"[image] shape={img_shape}  {len(dot_points)} dot positions")
    if len(dot_points) == 0:
        print("[image] No dark pixels found — check --threshold / --invert.")
        return

    # ---- save preview image --------------------------------------------
    stem         = Path(image_path).stem
    preview_path = f"{stem}_preview.png"
    save_preview(dot_points, preview_path)

    if dry_run:
        print("[robot] dry-run mode — skipping robot motion.")
        return

    # ---- connect to robot ----------------------------------------------
    print("[robot] connecting to Franka arm …")
    fa = FrankaArm()

    # Hold pen in gripper
    print(f"[robot] closing gripper to hold pen (width={PEN_GRIP_WIDTH} m) …")
    fa.goto_gripper(PEN_GRIP_WIDTH)

    # Move to home joints
    print("[robot] moving to home configuration …")
    fa.reset_joints()

    # ---- draw ----------------------------------------------------------
    print("[robot] starting drawing …")
    _draw_dots_real(fa, dot_points)

    # ---- return home ---------------------------------------------------
    print("[robot] returning to home configuration …")
    fa.reset_joints()

    # Release pen
    fa.open_gripper()
    print("[robot] drawing complete!")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Franka Panda draws on real hardware using frankapy (dots style).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Calibration reminder
--------------------
Edit DRAW_Z, PAPER_CENTER_XY, and DRAW_AREA at the top of this file to match
your physical setup before running on the real robot.

Use --dry-run to generate and inspect the preview PNG without moving the robot.
""",
    )
    p.add_argument("image",
                   help="Input image path (PNG, JPG, BMP, …).")
    p.add_argument("--max-points", type=int, default=400,
                   help="Max draw-points extracted from the image (default: 400).")
    p.add_argument("--threshold", type=int, default=128,
                   help="Grayscale binarization threshold 0–255 (default: 128).")
    p.add_argument("--invert", action="store_true",
                   help="Invert the binary mask (draw light pixels instead of dark).")
    p.add_argument("--resolution", type=int, default=150,
                   help="Resize input image so its longer side ≤ this value (default: 150).")
    p.add_argument("--dry-run", action="store_true",
                   help="Generate preview PNG only — do not connect to or move the robot.")
    return p.parse_args()


def main():
    args = parse_args()
    run_drawing_real(
        image_path  = args.image,
        max_points  = args.max_points,
        threshold   = args.threshold,
        invert      = args.invert,
        resolution  = args.resolution,
        dry_run     = args.dry_run,
    )


if __name__ == "__main__":
    main()
