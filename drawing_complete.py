"""
drawing_complete.py
Franka Emika Panda arm draws a picture on paper in MuJoCo.

Before the simulation starts the processed draw-points are rendered to a PNG
preview file (<input_stem>_preview.png) so you can check the output before
committing to the full robot run.

Two drawing styles are available via --style:

  lines  (default)
      Contour-traces image edges into connected strokes.  The pen moves
      continuously along each stroke; ink marks form a dense sphere chain.

  dots
      Samples dark pixels across the image.  The arm stamps a single sphere
      at each dot position for a pointillist look.

Resolution and mark size are exposed as CLI flags so you can tune the output
without editing source code.
"""

import argparse
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco as mj
import numpy as np
from mujoco import viewer

import RobotUtil as rt
from image_to_points import image_to_strokes, image_to_dot_points, save_preview


# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------
ROOT_MODEL_XML = "franka_emika_panda/panda_torque_table.xml"
SCENE_MODEL_XML = "franka_emika_panda/panda_torque_table_drawing.xml"

# ---------------------------------------------------------------------------
# Controller gains
# ---------------------------------------------------------------------------
KP = np.array([120, 120, 100, 90, 60, 40, 30], dtype=float)
KD = np.array([8,   8,   6,   5,  4,  3,  2],  dtype=float)

SEGMENT_DURATION = 1.5
HOLD_DURATION    = 0.15
PEN_HOLD_WIDTH   = 0.012

# ---------------------------------------------------------------------------
# Scene geometry
# ---------------------------------------------------------------------------
TABLE_CENTER    = np.array([0.55, 0.0, 0.05])
TABLE_HALF_SIZE = np.array([0.30, 0.40, 0.05])
TABLE_TOP_Z     = TABLE_CENTER[2] + TABLE_HALF_SIZE[2]   # 0.10 m

PAPER_HALF_XY = 0.13
PAPER_HALF_H  = 0.001
PAPER_CENTER  = np.array([0.55, 0.0, TABLE_TOP_Z + PAPER_HALF_H])
PAPER_TOP_Z   = PAPER_CENTER[2] + PAPER_HALF_H           # ≈ 0.102 m

# Pen-tip-site targets (not grip_site)
DRAW_Z = PAPER_TOP_Z
LIFT_Z = DRAW_Z + 0.08

DRAW_AREA   = 0.22
DRAW_ORIG_X = PAPER_CENTER[0] - DRAW_AREA / 2
DRAW_ORIG_Y = PAPER_CENTER[1] - DRAW_AREA / 2

PEN_UP_DIST = 0.07

# ---------------------------------------------------------------------------
# Robot configuration
# ---------------------------------------------------------------------------
Q_HOME = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.8], dtype=float)

DRAW_ROT = np.array([
    [1.0,  0.0,  0.0],
    [0.0, -1.0,  0.0],
    [0.0,  0.0, -1.0],
], dtype=float)

# ---------------------------------------------------------------------------
# Ink-mark defaults (overridden at runtime by --mark-size)
# ---------------------------------------------------------------------------
_DEFAULT_MARK_SIZE_LINE = 0.0018   # sphere radius for lines (m)
_DEFAULT_MARK_SIZE_DOT  = 0.0045   # sphere radius for dots  (m)
MARK_INTERVAL = 8                  # add one mark every N sim steps (lines mode)

_MARK_RGBA = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
_EYE3      = np.eye(3, dtype=np.float64).flatten()


def _add_ink_mark(scn, pos: np.ndarray, size: float) -> None:
    """Inject a black sphere into viewer.user_scn at *pos*."""
    if scn is None or scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mj.mjv_initGeom(
        g,
        mj.mjtGeom.mjGEOM_SPHERE,
        np.array([size, size, size], dtype=np.float64),
        pos.astype(np.float64),
        _EYE3,
        _MARK_RGBA,
    )
    scn.ngeom += 1


# ---------------------------------------------------------------------------
# XML scene builder
# ---------------------------------------------------------------------------
def _fmt(vals) -> str:
    return " ".join(f"{float(v):.6f}" for v in vals)


def build_scene_xml() -> str:
    """
    Extend the base Panda XML with drawing table, paper, and visual pen.

    Pen geometry in pen_body frame (pen_body at hand z=0.103):
      shaft      z = 0.000…0.044  (r=4 mm, half-len 0.022)
      ink tip    z = 0.044…0.052  (r=2 mm, half-len 0.004)
      pen_tip_site z = 0.052  ← IK target

    When pointing straight down (DRAW_ROT), pen_tip_site is 0.052 m below
    pen_body origin = 0.155 m below the hand body origin in world –Z.
    """
    tree = ET.parse(ROOT_MODEL_XML)
    root = tree.getroot()
    wb   = root.find("worldbody")

    # table (visual-only — no arm-link collisions)
    tb = ET.SubElement(wb, "body", {"name": "drawing_table", "pos": _fmt(TABLE_CENTER)})
    ET.SubElement(tb, "geom", {
        "type": "box", "size": _fmt(TABLE_HALF_SIZE),
        "rgba": "0.73 0.73 0.77 1.0",
        "contype": "0", "conaffinity": "0",
    })

    # paper
    pb = ET.SubElement(wb, "body", {"name": "paper", "pos": _fmt(PAPER_CENTER)})
    ET.SubElement(pb, "geom", {
        "type": "box",
        "size": f"{PAPER_HALF_XY:.6f} {PAPER_HALF_XY:.6f} {PAPER_HALF_H:.6f}",
        "rgba": "1.0 1.0 1.0 1.0",
        "contype": "0", "conaffinity": "0",
    })

    # pen on hand
    hand = root.find(".//body[@name='hand']")
    if hand is not None:
        pen = ET.SubElement(hand, "body", {"name": "pen", "pos": "0 0 0.103"})
        ET.SubElement(pen, "geom", {
            "name": "pen_shaft", "type": "cylinder",
            "size": "0.004 0.022", "pos": "0 0 0.022",
            "rgba": "0.20 0.12 0.04 1",
            "contype": "0", "conaffinity": "0", "group": "2",
        })
        ET.SubElement(pen, "geom", {
            "name": "pen_ink_tip", "type": "cylinder",
            "size": "0.002 0.004", "pos": "0 0 0.048",
            "rgba": "0.0 0.0 0.0 1",
            "contype": "0", "conaffinity": "0", "group": "2",
        })
        ET.SubElement(pen, "site", {
            "name": "pen_tip_site", "pos": "0 0 0.052",
            "size": "0.002", "rgba": "1 0 0 0",
        })

    tree.write(SCENE_MODEL_XML, encoding="utf-8", xml_declaration=True)
    return SCENE_MODEL_XML


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------
def norm_to_world(nx: float, ny: float) -> np.ndarray:
    wx = DRAW_ORIG_X + (1.0 - ny) * DRAW_AREA
    wy = DRAW_ORIG_Y + nx * DRAW_AREA
    return np.array([wx, wy, DRAW_Z])


def _group_points(points: np.ndarray) -> list[np.ndarray]:
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
# IK solver
# ---------------------------------------------------------------------------
def _set_qpos(model, data, q, arm_idx):
    data.qpos[arm_idx] = q
    data.qvel[arm_idx] = 0.0
    mj.mj_forward(model, data)


def solve_ik(
    model, data, site_id, q_seed, target_p, arm_idx,
    max_iters=200, tol=1e-4, damping=1e-2, step=0.5,
):
    q = q_seed.copy()
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    for _ in range(max_iters):
        _set_qpos(model, data, q, arm_idx)
        cur_p = data.site_xpos[site_id].copy()
        cur_R = data.site_xmat[site_id].reshape(3, 3).copy()
        ep = target_p - cur_p
        eR = 0.5 * (
            np.cross(cur_R[:, 0], DRAW_ROT[:, 0]) +
            np.cross(cur_R[:, 1], DRAW_ROT[:, 1]) +
            np.cross(cur_R[:, 2], DRAW_ROT[:, 2])
        )
        err = np.hstack([ep, eR])
        if np.linalg.norm(err) < tol:
            return q
        mj.mj_jacSite(model, data, jacp, jacr, site_id)
        J  = np.vstack([jacp[:, arm_idx], jacr[:, arm_idx]])
        JT = J.T
        dq = JT @ np.linalg.solve(J @ JT + damping * np.eye(6), err)
        q  = q + step * dq
        for i, jidx in enumerate(arm_idx):
            q[i] = np.clip(q[i], *model.jnt_range[jidx])
    return q


# ---------------------------------------------------------------------------
# Segment executor
# ---------------------------------------------------------------------------
def execute_segment(
    model, data, arm_idx, gripper_idx,
    q_start, q_goal, duration, v=None,
    hold_secs=HOLD_DURATION,
    mark_every=0,        # >0 → stamp ink sphere every N sim steps
    mark_scn=None,
    mark_site_id=None,
    mark_size=_DEFAULT_MARK_SIZE_LINE,
):
    dt      = model.opt.timestep
    n_steps = max(1, int(duration / dt))
    n_hold  = max(0, int(hold_secs / dt))
    t       = 0.0
    for step in range(n_steps + n_hold):
        q_des, qd_des = rt.interp_min_jerk(q_start, q_goal, t, duration)
        q  = data.qpos[arm_idx].copy()
        qd = data.qvel[arm_idx].copy()
        tau = KP * (q_des - q) + KD * (qd_des - qd)
        data.ctrl[arm_idx]     = tau + data.qfrc_bias[:7]
        data.ctrl[gripper_idx] = PEN_HOLD_WIDTH
        mj.mj_step(model, data)
        if mark_every > 0 and mark_scn is not None and mark_site_id is not None:
            if step % mark_every == 0:
                _add_ink_mark(mark_scn, data.site_xpos[mark_site_id].copy(), mark_size)
        if v is not None:
            v.sync()
        t += dt


# ---------------------------------------------------------------------------
# Drawing routines
# ---------------------------------------------------------------------------
def _settle(model, data, arm_idx, gripper_idx, v):
    for _ in range(400):
        q  = data.qpos[arm_idx].copy()
        qd = data.qvel[arm_idx].copy()
        tau = KP * (Q_HOME - q) - KD * qd
        data.ctrl[arm_idx]     = tau + data.qfrc_bias[:7]
        data.ctrl[gripper_idx] = PEN_HOLD_WIDTH
        mj.mj_step(model, data)
        if v is not None:
            v.sync()


def _draw_lines(model, data, arm_idx, gripper_idx, site_id,
                strokes, seg, v, mark_size):
    scn   = v.user_scn if v is not None else None
    q_cur = Q_HOME.copy()

    for si, stroke in enumerate(strokes):
        print(f"[sim] stroke {si+1}/{len(strokes)}  ({len(stroke)} pts)", flush=True)
        for pi, pt in enumerate(stroke):
            wp = norm_to_world(pt[0], pt[1])
            if pi == 0:
                hover = wp.copy(); hover[2] = LIFT_Z
                q_hov = solve_ik(model, data, site_id, q_cur, hover, arm_idx)
                execute_segment(model, data, arm_idx, gripper_idx,
                                q_cur, q_hov, seg, v)
                q_draw = solve_ik(model, data, site_id, q_hov, wp, arm_idx)
                execute_segment(model, data, arm_idx, gripper_idx,
                                q_hov, q_draw, seg * 0.4, v,
                                mark_every=MARK_INTERVAL,
                                mark_scn=scn, mark_site_id=site_id,
                                mark_size=mark_size)
                q_cur = q_draw
            else:
                prev = norm_to_world(stroke[pi-1][0], stroke[pi-1][1])
                dist = float(np.linalg.norm(wp[:2] - prev[:2]))
                dur  = float(np.clip(dist / 0.08, 0.15, seg * 0.5))
                q_draw = solve_ik(model, data, site_id, q_cur, wp, arm_idx)
                execute_segment(model, data, arm_idx, gripper_idx,
                                q_cur, q_draw, dur, v,
                                hold_secs=0.0,
                                mark_every=MARK_INTERVAL,
                                mark_scn=scn, mark_site_id=site_id,
                                mark_size=mark_size)
                q_cur = q_draw

        last = norm_to_world(stroke[-1][0], stroke[-1][1])
        lift = last.copy(); lift[2] = LIFT_Z
        q_lift = solve_ik(model, data, site_id, q_cur, lift, arm_idx)
        execute_segment(model, data, arm_idx, gripper_idx,
                        q_cur, q_lift, seg * 0.4, v)
        q_cur = q_lift

    return q_cur


def _draw_dots(model, data, arm_idx, gripper_idx, site_id,
               dot_points, seg, v, mark_size):
    scn    = v.user_scn if v is not None else None
    groups = _group_points(dot_points)
    q_cur  = Q_HOME.copy()

    for gi, group in enumerate(groups):
        print(f"[sim] dot group {gi+1}/{len(groups)}  ({len(group)} dots)", flush=True)
        for pi, pt in enumerate(group):
            wp = norm_to_world(pt[0], pt[1])
            if pi == 0:
                hover = wp.copy(); hover[2] = LIFT_Z
                q_hov = solve_ik(model, data, site_id, q_cur, hover, arm_idx)
                execute_segment(model, data, arm_idx, gripper_idx,
                                q_cur, q_hov, seg, v)
                q_draw = solve_ik(model, data, site_id, q_hov, wp, arm_idx)
                execute_segment(model, data, arm_idx, gripper_idx,
                                q_hov, q_draw, seg * 0.4, v,
                                hold_secs=0.2)
            else:
                prev = norm_to_world(group[pi-1][0], group[pi-1][1])
                dist = float(np.linalg.norm(wp[:2] - prev[:2]))
                dur  = float(np.clip(dist / 0.08, 0.15, seg * 0.5))
                q_draw = solve_ik(model, data, site_id, q_cur, wp, arm_idx)
                execute_segment(model, data, arm_idx, gripper_idx,
                                q_cur, q_draw, dur, v,
                                hold_secs=0.15)

            # stamp the actual pen-tip position after each touch
            if scn is not None and site_id is not None:
                _add_ink_mark(scn, data.site_xpos[site_id].copy(), mark_size)
            q_cur = q_draw

        last = norm_to_world(group[-1][0], group[-1][1])
        lift = last.copy(); lift[2] = LIFT_Z
        q_lift = solve_ik(model, data, site_id, q_cur, lift, arm_idx)
        execute_segment(model, data, arm_idx, gripper_idx,
                        q_cur, q_lift, seg * 0.4, v)
        q_cur = q_lift

    return q_cur


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------
def run_drawing(
    image_path: str,
    style: str = "lines",
    max_points: int = 400,
    threshold: int = 128,
    invert: bool = False,
    resolution: int = 150,
    mark_size_scale: float = 1.0,
    viewer_enabled: bool = True,
    speed: float = 1.0,
) -> None:
    """
    Parameters
    ----------
    resolution      : int
        Resize the input image so its longer side ≤ this value before
        processing.  Higher → finer detail, more points, slower IK.
        Default 150.  Recommended range 50–400.
    mark_size_scale : float
        Multiplier applied to the default ink-sphere radius.
        1.0 = default, 0.5 = thinner marks, 2.0 = thicker marks.
    """
    mark_size = (
        _DEFAULT_MARK_SIZE_LINE if style == "lines" else _DEFAULT_MARK_SIZE_DOT
    ) * mark_size_scale

    # ---- image → draw-points -------------------------------------------
    print(f"[image] loading '{image_path}'  style={style}  resolution={resolution}")
    if style == "lines":
        strokes, img_shape = image_to_strokes(
            image_path, max_points=max_points, threshold=threshold,
            invert=invert, resolution=resolution)
        total = sum(len(s) for s in strokes)
        print(f"[image] shape={img_shape}  {len(strokes)} strokes  {total} pts")
        if total == 0:
            print("[image] No draw-points found — check --threshold / --invert.")
            return
        draw_data = strokes
    else:
        dot_points, img_shape = image_to_dot_points(
            image_path, max_points=max_points, threshold=threshold,
            invert=invert, resolution=resolution)
        print(f"[image] shape={img_shape}  {len(dot_points)} dot positions")
        if len(dot_points) == 0:
            print("[image] No dark pixels found — check --threshold / --invert.")
            return
        draw_data = dot_points

    # ---- save preview image BEFORE simulation --------------------------
    stem         = Path(image_path).stem
    preview_path = f"{stem}_preview.png"
    save_preview(draw_data, style, preview_path)   # prints its own status line

    # ---- MuJoCo scene --------------------------------------------------
    scene_xml   = build_scene_xml()
    model       = mj.MjModel.from_xml_path(scene_xml)
    data        = mj.MjData(model)
    arm_idx     = list(range(7))
    gripper_idx = 7
    site_id     = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, "pen_tip_site")

    data.qpos[arm_idx] = Q_HOME
    data.qvel[arm_idx] = 0.0
    data.ctrl[gripper_idx] = PEN_HOLD_WIDTH
    mj.mj_forward(model, data)

    seg = SEGMENT_DURATION / speed

    def execute(v=None):
        print("[sim] settling at home …")
        _settle(model, data, arm_idx, gripper_idx, v)

        if style == "lines":
            q_end = _draw_lines(model, data, arm_idx, gripper_idx,
                                site_id, strokes, seg, v, mark_size)
        else:
            q_end = _draw_dots(model, data, arm_idx, gripper_idx,
                               site_id, dot_points, seg, v, mark_size)

        print("[sim] returning home …")
        execute_segment(model, data, arm_idx, gripper_idx,
                        q_end, Q_HOME, seg * 2.0, v)
        print("[sim] drawing complete!")

    if viewer_enabled:
        with viewer.launch_passive(model, data) as v:
            v.cam.lookat[:]  = [0.50, 0.0, 0.20]
            v.cam.distance   = 1.5
            v.cam.azimuth    = 140
            v.cam.elevation  = -28
            v.user_scn.ngeom = 0
            v.sync()
            execute(v)
            time.sleep(2.0)
    else:
        execute(None)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Franka Panda draws a picture from an input image in MuJoCo.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Resolution and mark-size guide
-------------------------------
--resolution N   (default 150)
  The input image is resized so its longer side is at most N pixels before
  edge-detection / pixel sampling.  Increasing N raises detail but also raises
  point count and IK time.

    50  → very coarse sketch (~50–150 pts)
   150  → default, good balance
   300  → fine detail, noticeably slower
   500  → very fine, use --max-points 800+ to benefit

--mark-size F   (default 1.0)
  Scales the radius of ink spheres in the MuJoCo viewer.

    0.3 → hairline marks
    1.0 → default
    2.0 → bold strokes / large dots
    3.0 → very fat marks
""",
    )
    p.add_argument("image",
                   help="Input image path (PNG, JPG, BMP, …).")
    p.add_argument("--style", choices=["lines", "dots"], default="lines",
                   help="'lines' traces contour strokes; 'dots' stamps point marks "
                        "(default: lines).")
    p.add_argument("--max-points", type=int, default=400,
                   help="Max draw-points extracted from the image (default: 400).")
    p.add_argument("--threshold", type=int, default=128,
                   help="Grayscale binarization threshold 0–255 (default: 128). "
                        "Pixels darker than this value are drawn.")
    p.add_argument("--invert", action="store_true",
                   help="Invert the binary mask (draw light pixels instead of dark).")
    p.add_argument("--resolution", type=int, default=150,
                   help="Resize input image so its longer side ≤ this value before "
                        "processing (default: 150).  Higher = finer detail, more points.")
    p.add_argument("--mark-size", type=float, default=1.0, dest="mark_size",
                   help="Scale factor for ink-sphere radius in the viewer "
                        "(default: 1.0).  0.5 = thinner, 2.0 = thicker.")
    p.add_argument("--no-viewer", action="store_true",
                   help="Run headless (no MuJoCo viewer window).")
    p.add_argument("--speed", type=float, default=1.0,
                   help="Playback speed multiplier (default: 1.0).")
    return p.parse_args()


def main():
    args = parse_args()
    run_drawing(
        image_path      = args.image,
        style           = args.style,
        max_points      = args.max_points,
        threshold       = args.threshold,
        invert          = args.invert,
        resolution      = args.resolution,
        mark_size_scale = args.mark_size,
        viewer_enabled  = not args.no_viewer,
        speed           = args.speed,
    )


if __name__ == "__main__":
    main()
