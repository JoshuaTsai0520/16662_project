"""
drawing_complete.py
Franka Emika Panda arm draws a picture on paper in MuJoCo.

Two drawing styles are supported via --style:

  lines  (default)
      Contour-traces the image edges into connected strokes.  The pen moves
      continuously along each stroke at paper height; ink marks are deposited
      as a dense chain of spheres in the viewer's user_scn, giving a solid
      line appearance.

  dots
      Samples dark pixels spread across the image.  The arm lifts the pen
      between groups of dots that are far apart; a single sphere is stamped
      at each commanded dot position, giving a pointillist appearance.

Ink marks are rendered by adding mjvGeom spheres to viewer.user_scn in
real-time as the arm draws — they appear as black marks on the white paper.
"""

import argparse
import time
import xml.etree.ElementTree as ET

import mujoco as mj
import numpy as np
from mujoco import viewer

import RobotUtil as rt
from image_to_points import image_to_strokes, image_to_dot_points


# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------
ROOT_MODEL_XML = "franka_emika_panda/panda_torque_table.xml"
SCENE_MODEL_XML = "franka_emika_panda/panda_torque_table_drawing.xml"

# ---------------------------------------------------------------------------
# Controller gains  (identical to hanoi project)
# ---------------------------------------------------------------------------
KP = np.array([120, 120, 100, 90, 60, 40, 30], dtype=float)
KD = np.array([8,   8,   6,   5,  4,  3,  2],  dtype=float)

SEGMENT_DURATION = 1.5    # default seconds per motion segment
HOLD_DURATION    = 0.15   # seconds to dwell at end of pen-up/lift segments
PEN_HOLD_WIDTH   = 0.012  # gripper width while holding pen (m)

# ---------------------------------------------------------------------------
# Scene geometry
# ---------------------------------------------------------------------------
TABLE_CENTER    = np.array([0.55, 0.0, 0.05])
TABLE_HALF_SIZE = np.array([0.30, 0.40, 0.05])
TABLE_TOP_Z     = TABLE_CENTER[2] + TABLE_HALF_SIZE[2]   # = 0.10 m

PAPER_HALF_XY = 0.13    # half-side of square paper → 26 cm
PAPER_HALF_H  = 0.001
PAPER_CENTER  = np.array([0.55, 0.0, TABLE_TOP_Z + PAPER_HALF_H])
PAPER_TOP_Z   = PAPER_CENTER[2] + PAPER_HALF_H           # ≈ 0.102 m

# Targets for the PEN TIP SITE (not grip_site)
DRAW_Z = PAPER_TOP_Z          # pen tip sits exactly on paper surface
LIFT_Z = DRAW_Z + 0.08        # pen tip height when travelling between strokes

DRAW_AREA   = 0.22            # usable drawing region side (m)
DRAW_ORIG_X = PAPER_CENTER[0] - DRAW_AREA / 2
DRAW_ORIG_Y = PAPER_CENTER[1] - DRAW_AREA / 2

# Pen-up distance threshold (normalised image coords) for grouping dots/lines
PEN_UP_DIST = 0.07

# ---------------------------------------------------------------------------
# Robot home pose and drawing orientation
# ---------------------------------------------------------------------------
Q_HOME = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.8], dtype=float)

DRAW_ROT = np.array([          # end-effector pointing straight down
    [1.0,  0.0,  0.0],
    [0.0, -1.0,  0.0],
    [0.0,  0.0, -1.0],
], dtype=float)

# ---------------------------------------------------------------------------
# Ink-mark rendering parameters
# ---------------------------------------------------------------------------
MARK_INTERVAL  = 8      # add one ink sphere every N sim steps during pen-down
MARK_SIZE_LINE = 0.0018  # sphere radius for lines mode (thin trail)
MARK_SIZE_DOT  = 0.0045  # sphere radius for dots mode (fatter stamp)
_MARK_RGBA = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
_EYE3      = np.eye(3, dtype=np.float64).flatten()


def _add_ink_mark(scn, pos: np.ndarray, size: float) -> None:
    """
    Inject one black sphere into the viewer's user scene at *pos*.
    Silently does nothing when the scene is full or unavailable.
    """
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
    Extend the base Panda XML with a drawing table, white paper, and a
    visual pen whose tip carries the IK target site (pen_tip_site).

    Pen geometry in pen_body-local frame (pen_body origin = hand z=0.103):
      shaft  : z = 0.000 … 0.044   (cylinder r=4 mm, half-len=0.022)
      ink tip: z = 0.044 … 0.052   (cylinder r=2 mm, half-len=0.004)
      pen_tip_site : z = 0.052      ← IK target, actual paper-contact point

    When the arm points straight down (DRAW_ROT), the hand's +z points in
    world –Z, so pen_tip_site sits 0.052 m below pen_body origin (= 0.155 m
    below the hand body origin), and 0.052 m below grip_site.
    """
    tree = ET.parse(ROOT_MODEL_XML)
    root = tree.getroot()
    worldbody = root.find("worldbody")

    # --- drawing table (visual-only — no collision with arm links) ---
    tb = ET.SubElement(worldbody, "body", {"name": "drawing_table", "pos": _fmt(TABLE_CENTER)})
    ET.SubElement(tb, "geom", {
        "type": "box", "size": _fmt(TABLE_HALF_SIZE),
        "rgba": "0.73 0.73 0.77 1.0",
        "contype": "0", "conaffinity": "0",
    })

    # --- paper (visual-only, white) ---
    pb = ET.SubElement(worldbody, "body", {"name": "paper", "pos": _fmt(PAPER_CENTER)})
    ET.SubElement(pb, "geom", {
        "type": "box",
        "size": f"{PAPER_HALF_XY:.6f} {PAPER_HALF_XY:.6f} {PAPER_HALF_H:.6f}",
        "rgba": "1.0 1.0 1.0 1.0",
        "contype": "0", "conaffinity": "0",
    })

    # --- pen attached to the robot hand ---
    hand = root.find(".//body[@name='hand']")
    if hand is not None:
        pen_body = ET.SubElement(hand, "body", {"name": "pen", "pos": "0 0 0.103"})
        ET.SubElement(pen_body, "geom", {   # wooden shaft
            "name": "pen_shaft", "type": "cylinder",
            "size": "0.004 0.022", "pos": "0 0 0.022",
            "rgba": "0.20 0.12 0.04 1",
            "contype": "0", "conaffinity": "0", "group": "2",
        })
        ET.SubElement(pen_body, "geom", {   # ink tip
            "name": "pen_ink_tip", "type": "cylinder",
            "size": "0.002 0.004", "pos": "0 0 0.048",
            "rgba": "0.0 0.0 0.0 1",
            "contype": "0", "conaffinity": "0", "group": "2",
        })
        # IK target site — exactly at the physical end of the pen
        ET.SubElement(pen_body, "site", {
            "name": "pen_tip_site", "pos": "0 0 0.052",
            "size": "0.002", "rgba": "1 0 0 0",   # invisible
        })

    tree.write(SCENE_MODEL_XML, encoding="utf-8", xml_declaration=True)
    return SCENE_MODEL_XML


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------
def norm_to_world(nx: float, ny: float) -> np.ndarray:
    """
    Normalised image (x, y) ∈ [0,1]² → 3-D world position at DRAW_Z.
    Image x left→right maps to world Y.
    Image y top→bottom maps to world X (reversed: top = far from robot).
    """
    wx = DRAW_ORIG_X + (1.0 - ny) * DRAW_AREA
    wy = DRAW_ORIG_Y + nx * DRAW_AREA
    return np.array([wx, wy, DRAW_Z])


def _group_points(points: np.ndarray) -> list[np.ndarray]:
    """Split a flat point array into pen-down strokes by distance gap."""
    if len(points) == 0:
        return []
    strokes: list[list] = [[points[0]]]
    for pt in points[1:]:
        if np.linalg.norm(pt - strokes[-1][-1]) > PEN_UP_DIST:
            strokes.append([pt])
        else:
            strokes[-1].append(pt)
    return [np.array(s) for s in strokes]


# ---------------------------------------------------------------------------
# IK solver
# ---------------------------------------------------------------------------
def _set_qpos(model, data, q, arm_idx):
    data.qpos[arm_idx] = q
    data.qvel[arm_idx] = 0.0
    mj.mj_forward(model, data)


def solve_ik(
    model, data, site_id, q_seed: np.ndarray, target_p: np.ndarray, arm_idx,
    max_iters: int = 200, tol: float = 1e-4, damping: float = 1e-2, step: float = 0.5,
) -> np.ndarray:
    """Damped least-squares IK targeting pen_tip_site at target_p with DRAW_ROT."""
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

    return q  # best-effort


# ---------------------------------------------------------------------------
# Segment executor
# ---------------------------------------------------------------------------
def execute_segment(
    model, data, arm_idx, gripper_idx,
    q_start: np.ndarray, q_goal: np.ndarray,
    duration: float,
    v=None,
    hold_secs: float = HOLD_DURATION,
    mark_every: int = 0,      # >0 → add an ink sphere every N sim steps
    mark_scn=None,            # viewer.user_scn
    mark_site_id=None,        # index of pen_tip_site
    mark_size: float = MARK_SIZE_LINE,
) -> None:
    """
    Execute one PD-controlled joint-space segment with min-jerk interpolation.

    mark_every > 0 enables real-time ink marks: every *mark_every* simulation
    steps the current pen_tip_site world position is stamped as a black sphere
    in *mark_scn*.  Set hold_secs=0 for smooth continuous pen-down motion.
    """
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
# Drawing routines (lines and dots share the same settle + home-return code)
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


def _draw_lines(model, data, arm_idx, gripper_idx, site_id, strokes, seg, v):
    """Execute lines-mode drawing: continuous strokes with dense ink trail."""
    scn = v.user_scn if v is not None else None
    q_cur = Q_HOME.copy()

    for si, stroke in enumerate(strokes):
        print(f"[sim] stroke {si + 1}/{len(strokes)}  ({len(stroke)} pts)", flush=True)

        for pi, pt in enumerate(stroke):
            wp = norm_to_world(pt[0], pt[1])

            if pi == 0:
                # Lift to hover, lower to paper
                hover = wp.copy(); hover[2] = LIFT_Z
                q_hov = solve_ik(model, data, site_id, q_cur, hover, arm_idx)
                execute_segment(model, data, arm_idx, gripper_idx,
                                q_cur, q_hov, seg, v)

                q_draw = solve_ik(model, data, site_id, q_hov, wp, arm_idx)
                execute_segment(model, data, arm_idx, gripper_idx,
                                q_hov, q_draw, seg * 0.4, v,
                                mark_every=MARK_INTERVAL, mark_scn=scn,
                                mark_site_id=site_id, mark_size=MARK_SIZE_LINE)
                q_cur = q_draw
            else:
                prev = norm_to_world(stroke[pi - 1][0], stroke[pi - 1][1])
                dist = float(np.linalg.norm(wp[:2] - prev[:2]))
                dur  = float(np.clip(dist / 0.08, 0.15, seg * 0.5))

                q_draw = solve_ik(model, data, site_id, q_cur, wp, arm_idx)
                # hold_secs=0: arm flows continuously (no per-point pause)
                # mark_every adds ink marks during the move
                execute_segment(model, data, arm_idx, gripper_idx,
                                q_cur, q_draw, dur, v,
                                hold_secs=0.0,
                                mark_every=MARK_INTERVAL, mark_scn=scn,
                                mark_site_id=site_id, mark_size=MARK_SIZE_LINE)
                q_cur = q_draw

        # Lift after stroke
        last = norm_to_world(stroke[-1][0], stroke[-1][1])
        lift = last.copy(); lift[2] = LIFT_Z
        q_lift = solve_ik(model, data, site_id, q_cur, lift, arm_idx)
        execute_segment(model, data, arm_idx, gripper_idx, q_cur, q_lift, seg * 0.4, v)
        q_cur = q_lift

    return q_cur


def _draw_dots(model, data, arm_idx, gripper_idx, site_id, dot_points, seg, v):
    """
    Execute dots-mode drawing.

    Consecutive dots closer than PEN_UP_DIST are grouped into one pen-down
    pass (the arm slides at paper height between them).  A single ink sphere
    is stamped at each commanded dot position.
    """
    scn    = v.user_scn if v is not None else None
    groups = _group_points(dot_points)
    q_cur  = Q_HOME.copy()

    for gi, group in enumerate(groups):
        print(f"[sim] dot group {gi + 1}/{len(groups)}  ({len(group)} dots)", flush=True)

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
                                hold_secs=0.2)   # brief dwell = visible dot
                # stamp ink mark at actual pen tip position
                if scn is not None and site_id is not None:
                    _add_ink_mark(scn, data.site_xpos[site_id].copy(), MARK_SIZE_DOT)
                q_cur = q_draw
            else:
                prev = norm_to_world(group[pi - 1][0], group[pi - 1][1])
                dist = float(np.linalg.norm(wp[:2] - prev[:2]))
                dur  = float(np.clip(dist / 0.08, 0.15, seg * 0.5))

                q_draw = solve_ik(model, data, site_id, q_cur, wp, arm_idx)
                execute_segment(model, data, arm_idx, gripper_idx,
                                q_cur, q_draw, dur, v,
                                hold_secs=0.15)  # slight dwell per dot
                if scn is not None and site_id is not None:
                    _add_ink_mark(scn, data.site_xpos[site_id].copy(), MARK_SIZE_DOT)
                q_cur = q_draw

        # Lift after group
        last = norm_to_world(group[-1][0], group[-1][1])
        lift = last.copy(); lift[2] = LIFT_Z
        q_lift = solve_ik(model, data, site_id, q_cur, lift, arm_idx)
        execute_segment(model, data, arm_idx, gripper_idx, q_cur, q_lift, seg * 0.4, v)
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
    viewer_enabled: bool = True,
    speed: float = 1.0,
) -> None:
    # ---- image processing -----------------------------------------------
    print(f"[image] loading '{image_path}'  style={style}")
    if style == "lines":
        strokes, img_shape = image_to_strokes(
            image_path, max_points=max_points, threshold=threshold, invert=invert)
        total = sum(len(s) for s in strokes)
        print(f"[image] shape={img_shape}  {len(strokes)} strokes  {total} pts")
        if total == 0:
            print("[image] No draw-points found — check --threshold / --invert.")
            return
    else:  # dots
        dot_points, img_shape = image_to_dot_points(
            image_path, max_points=max_points, threshold=threshold, invert=invert)
        print(f"[image] shape={img_shape}  {len(dot_points)} dot positions")
        if len(dot_points) == 0:
            print("[image] No dark pixels found — check --threshold / --invert.")
            return

    # ---- MuJoCo scene ---------------------------------------------------
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
                                site_id, strokes, seg, v)
        else:
            q_end = _draw_dots(model, data, arm_idx, gripper_idx,
                               site_id, dot_points, seg, v)

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
            v.user_scn.ngeom = 0   # clear any leftover custom geoms
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
        description="Franka Panda draws a picture from an input image in MuJoCo."
    )
    p.add_argument("image",
                   help="Path to the input image (PNG, JPG, BMP, …).")
    p.add_argument("--style", choices=["lines", "dots"], default="lines",
                   help="Drawing style: 'lines' traces contour strokes; "
                        "'dots' stamps individual point marks (default: lines).")
    p.add_argument("--max-points", type=int, default=400,
                   help="Max draw-points extracted from the image (default: 400).")
    p.add_argument("--threshold", type=int, default=128,
                   help="Grayscale binarization threshold 0–255 (default: 128). "
                        "Pixels darker than this value are drawn.")
    p.add_argument("--invert", action="store_true",
                   help="Invert the binary mask (draw light pixels instead of dark).")
    p.add_argument("--no-viewer", action="store_true",
                   help="Run headless (no MuJoCo viewer window).")
    p.add_argument("--speed", type=float, default=1.0,
                   help="Playback speed multiplier (e.g. 2.0 = twice as fast).")
    return p.parse_args()


def main():
    args = parse_args()
    run_drawing(
        image_path     = args.image,
        style          = args.style,
        max_points     = args.max_points,
        threshold      = args.threshold,
        invert         = args.invert,
        viewer_enabled = not args.no_viewer,
        speed          = args.speed,
    )


if __name__ == "__main__":
    main()
