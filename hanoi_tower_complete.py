import argparse
import time
import xml.etree.ElementTree as ET

import mujoco as mj
import numpy as np
from mujoco import viewer

import RobotUtil as rt


ROOT_MODEL_XML = "franka_emika_panda/panda_torque_table.xml"
SCENE_MODEL_XML = "franka_emika_panda/panda_torque_table_hanoi_complete.xml"

ALL_BLOCK_NAMES = [chr(ord("A") + i) for i in range(26)]
MAX_BLOCK_COUNT = len(ALL_BLOCK_NAMES)
BLOCK_RANK = {name: idx for idx, name in enumerate(ALL_BLOCK_NAMES)}

KP = np.array([120, 120, 100, 90, 60, 40, 30], dtype=float)
KD = np.array([8, 8, 6, 5, 4, 3, 2], dtype=float)
SEGMENT_DURATION = 2.0
HOLD_DURATION = 0.5

OPEN_WIDTH = 0.04
CLOSED_WIDTH = 0.008

TABLE_CENTER = np.array([0.55, 0.0, 0.02], dtype=float)
TABLE_HALF_SIZE = np.array([0.34, 0.45, 0.02], dtype=float)
TABLE_TOP_Z = TABLE_CENTER[2] + TABLE_HALF_SIZE[2]

PAD_HALF_HEIGHT = 0.004
BLOCK_SIZE = 0.02
BLOCK_DENSITY = 500.0
STACK_SPACING = 0.18
STACK_X = 0.50
STACK_Y = {1: -STACK_SPACING, 2: 0.0, 3: STACK_SPACING}
BLOCK_GAP = 0.0005
STACK_BASE_Z = TABLE_TOP_Z + 2.0 * PAD_HALF_HEIGHT + BLOCK_SIZE + BLOCK_GAP

HOVER_Z = 0.14
GRASP_Z = 0.0
PLACE_Z = -0.008
SIDE_HOVER_OFFSET = 0.12
SIDE_GRASP_OFFSET = 0.055

Q_HOME = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.8], dtype=float)
READY_POS = np.array([STACK_X, 0.0, STACK_BASE_Z + 0.22], dtype=float)

PAD_COLORS = {
    1: [0.85, 0.35, 0.35, 0.7],
    2: [0.35, 0.75, 0.40, 0.7],
    3: [0.35, 0.48, 0.90, 0.7],
}


def format_vec(values):
    return " ".join(f"{float(v):.6f}" for v in values)


def get_active_blocks(n_blocks):
    if n_blocks < 1 or n_blocks > MAX_BLOCK_COUNT:
        raise ValueError(f"n must be between 1 and {MAX_BLOCK_COUNT}.")
    active_names = ALL_BLOCK_NAMES[:n_blocks]
    return active_names, list(reversed(active_names))


def block_color(block_name):
    idx = BLOCK_RANK[block_name]
    hue = (idx * 0.137) % 1.0
    sat = 0.72
    val = 0.92

    h6 = hue * 6.0
    sector = int(h6) % 6
    frac = h6 - int(h6)
    p = val * (1.0 - sat)
    q = val * (1.0 - frac * sat)
    t = val * (1.0 - (1.0 - frac) * sat)

    if sector == 0:
        rgb = (val, t, p)
    elif sector == 1:
        rgb = (q, val, p)
    elif sector == 2:
        rgb = (p, val, t)
    elif sector == 3:
        rgb = (p, q, val)
    elif sector == 4:
        rgb = (t, p, val)
    else:
        rgb = (val, p, q)

    return [rgb[0], rgb[1], rgb[2], 1.0]


def stack_position(slot, level):
    return np.array(
        [
            STACK_X,
            STACK_Y[slot],
            STACK_BASE_Z + level * (2.0 * BLOCK_SIZE + BLOCK_GAP),
        ],
        dtype=float,
    )


def add_static_box(worldbody, name, pos, size, rgba):
    body = ET.SubElement(worldbody, "body", {"name": name, "pos": format_vec(pos)})
    ET.SubElement(
        body,
        "geom",
        {
            "name": f"{name}_geom",
            "type": "box",
            "size": format_vec(size),
            "rgba": format_vec(rgba),
            "friction": "1.0 0.2 0.1",
        },
    )


def add_free_block(worldbody, name, pos, rgba):
    body = ET.SubElement(worldbody, "body", {"name": f"block_{name}", "pos": format_vec(pos)})
    ET.SubElement(body, "freejoint", {"name": f"block_{name}_freejoint"})
    ET.SubElement(
        body,
        "geom",
        {
            "name": f"block_{name}_geom",
            "type": "box",
            "size": f"{BLOCK_SIZE:.6f} {BLOCK_SIZE:.6f} {BLOCK_SIZE:.6f}",
            "density": f"{BLOCK_DENSITY:.6f}",
            "rgba": format_vec(rgba),
            "friction": "1.2 0.2 0.1",
        },
    )


def build_scene_xml(active_bottom_to_top):
    tree = ET.parse(ROOT_MODEL_XML)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError("Base Panda XML is missing <worldbody>.")

    add_static_box(
        worldbody,
        name="hanoi_table",
        pos=TABLE_CENTER,
        size=TABLE_HALF_SIZE,
        rgba=[0.73, 0.73, 0.77, 1.0],
    )

    for slot in (1, 2, 3):
        add_static_box(
            worldbody,
            name=f"slot_{slot}",
            pos=np.array([STACK_X, STACK_Y[slot], TABLE_TOP_Z + PAD_HALF_HEIGHT], dtype=float),
            size=np.array([0.055, 0.055, PAD_HALF_HEIGHT], dtype=float),
            rgba=PAD_COLORS[slot],
        )

    for level, block_name in enumerate(active_bottom_to_top):
        add_free_block(
            worldbody,
            name=block_name,
            pos=stack_position(1, level),
            rgba=block_color(block_name),
        )

    tree.write(SCENE_MODEL_XML, encoding="utf-8", xml_declaration=True)
    return SCENE_MODEL_XML


def rotmat_top_down():
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
        ]
    )


def rotmat_side_grasp_from_left():
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
        ]
    )


def rotmat_side_grasp_from_right():
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ]
    )


def rotz(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def pose_error(target_p, target_R, cur_p, cur_R):
    ep = target_p - cur_p
    eR = 0.5 * (
        np.cross(cur_R[:, 0], target_R[:, 0])
        + np.cross(cur_R[:, 1], target_R[:, 1])
        + np.cross(cur_R[:, 2], target_R[:, 2])
    )
    return np.hstack([ep, eR])


def set_arm_qpos(model, data, q, arm_idx):
    data.qpos[arm_idx] = q
    data.qvel[arm_idx] = 0.0
    mj.mj_forward(model, data)


def solve_ik_damped(
    model,
    data,
    site_name,
    q_init,
    target_p,
    target_R,
    arm_idx,
    max_iters=300,
    tol=1e-4,
    damping=1e-2,
    step_size=0.5,
):
    q = q_init.copy()
    sid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, site_name)
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))

    for _ in range(max_iters):
        set_arm_qpos(model, data, q, arm_idx)
        cur_p = data.site_xpos[sid].copy()
        cur_R = data.site_xmat[sid].reshape(3, 3).copy()
        err = pose_error(target_p, target_R, cur_p, cur_R)
        if np.linalg.norm(err) < tol:
            return q

        mj.mj_jacSite(model, data, jacp, jacr, sid)
        J = np.vstack([jacp[:, arm_idx], jacr[:, arm_idx]])
        JT = J.T
        dq = JT @ np.linalg.solve(J @ JT + damping * np.eye(6), err)
        q = q + step_size * dq

        for i, jidx in enumerate(arm_idx):
            qmin, qmax = model.jnt_range[jidx]
            q[i] = np.clip(q[i], qmin, qmax)

    raise RuntimeError("IK did not converge")


def solve_ready_pose(model, data, arm_idx, ee_site, q_seed):
    return solve_ik_damped(model, data, ee_site, q_seed, READY_POS, rotmat_top_down(), arm_idx)


def get_body_pos(model, data, body_name):
    bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name)
    if bid < 0:
        raise ValueError(f"Body '{body_name}' not found.")
    return data.xpos[bid].copy()


def top_grasp_targets(target_pos):
    target_rot = rotmat_top_down() @ rotz(0.0)
    hover_pos = target_pos + np.array([0.0, 0.0, HOVER_Z], dtype=float)
    grasp_pos = target_pos + np.array([0.0, 0.0, GRASP_Z], dtype=float)
    return hover_pos, grasp_pos, target_rot


def side_grasp_targets(target_pos, slot):
    if slot == 1:
        approach_sign = -1.0
        target_rot = rotmat_side_grasp_from_right()
    elif slot == 3:
        approach_sign = 1.0
        target_rot = rotmat_side_grasp_from_left()
    else:
        approach_sign = 1.0
        target_rot = rotmat_side_grasp_from_left()

    hover_pos = target_pos + np.array([0.0, approach_sign * SIDE_HOVER_OFFSET, 0.0], dtype=float)
    grasp_pos = target_pos + np.array([0.0, approach_sign * SIDE_GRASP_OFFSET, GRASP_Z], dtype=float)
    return hover_pos, grasp_pos, target_rot


def solve_single_move_ik(
    model,
    data,
    arm_idx,
    ee_site,
    pick_pos,
    pick_slot,
    place_pos,
    place_slot,
    q_seed,
    grasp_mode,
):
    if grasp_mode == "side":
        pick_hover_pos, pick_grasp_pos, r_pick = side_grasp_targets(pick_pos, pick_slot)
        place_hover_pos, place_drop_pos, r_place = side_grasp_targets(
            place_pos + np.array([0.0, 0.0, PLACE_Z], dtype=float),
            place_slot,
        )
    elif grasp_mode == "top":
        pick_hover_pos, pick_grasp_pos, r_pick = top_grasp_targets(pick_pos)
        place_hover_pos, place_drop_pos, r_place = top_grasp_targets(
            place_pos + np.array([0.0, 0.0, PLACE_Z], dtype=float)
        )
    else:
        raise ValueError(f"Unsupported grasp_mode: {grasp_mode}")

    q_ph = solve_ik_damped(model, data, ee_site, q_seed, pick_hover_pos, r_pick, arm_idx)
    q_pg = solve_ik_damped(model, data, ee_site, q_ph, pick_grasp_pos, r_pick, arm_idx)
    q_lh = solve_ik_damped(model, data, ee_site, q_ph, place_hover_pos, r_place, arm_idx)
    q_ld = solve_ik_damped(model, data, ee_site, q_lh, place_drop_pos, r_place, arm_idx)
    return q_ph, q_pg, q_lh, q_ld


def build_move_waypoints(q_ready_start, move_ik, q_ready_end):
    q_ph, q_pg, q_lh, q_ld = move_ik
    return [
        (q_ready_start, OPEN_WIDTH),
        (q_ph, OPEN_WIDTH),
        (q_pg, OPEN_WIDTH),
        (q_pg, CLOSED_WIDTH),
        (q_pg, CLOSED_WIDTH),
        (q_ph, CLOSED_WIDTH),
        (q_lh, CLOSED_WIDTH),
        (q_ld, CLOSED_WIDTH),
        (q_ld, OPEN_WIDTH),
        (q_lh, OPEN_WIDTH),
        (q_ready_end, OPEN_WIDTH),
    ]


def execute_waypoints(model, data, arm_idx, gripper_idx, waypoints, viewer_handle=None):
    dt = model.opt.timestep
    segment_steps = max(1, int(SEGMENT_DURATION / dt))
    hold_steps = int(HOLD_DURATION / dt)

    for i in range(len(waypoints) - 1):
        q_start, gripper_cmd = waypoints[i]
        q_goal, _ = waypoints[i + 1]
        t = 0.0

        for _ in range(segment_steps + hold_steps):
            q_des, qd_des = rt.interp_min_jerk(q_start, q_goal, t, SEGMENT_DURATION)
            q = data.qpos[arm_idx].copy()
            qd = data.qvel[arm_idx].copy()
            tau = KP * (q_des - q) + KD * (qd_des - qd)
            data.ctrl[arm_idx] = tau + data.qfrc_bias[:7]
            data.ctrl[gripper_idx] = gripper_cmd
            mj.mj_step(model, data)
            if viewer_handle is not None:
                viewer_handle.sync()
            t += dt


def plan_hanoi(n, src, dst, aux, moves):
    if n == 0:
        return
    plan_hanoi(n - 1, src, aux, dst, moves)
    moves.append((src, dst))
    plan_hanoi(n - 1, aux, dst, src, moves)


def validate_move(pegs, src, dst):
    if not pegs[src]:
        raise RuntimeError(f"Position {src} is empty.")

    block = pegs[src][-1]
    if pegs[dst]:
        top_dst = pegs[dst][-1]
        if BLOCK_RANK[block] > BLOCK_RANK[top_dst]:
            raise RuntimeError(f"Illegal move: {block} cannot be placed on {top_dst}.")
    return block


def create_initial_state(active_bottom_to_top):
    return {
        1: active_bottom_to_top.copy(),
        2: [],
        3: [],
    }


def create_goal_state(active_bottom_to_top):
    return {
        1: [],
        2: [],
        3: active_bottom_to_top.copy(),
    }


def settle_scene(model, data, arm_idx, gripper_idx, q_hold, grip_width, steps, viewer_handle=None):
    for _ in range(steps):
        q = data.qpos[arm_idx].copy()
        qd = data.qvel[arm_idx].copy()
        tau = KP * (q_hold - q) - KD * qd
        data.ctrl[arm_idx] = tau + data.qfrc_bias[:7]
        data.ctrl[gripper_idx] = grip_width
        mj.mj_step(model, data)
        if viewer_handle is not None:
            viewer_handle.sync()


def run_hanoi(n_blocks, grasp_mode, viewer_enabled=True):
    active_blocks, active_bottom_to_top = get_active_blocks(n_blocks)
    scene_xml = build_scene_xml(active_bottom_to_top)

    model = mj.MjModel.from_xml_path(scene_xml)
    data = mj.MjData(model)
    arm_idx = list(range(7))
    gripper_idx = 7
    ee_site = "grip_site"

    pegs = create_initial_state(active_bottom_to_top)
    goal_state = create_goal_state(active_bottom_to_top)
    moves = []
    plan_hanoi(n_blocks, 1, 3, 2, moves)

    data.qpos[arm_idx] = Q_HOME
    data.qvel[arm_idx] = 0.0
    data.ctrl[gripper_idx] = OPEN_WIDTH
    mj.mj_forward(model, data)

    try:
        q_ready = solve_ready_pose(model, data, arm_idx, ee_site, Q_HOME)
    except RuntimeError:
        q_ready = Q_HOME.copy()

    def execute(viewer_handle=None):
        nonlocal q_ready

        settle_scene(model, data, arm_idx, gripper_idx, q_ready, OPEN_WIDTH, 200, viewer_handle)

        for move_idx, (src, dst) in enumerate(moves, start=1):
            block_name = validate_move(pegs, src, dst)
            pick_pos = get_body_pos(model, data, f"block_{block_name}")
            place_level = len(pegs[dst])
            place_pos = stack_position(dst, place_level)

            print(f"Move {move_idx:02d}/{len(moves)}: {block_name}  {src} -> {dst}")
            move_ik = solve_single_move_ik(
                model,
                data,
                arm_idx,
                ee_site,
                pick_pos,
                src,
                place_pos,
                dst,
                q_ready,
                grasp_mode,
            )

            try:
                q_ready_end = solve_ready_pose(model, data, arm_idx, ee_site, move_ik[2])
            except RuntimeError:
                q_ready_end = q_ready.copy()

            waypoints = build_move_waypoints(q_ready, move_ik, q_ready_end)
            execute_waypoints(model, data, arm_idx, gripper_idx, waypoints, viewer_handle)

            pegs[src].pop()
            pegs[dst].append(block_name)
            q_ready = q_ready_end.copy()

            settle_scene(model, data, arm_idx, gripper_idx, q_ready, OPEN_WIDTH, 120, viewer_handle)

    if viewer_enabled:
        with viewer.launch_passive(model, data) as v:
            v.cam.lookat[:] = [0.48, 0.0, 0.18]
            v.cam.distance = 1.6
            v.cam.azimuth = 120
            v.cam.elevation = -24
            v.sync()
            execute(v)
            time.sleep(1.0)
    else:
        execute(None)

    if pegs != goal_state:
        raise RuntimeError("Execution finished but the final tower is not at position 3.")

    print("Finished. Final state:")
    print(pegs)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tower of Hanoi with a MuJoCo Panda arm using real Lab 3-style grasp/release."
    )
    parser.add_argument(
        "--n",
        type=int,
        default=MAX_BLOCK_COUNT,
        help=f"Total number of blocks to use (1 to {MAX_BLOCK_COUNT}).",
    )
    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="Run without opening the MuJoCo viewer.",
    )
    parser.add_argument(
        "--grasp-mode",
        choices=["top", "side"],
        default="side",
        help="Choose whether the gripper performs top grasp or side grasp.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_hanoi(n_blocks=args.n, grasp_mode=args.grasp_mode, viewer_enabled=not args.no_viewer)


if __name__ == "__main__":
    main()
