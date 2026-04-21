# Code Walkthrough

This document traces the execution path of both projects in the repository —
the **Drawing** project and the **Hanoi Tower** project — explaining what every
major piece of code does and how the pieces fit together.

---

## Repository layout

```
16662_project/
├── drawing_complete.py        # Robot arm drawing simulation (main script)
├── image_to_points.py         # Image → draw-points + preview PNG
├── hanoi_tower_complete.py    # Tower of Hanoi robot simulation (main script)
├── RobotUtil.py               # Shared math utilities (IK, interpolation, …)
└── franka_emika_panda/
    ├── panda_torque_table.xml           # Base Panda robot model (hand-written)
    ├── panda_torque_table_drawing.xml   # Drawing scene  (auto-generated at runtime)
    └── panda_torque_table_hanoi_complete.xml  # Hanoi scene (auto-generated at runtime)
```

Both projects share the same physical robot model and the same control
strategy.  The scene XML files are regenerated every run so that objects
(blocks, paper, pen) are always placed correctly.

---

## Part 1 — Drawing project

### 1.1 Entry point

```
python3 drawing_complete.py  logo.png  --style lines  --resolution 200
```

`main()` in `drawing_complete.py` calls `parse_args()` then `run_drawing()`.
All runtime parameters (image path, style, resolution, mark size, speed) flow
through `run_drawing()`.

### 1.2 Image processing  (`image_to_points.py`)

`run_drawing()` calls either `image_to_strokes()` or `image_to_dot_points()`
depending on `--style`.

#### `_load_binary(image_path, threshold, invert, resolution)`

```
PNG/JPG  →  grayscale  →  resize (longer side ≤ resolution)  →  bool array
```

- Opens the image and converts to single-channel grayscale (`"L"` mode).
- If the image is larger than `resolution` pixels on its longer side it is
  scaled down with `Image.LANCZOS` (high-quality downsampling).
- Every pixel whose brightness is **below** `threshold` becomes `True`
  (= "draw here").  `--invert` flips this.

The resize step is the primary quality knob.  A 150-pixel working image
produces ~500–2000 edge pixels; a 300-pixel image produces ~2000–8000.

#### `_edge_mask(binary)`

Applies PIL's `FIND_EDGES` kernel to the boolean mask, then thresholds the
result at 32.  This produces a 1-pixel-thin outline of every dark region,
reducing the total point count while keeping the shape.

#### `_trace_contours(mask)`  ← lines mode only

```python
remaining = {all edge pixels as (x, y) tuples}
while remaining:
    pick any start pixel
    while a connected 8-neighbor exists in remaining:
        prefer the neighbor that continues the current direction
        append it and remove from remaining
    → one stroke (ordered list of pixels)
```

The direction-preference heuristic (`nbrs.sort(key=…)`) stops the tracer
from zig-zagging at junctions, giving smoother curves.

Each call to this function produces a **list of strokes** — each stroke is an
`(K, 2)` array of pixel coordinates in drawing order.

#### `_subsample_strokes(strokes, max_points)`

After tracing, the total pixel count can be very large.  This function thins
each stroke proportionally so that the grand total stays ≤ `max_points`.  At
least 2 points are kept per stroke so that single-pixel strokes are not
collapsed to nothing.

#### `_nearest_neighbor_order(points)`  ← dots mode only

Greedy O(n²) nearest-neighbor: repeatedly pick the unvisited point closest to
the current position.  This minimises total pen travel between dots.

#### `image_to_strokes()` and `image_to_dot_points()`

Both functions call `_load_binary`, then apply the appropriate processing
chain, then **normalise** all pixel coordinates to [0, 1]²:

```
x_norm = x_pixel / (image_width  - 1)
y_norm = y_pixel / (image_height - 1)
```

`image_to_strokes` returns a `list[ndarray]` (one array per stroke).
`image_to_dot_points` returns a single flat `(N, 2)` array.

#### `save_preview(draw_data, style, output_path)`

Renders the normalised draw-points onto a 512×512 white PIL image and saves
it as `<stem>_preview.png` **before the simulation starts**, so you can check
the result without waiting for the robot.

- Lines mode: draws connected polylines with `ImageDraw.line()`.
- Dots mode: draws filled circles with `ImageDraw.ellipse()`.

---

### 1.3 Scene construction  (`build_scene_xml()`)

`run_drawing()` calls `build_scene_xml()` which:

1. **Parses** `franka_emika_panda/panda_torque_table.xml` (the base Panda
   model) using Python's `xml.etree.ElementTree`.
2. **Appends** three new bodies to `<worldbody>`:

   | Body | What it is | Collision |
   |------|-----------|-----------|
   | `drawing_table` | Grey box, top surface at z = 0.10 m | Off — prevents arm links colliding with it |
   | `paper` | Thin white box on the table | Off — visual only |
   | `pen` (child of `hand`) | Cylinder geoms inside the gripper | Off — visual only |

3. Inside `pen_body` (attached to the robot hand at hand-frame z = 0.103 m)
   it adds a **site** called `pen_tip_site` at local position z = 0.052 m:

   ```
   hand body origin
     └─ pen_body  (z = 0.103 in hand frame)
          ├─ pen_shaft geom  (z = 0.000 … 0.044)
          ├─ pen_ink_tip geom  (z = 0.044 … 0.052)
          └─ pen_tip_site  ← z = 0.052  ← IK target
   ```

   When the arm points straight down (`DRAW_ROT`), the hand's +z axis aligns
   with world −Z, so `pen_tip_site` ends up **0.052 m below `pen_body`
   origin** = **0.155 m below the hand body origin** in world space.

4. **Writes** the modified XML to `panda_torque_table_drawing.xml` and
   returns the path.

---

### 1.4 IK solver  (`solve_ik()`)

```
Input : target 3-D position for pen_tip_site, current joint angles (seed)
Output: joint angles q[0..6] such that pen_tip_site ≈ target position
        and end-effector orientation ≈ DRAW_ROT (pointing straight down)
```

Uses **damped least-squares** (also called Levenberg–Marquardt):

```python
# Each iteration:
err = [position_error (3D),  orientation_error (3D)]   # 6-vector

J   = jacobian of [pen_tip_site pos, site orientation] w.r.t. joint angles
                                                         # 6×7 matrix

dq  = J^T · (J·J^T + λ·I)^{-1} · err   # damped pseudo-inverse step
q  += step_size · dq
```

- `mj.mj_jacSite` computes the 3×nv translational and rotational Jacobians
  in one MuJoCo call.
- The damping term `λ = 0.01` prevents blow-up near singular configurations.
- Joint limits are clamped after every step.
- Returns best-effort result if the solver has not converged after
  `max_iters = 200` iterations.

The orientation error uses the **cross-product formula** for small-angle
rotation error:

```python
eR = 0.5 * (cross(cur_R[:,0], target_R[:,0])
           + cross(cur_R[:,1], target_R[:,1])
           + cross(cur_R[:,2], target_R[:,2]))
```

---

### 1.5 Segment executor  (`execute_segment()`)

Moves the arm from joint configuration `q_start` to `q_goal` over `duration`
seconds using **PD torque control with gravity compensation**:

```python
for each simulation step:
    q_des, qd_des = interp_min_jerk(q_start, q_goal, t, duration)
    τ = Kp · (q_des − q) + Kd · (qd_des − qd)   # PD
    ctrl = τ + qfrc_bias[:7]                       # + gravity compensation
```

- `interp_min_jerk` (from `RobotUtil.py`) produces a smooth quintic
  polynomial profile — zero velocity and acceleration at both endpoints, so
  motions start and stop smoothly.
- `data.qfrc_bias[:7]` is MuJoCo's gravity + Coriolis compensation vector for
  the 7 arm joints, read directly from the simulator at every timestep.
- After `n_steps` motion steps, an optional `hold_secs` dwell is appended
  (the arm holds the goal pose).  In-stroke pen-down moves use `hold_secs=0`
  so the arm flows continuously from point to point.

#### Ink marks

`execute_segment()` accepts optional `mark_every`, `mark_scn`, `mark_site_id`
parameters.  When `mark_every > 0`, every N simulation steps it reads the
current world position of `pen_tip_site` and calls `_add_ink_mark()`:

```python
def _add_ink_mark(scn, pos, size):
    g = scn.geoms[scn.ngeom]
    mj.mjv_initGeom(g, mjGEOM_SPHERE, [size,size,size], pos, eye3, black_rgba)
    scn.ngeom += 1
```

`viewer.user_scn` is a secondary `mjvScene` that MuJoCo renders on top of the
regular scene.  Adding spheres there at the actual pen-tip positions creates
persistent black marks that accumulate on the paper as the arm draws.

---

### 1.6 Drawing loops

#### Lines mode  (`_draw_lines()`)

For each stroke from `image_to_strokes`:

```
First point of stroke:
  1. solve_ik → hover position (pen_tip at LIFT_Z = DRAW_Z + 0.08 m)
  2. execute_segment: move to hover          (hold = default)
  3. solve_ik → draw position (pen_tip at DRAW_Z = paper surface)
  4. execute_segment: lower pen to paper     (marks added, hold = default)

Each subsequent point:
  5. solve_ik → next draw position
  6. execute_segment: slide pen along paper  (marks added, hold_secs = 0)
     ↑ no pause here — arm flows continuously, forming a line

After last point of stroke:
  7. solve_ik → lift position (pen_tip back to LIFT_Z)
  8. execute_segment: lift pen               (no marks)
```

Mark spheres are added every `MARK_INTERVAL = 8` simulation steps during
steps 4 and 6.  This dense chain of 1.8 mm spheres looks like a continuous
ink line.

#### Dots mode  (`_draw_dots()`)

`_group_points()` first clusters the flat dot array into groups where
consecutive dots are no more than `PEN_UP_DIST = 0.07` (normalised) apart.
Within each group the arm slides at paper height; between groups it lifts.

For each dot:
```
First dot of group:  hover → lower (0.2 s dwell) → stamp mark
Subsequent dots:     slide to next position (0.15 s dwell) → stamp mark
After last dot:      lift
```

One 4.5 mm sphere is stamped at the **actual** `data.site_xpos[site_id]`
position after each pen-touch — one discrete dot per commanded position.

---

### 1.7 Coordinate mapping  (`norm_to_world()`)

Normalised image coordinates (x_norm, y_norm) map to 3-D world XY:

```
world_X = DRAW_ORIG_X + (1 − y_norm) × DRAW_AREA
world_Y = DRAW_ORIG_Y +      x_norm  × DRAW_AREA
world_Z = DRAW_Z
```

- Image **x** (left → right) maps to world **Y**.
- Image **y** (top → bottom) maps to world **X reversed**, so the top of the
  image is the far side of the paper (away from the robot base) and the bottom
  is the near side.  This keeps the image right-side-up when viewed from
  above.

---

## Part 2 — Hanoi Tower project  (`hanoi_tower_complete.py`)

### 2.1 Overview

The arm solves and executes the Tower of Hanoi puzzle: moving a stack of
`n` blocks from peg 1 to peg 3 using peg 2 as auxiliary.

```
python3 hanoi_tower_complete.py --n 3 --grasp-mode top
```

### 2.2 Scene generation  (`build_scene_xml()`)

Same approach as the drawing project: parse the base Panda XML, add bodies,
write a new XML file.

- **Table**: grey box (same dimensions as drawing table).
- **3 pad bodies**: coloured squares on the table marking peg positions.
- **N block bodies**: free-floating boxes with `<freejoint>`, meaning MuJoCo
  simulates their full 6-DOF physics (they can fall, slide, stack under gravity).

Block positions are computed by `stack_position(slot, level)`:

```python
z = TABLE_TOP_Z + 2 * PAD_HALF_HEIGHT + BLOCK_SIZE + (level * (2*BLOCK_SIZE + GAP))
```

### 2.3 Hanoi planning  (`plan_hanoi()`)

Classic recursive algorithm:

```python
def plan_hanoi(n, src, dst, aux, moves):
    if n == 0: return
    plan_hanoi(n-1, src, aux, dst, moves)  # move n-1 blocks to aux
    moves.append((src, dst))               # move largest block to dst
    plan_hanoi(n-1, aux, dst, src, moves)  # move n-1 blocks from aux to dst
```

For n=3 this produces 7 moves; for n discs it produces 2ⁿ−1 moves.

### 2.4 Grasp modes

Two approaches for picking up a block:

| Mode | Gripper approach | When to use |
|------|-----------------|-------------|
| `top` | Descend vertically, grip from above | All stacks |
| `side` | Approach horizontally, grip from the side | Tall stacks where top-down is obstructed |

`top_grasp_targets()` and `side_grasp_targets()` compute hover position,
grasp position, and end-effector rotation for each mode.

### 2.5 Move execution

For each planned move `(src, dst)`:

1. Read the **current** block position from `data.xpos` (not the planned
   position — the block may have shifted due to physics).
2. Solve IK for 4 key poses: pick-hover, pick-grasp, place-hover, place-drop.
3. Build a **waypoint sequence**:
   ```
   ready → pick-hover → pick-grasp → [close gripper] →
   pick-hover → place-hover → place-drop → [open gripper] →
   place-hover → ready
   ```
4. Execute each segment with `execute_waypoints()` (same PD + gravity control
   as the drawing project, but using a fixed segment duration rather than
   distance-proportional timing).
5. A settle phase after each placement lets the block physics stabilise before
   the next move.

---

## Part 3 — Shared utilities  (`RobotUtil.py`)

### `interp_min_jerk(q_start, q_goal, t, T)`

Returns `(q_des, qd_des)` — desired joint position and velocity at time `t`
for a motion of duration `T`.

Uses a **quintic (5th-order) polynomial** profile:

```
τ = clip(t / T, 0, 1)
s(τ)  = 10τ³ − 15τ⁴ + 6τ⁵        ← position profile [0 → 1]
ṡ(τ) = (30τ² − 60τ³ + 30τ⁴) / T  ← velocity profile
```

The polynomial is chosen so that position, velocity, and acceleration are all
zero at τ=0 and τ=1, giving the smoothest possible motion (minimum jerk =
minimum rate of change of acceleration).

### `rpyxyz2H(rpy, xyz)`

Builds a 4×4 homogeneous transformation matrix from roll-pitch-yaw angles and
a translation vector.  Used internally for collision checking.

### `axis_angle_between(v1, v2)`

Computes the rotation axis and angle needed to rotate unit vector `v1` to
align with unit vector `v2`.  Handles the degenerate cases (parallel,
anti-parallel vectors) explicitly.

### `CheckBoxBoxCollision(pointsA, axesA, pointsB, axesB)`

Separating Axis Theorem (SAT) collision check between two oriented bounding
boxes.  Tests 15 axes: 3 face normals of A, 3 face normals of B, and 9
edge-edge cross products.

---

## Part 4 — Base robot model  (`panda_torque_table.xml`)

The base XML describes the complete Franka Emika Panda:

- **7 revolute arm joints** (`joint1` … `joint7`), each with joint limits and
  damping.
- **2 prismatic finger joints** (`finger_joint1`, `finger_joint2`), coupled by
  a `<tendon>` so they move symmetrically.
- **8 actuators**: 7 general force actuators for the arm (torque control), and
  1 position actuator on the finger tendon (gripper open/close).
- **`grip_site`**: a site at the center of the fingertip gap — the original IK
  target for grasping tasks.  The drawing project adds `pen_tip_site` in the
  generated XML instead.

### Actuator indexing

```
data.ctrl[0..6]  → arm joint torques   (arm_idx = list(range(7)))
data.ctrl[7]     → gripper tendon      (gripper_idx = 7)
data.qfrc_bias[:7] → gravity + Coriolis compensation for the 7 arm joints
```

---

## Part 5 — Data flow diagram

```
                         Drawing project
─────────────────────────────────────────────────────────────────
  Input image
      │
      ▼
  _load_binary()         resize + threshold → bool mask
      │
      ├── lines → _edge_mask() → _trace_contours() → list of strokes
      │                                   │
      │                          _subsample_strokes()
      │                                   │
      └── dots  → np.where() → _nearest_neighbor_order() → flat points
                                          │
                              ┌───────────┘
                              │
                        save_preview()   ← PNG written here (before sim)
                              │
                        build_scene_xml()   ← table + paper + pen XML
                              │
                        MjModel.from_xml_path()
                              │
                   ┌──────────┴──────────┐
                   │                     │
             lines mode               dots mode
          _draw_lines()            _draw_dots()
                   │                     │
           for each point:        for each point:
             solve_ik()             solve_ik()
                   │                     │
          execute_segment()     execute_segment()
          + mark_every=8        + stamp 1 sphere after each touch
                   │
            _add_ink_mark()  →  viewer.user_scn  ← visible black marks


                         Hanoi project
─────────────────────────────────────────────────────────────────
  n blocks
      │
      ▼
  build_scene_xml()       table + pads + free blocks
      │
  plan_hanoi()            recursive move sequence
      │
  for each move:
    get_body_pos()        read current block position from physics
    solve_single_move_ik()  4 IK solutions per move
    build_move_waypoints()  11-step waypoint list
    execute_waypoints()   PD torque control along waypoints
    settle_scene()        let block physics stabilise
```

---

## Quick reference — key constants

| Constant | File | Value | Meaning |
|----------|------|-------|---------|
| `DEFAULT_RESOLUTION` | `image_to_points.py` | 300 | Default image resize limit (px) |
| `DRAW_Z` | `drawing_complete.py` | ≈ 0.102 m | World Z of pen tip while drawing |
| `LIFT_Z` | `drawing_complete.py` | ≈ 0.182 m | World Z of pen tip while travelling |
| `DRAW_AREA` | `drawing_complete.py` | 0.22 m | Side of drawable square on paper |
| `PEN_UP_DIST` | `drawing_complete.py` | 0.07 (normalised) | Gap that triggers pen-lift |
| `MARK_INTERVAL` | `drawing_complete.py` | 8 steps | Ink sphere density (lines mode) |
| `SEGMENT_DURATION` | both | 1.5 s | Default arm segment duration |
| `KP / KD` | both | [120…30] / [8…2] | PD controller gains per joint |
| `Q_HOME` | both | [0, −0.5, 0, −2, 0, 1.5, 0.8] | Safe home joint configuration |
