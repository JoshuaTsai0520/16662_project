# Robot Arm Drawing Project

A Franka Emika Panda arm draws a picture on a sheet of paper inside MuJoCo.
The robot uses the same torque-control scheme as the Hanoi Tower project.

---

## Files

| File | Purpose |
|------|---------|
| `drawing_complete.py` | Main script — builds the scene, processes the image, solves IK, and runs the simulation. |
| `image_to_points.py` | Image-processing helper — converts an input picture to an ordered list of 2-D draw-points. |
| `franka_emika_panda/panda_torque_table_drawing.xml` | MuJoCo scene XML (auto-generated at runtime, do not edit by hand). |
| `franka_emika_panda/panda_torque_table.xml` | Base Panda robot model (shared with the Hanoi project). |
| `RobotUtil.py` | Shared robot utilities — minimum-jerk interpolation, etc. (shared with the Hanoi project). |

---

## How it works

### 1 — Image processing (`image_to_points.py`)

1. Load the input image and convert it to **grayscale**.
2. Apply a **binary threshold** (default 128) — pixels darker than the threshold are marked as "draw here".
3. Optionally extract only the **edge pixels** of the mask (default on) using PIL's FIND_EDGES filter, giving a contour-style drawing.
4. **Sub-sample** to at most `--max-points` points.
5. **Reorder** with a greedy nearest-neighbor pass to minimise total pen travel.
6. **Normalise** coordinates to [0, 1]².

### 2 — Scene construction

A new MuJoCo XML is generated each run from the base Panda model, adding:
- A **drawing table** (grey box, 60 × 80 × 10 cm, top surface at z = 0.10 m).
- A **sheet of paper** (white box, 26 × 26 cm, on top of the table).
- A **visual pen** (thin dark cylinder) attached to the robot's hand.

### 3 — Drawing motion

Normalised image points are mapped to 3-D world coordinates on the paper:
- Image *x* (left → right) maps to world *Y*.
- Image *y* (top → bottom) maps to world *X* (top of image = far side of paper).

The point sequence is split into **strokes** separated by pen-up moves.
For each stroke the robot:
1. Lifts the pen and moves to the hover position above the first point.
2. Lowers the pen to the paper surface.
3. Draws through all points in the stroke with pen down.
4. Lifts the pen after the last point.

IK is solved with a damped least-squares method; joint motion uses PD + gravity-compensation torque control with minimum-jerk interpolation.

---

## Dependencies

```
mujoco
numpy
Pillow
```

Install with:

```bash
pip install mujoco numpy Pillow
```

---

## Usage

```bash
python drawing_complete.py <image_path> [options]
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `image` | *(required)* | Path to any image file (PNG, JPG, BMP, …). |
| `--style lines\|dots` | `lines` | **lines**: traces image contours as continuous strokes. **dots**: stamps a discrete dot at each sampled dark pixel. |
| `--max-points N` | `400` | Maximum draw-points extracted from the image. More points → finer detail, longer runtime. |
| `--threshold T` | `128` | Grayscale threshold (0–255). Pixels **darker** than T are drawn. |
| `--invert` | off | Draw light pixels instead of dark ones (useful for white-on-black images). |
| `--no-viewer` | off | Run headless (no MuJoCo viewer window). |
| `--speed X` | `1.0` | Playback speed multiplier (e.g. `2.0` = twice as fast). |

### Examples

Draw contours of `logo.png` as continuous lines (default):

```bash
python drawing_complete.py logo.png --style lines
```

Draw a pointillist dot pattern:

```bash
python drawing_complete.py logo.png --style dots
```

Draw a white-on-black image (invert the mask):

```bash
python drawing_complete.py white_on_black.png --invert
```

Use more points for higher detail (slower):

```bash
python drawing_complete.py photo.png --max-points 600 --threshold 100
```

Run headless at double speed:

```bash
python drawing_complete.py logo.png --no-viewer --speed 2.0
```

---

## Tips

- **Best input images**: simple logos, icons, silhouettes, or line-art sketches work best.  
  The edge-detection step turns photos into contour drawings automatically.
- **Threshold tuning**: if the robot draws nothing, try `--threshold 200` (darker threshold).  
  If it draws too much noise, try `--threshold 80`.
- **Speed**: start with `--speed 2.0` or `--speed 3.0` to get through a full drawing faster.
- The MuJoCo viewer camera starts at an angle above the table. Use the mouse to orbit and zoom.
