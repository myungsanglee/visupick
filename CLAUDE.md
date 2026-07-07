# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

**VisuPick** — a single PySide6 desktop app for 3D-vision-guided bin picking with a KUKA robot (KRC5 + KSS 8.7) and a Zivid / RealSense 3D camera. One GUI covers the full workflow: data collection → hand-eye calibration → verification → bin picking / CAD 6D-pose matching / surface tracking.

UI text, code comments, and the `docs/` learning guides are in **Korean** — keep new user-facing strings and comments Korean to match.

## Commands

```bash
source venv/bin/activate       # venv/ is committed-ignored but present locally
pip install -r requirements.txt
python main.py                 # sole entry point — launches the GUI
```

There is **no test suite, linter, or build step**. Verification is manual through the GUI (needs a real robot + camera, or their SDKs stubbed). Do not assume `pytest`/`make`/CI exists.

## Architecture — the big picture

Understanding these cross-file relationships is the fast path to being productive:

**Shared state lives in one place.** [main.py](main.py) `VisuPickApp(QMainWindow)` owns the single `self.robot` (a `KUKARobot`) and `self.camera` (a `BaseCamera`). It constructs every tab passing *itself* as `main_window`. Tabs therefore reach hardware via `self.main.robot` / `self.main.camera` — they never own connections. Connect/disconnect buttons live on the main window, not the tabs.

**Five tabs, two mixins:**
- [main.py](main.py) defines `DataCollectionTab` and `VerificationTab` (both use `ImageViewerMixin`).
- `BinPickingTab` ([bin_picking_tab.py](bin_picking_tab.py)), `CADMatchingTab` ([cad_matching_tab.py](cad_matching_tab.py)), `SurfaceTrackingTab` ([surface_tracking_tab.py](surface_tracking_tab.py)) all mix in [robot_control_mixin.py](robot_control_mixin.py) `RobotControlMixin`, which supplies the *shared* robot-motion UI: single moves, the sequence queue, Z-safety limits, AUT speed cap, and the `Space`-key emergency stop. **Robot safety/motion behavior for those three tabs is edited in one place — the mixin — not per tab.**

**Camera abstraction (add a camera without touching the GUI):** [base_camera.py](base_camera.py) `BaseCamera(ABC)` is the interface; [camera_factory.py](camera_factory.py) holds `_REGISTRY` (name → module path → class name) and lazily imports the chosen SDK only when `create_camera()` is called (so missing SDKs don't break startup). To add a camera: implement `BaseCamera`, append to `_REGISTRY`, and it auto-appears in the GUI combo. Concrete impls: [zivid_camera.py](zivid_camera.py), [realsense_camera.py](realsense_camera.py), [percipio_camera.py](percipio_camera.py).

**KUKA communication is two layers** ([kuka_robot.py](kuka_robot.py)):
- `C3BridgeClient` — raw C3Bridge / KukaVarProxy protocol over **TCP port 7000** (`read_variable` / `write_variable` / `send_motion`).
- `KUKARobot` — high-level API on top: `get_tcp_position()`, and a **20-slot motion queue** (`add_move_ptp` / `add_move_lin` / `..._rel` return a slot index; `move_ptp` / `move_lin` are blocking convenience wrappers), plus `emergency_stop`, `safety_pause/resume`, `set_speed`, `clear_queue`.
- The robot side runs [krl/ext_move.src](krl/ext_move.src) (`+ ext_move.dat`), a KRL program that services that queue. Python enqueues; KRL executes. The queue depth, motion types (PTP/LIN), and the E-stop `RESUME` interrupt are a **contract between `KUKARobot` and `ext_move.src`** — change both together.

**Calibration is pure/Qt-free** ([calibration.py](calibration.py)) — safe to reason about in isolation. Key functions: `tcp_to_homogeneous` / `homogeneous_to_tcp` (KUKA ABC Euler ↔ 4×4), `compute_hand_eye` (runs 5 OpenCV methods → nonlinear refine → greedy outlier removal, picks most consistent), `estimate_pose_from_pointcloud` (Zivid path) with `solvePnP` fallback (RealSense path), `compute_approach_pose` (assumes gripper approaches along **Tool +Z**), `save/load_calibration_result`.

**Optional RF-DETR detector** (bin picking only) — [bin_picking_tab.py](bin_picking_tab.py) lazily `from detector import Detector` at button-press. Configured by env vars `RFDETR_DETECTOR_DIR` (added to `sys.path`) and `RFDETR_MODEL_PATH` (`.engine`/`.onnx`). If absent, the app still starts and every other tab works — only "객체 검출" fails with a guided error. Class mapping is `RFDETR_CLASSES` in that file; the seg model's mask is used when available.

## Conventions & gotchas

- **Calibration modes** track the camera type: Zivid → `pointcloud`, RealSense → `pnp`, `compare` auto-picks. Don't hardcode one path.
- **Coordinates/units:** positions are mm, angles are KUKA `A B C` Euler degrees. Homogeneous transforms are 4×4 numpy. Convert only via the `calibration.py` helpers.
- **User session data** lands in `data/session_*/pose_NNN/` (gitignored). `calibration_result.json` / `intrinsics.json` are also gitignored — never commit generated artifacts.
- Deep algorithm write-ups (Korean) are in [docs/](docs/): `hand_eye_calibration.md`, `bin_picking.md`, `cad_matching.md`, `kuka_communication.md`. Read the matching doc before changing an algorithm.
