# Track B: Vision-guided pick with a UR3 in MuJoCo

A UR3 with a wrist-mounted RGB-D camera detects a red cube on a tabletop, plans an approach,
and executes a pick. The cube position is randomised between trials, and the system is tested
for robustness under sensor noise, low friction and miscalibrated intrinsics.

## Requirements mapping

| Requirement | Where it is implemented |
|---|---|
| Camera calibration | `calibration.py`: intrinsics (`cv2.calibrateCamera`, outlier rejection) and hand-eye (`cv2.calibrateHandEye`, Park), refined by nonlinear reprojection minimisation of the mount and board pose |
| Depth-based object localisation | `perception.py`: HSV segmentation, back-projection of the depth image, top-face extraction, base-frame transform through calibrated hand-eye and encoder-based FK, yaw from the top-face outline |
| Approach planning (PBVS) | `pick_pipeline.py`: pre-grasp from the estimate, re-observation and correction (PBVS), descent, grasp verification, lift |
| State-space model, controllability, observability | `control.py` (`ArmModel`, `StateSpaceJointController.analysis`): linearised plant from MuJoCo, rank and conditioning tests, pole locations |
| State-space joint controller | `control.py`: LQR on the 12-state model, computed torque, gain scheduling per motion segment |
| Observer for unmeasured states | `estimation.py`: `JointObserver` (steady-state Kalman filter, joint velocities from encoder positions) and `TargetKF` (cube position and velocity from camera measurements) |
| Feedback controller on estimated state | `control.py` `torque()` uses `x_hat` from the observer, never the true state |
| Robustness | `run_experiments.py`: randomised cube position, yaw, mass and friction; four stress conditions |

The estimator and controller never read ground truth. The functions `truth_*` in `ur3_model.py`
are only called for scoring, after the trial has finished, or for reporting calibration accuracy.

## Files

```
ur3_model.py        UR3 kinematics (standard DH), MJCF model generation, simulator wrapper
calibration.py      camera calibration (intrinsics + hand-eye), calibration checks
perception.py       cube segmentation, depth localisation, yaw estimation
estimation.py       joint-state Kalman observer, target Kalman filter
control.py          state-space model, controllability/observability analysis, LQG joint controller
pick_pipeline.py    the pick sequence (scan -> observe -> pre-grasp -> PBVS -> grasp -> lift)
run_experiments.py  calibration + randomised robustness trials (writes results/)
make_figures.py     figures and the state-space summary (reads results/)
results/            outputs of the last run
```

## Running

Python 3.10+ with:

```
pip install mujoco numpy scipy matplotlib "opencv-python-headless==4.10.0.84"
```

OpenCV 4.10 is needed because the hand-eye solver (`calibrateHandEye`) is not in OpenCV 5.x.
Offscreen rendering uses OSMesa on Linux (`apt install libosmesa6`); the scripts set
`MUJOCO_GL=osmesa`. On a desktop with a display, set `MUJOCO_GL=glfw` before running.

```
python3 calibration.py           # calibration only, writes results/calibration*.{npz,json}
python3 run_experiments.py       # full experiment: 50 nominal + 4 x 20 stress trials
python3 run_experiments.py --quick   # 3 trials per condition, smoke test
python3 make_figures.py          # figures + results/state_space_analysis.json
python3 make_report.py           # results/Track_B_UR3_Vision_Pick_Report.pdf (needs wkhtmltopdf)
```

## Final results

| Condition | Trials | Grasp-and-lift success | Cube xy error, median / p95 (mm) | Yaw error, median (deg) |
|---|---|---|---|---|
| nominal | 50 | 50/50 | 0.38 / 1.16 | 0.08 |
| depth_noise_x3 | 20 | 20/20 | 1.26 / 2.23 | 0.11 |
| encoder_noise_x4 | 20 | 20/20 | 2.20 / 3.46 | 0.13 |
| heavy_slippery | 20 | 20/20 | 0.48 / 0.82 | 0.07 |
| intrinsics_3pct_error | 20 | 20/20 | 2.11 / 2.64 | 0.06 |

Total: 130/130 successful trials. The 95% Wilson lower bound on the success rate is about 97%.

Calibration (simulator ground truth, used for reporting only): reprojection RMS 0.35 px,
focal length error 0.2%, hand-eye/mount error 0.46 mm and 0.11 degrees.

State-space analysis: controllability and observability rank 12/12 for both the design model
and the full linearisation; closed-loop, observer and open-loop pole magnitudes are 0.975,
0.761 and 1.032 respectively.

## Reproducibility

Each condition's scene RNG is seeded with `seed0 + zlib.crc32(condition name) % 100000`, so the
scenes are identical across processes. (Python's `hash()` is randomised per process and was
deliberately not used.) Per-trial sensor noise is drawn from the same seeded stream.
