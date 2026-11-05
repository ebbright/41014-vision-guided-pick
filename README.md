# Track B: Vision-Guided Pick with a UR3 in MuJoCo

## 1. Project Overview

This project implements an integrated vision-guided robotic picking system using a six-degree-of-freedom UR3 manipulator in the MuJoCo simulation environment.

The robot uses a wrist-mounted RGB-D camera to detect and localise a red cube on a tabletop, plan an approach trajectory and execute a grasp-and-lift operation using a parallel-jaw gripper.

The cube position, orientation, mass and friction are randomised between trials. The system is evaluated under different sensor noise conditions and camera calibration errors to assess its robustness.

The project integrates:

- Robot modelling using Denavit–Hartenberg (DH) parameters.
- Camera intrinsic and hand-eye calibration.
- HSV-based object detection and depth-based localisation.
- Forward and inverse kinematics.
- Position-Based Visual Servoing (PBVS).
- State-space modelling and controllability and observability analysis.
- Kalman filtering for unmeasured state estimation.
- LQR feedback control with computed-torque compensation.
- Randomised experiments and robustness evaluation.

## 2. System Architecture

The simulated system consists of:

- A six-degree-of-freedom UR3 robotic manipulator.
- A wrist-mounted RGB-D camera.
- A parallel-jaw gripper.
- A tabletop workspace.
- A red cube with a side length of 4 cm.

### Integrated processing pipeline

The robot completes the picking task through the following sequence:

1. **Camera calibration:** Estimate camera intrinsics and the camera-to-flange transformation using checkerboard observations and hand-eye calibration.
2. **Object detection:** Segment the red cube using HSV colour information.
3. **Depth localisation:** Back-project depth measurements into three-dimensional coordinates.
4. **Coordinate transformation:** Transform the estimated object position into the robot base frame using calibrated hand-eye transformation and encoder-based forward kinematics.
5. **Target estimation:** Use a Kalman filter to estimate the cube's position and velocity from camera measurements.
6. **Motion planning:** Calculate inverse kinematics for the scan, pre-grasp, grasp and lift positions.
7. **Visual correction:** Apply PBVS correction at the pre-grasp position using updated camera observations.
8. **Feedback control:** Use computed-torque compensation and LQR feedback operating on the estimated robot state.
9. **Grasp verification:** Close the gripper, verify finger closure and lift the cube.
10. **Performance evaluation:** Compare estimated and true states after each trial.

The estimator and controller do not use simulator ground truth. Ground-truth values are only used for post-trial scoring and reporting calibration accuracy.

## 3. Requirements Mapping

| Requirement | Implementation |
|---|---|
| Camera calibration | `calibration.py`: intrinsic calibration using `cv2.calibrateCamera`, outlier rejection and hand-eye calibration using `cv2.calibrateHandEye` with the Park method |
| Depth-based object localisation | `perception.py`: HSV segmentation, depth-image back-projection, top-face extraction, base-frame transformation and yaw estimation |
| Approach planning (PBVS) | `pick_pipeline.py`: pre-grasp planning, re-observation, visual correction, descent, grasp verification and lifting |
| State-space modelling | `control.py`: MuJoCo-based linearised plant model |
| Controllability and observability | `control.py`: rank, conditioning and pole-location analysis |
| State-space joint controller | `control.py`: LQR on the 12-state model, computed-torque compensation and gain scheduling |
| Observer for unmeasured states | `estimation.py`: steady-state joint Kalman observer and target Kalman filter |
| Feedback controller on estimated state | `control.py`: `torque()` uses the estimated state `x_hat` rather than simulator ground truth |
| Robustness evaluation | `run_experiments.py`: randomised cube position, yaw, mass and friction, with four stress conditions |

## 4. Robot Modelling and State-Space Analysis

The UR3 is modelled using standard Denavit–Hartenberg parameters and approximate link masses.

The robot has six joints. Its state vector contains six joint positions and six joint velocities:

\[
x = \begin{bmatrix}q\\\dot q\end{bmatrix}
\]

This produces a 12-state model.

The design model uses computed-torque compensation and is represented by:

\[
\dot{x}=Ax+Bu
\]

The system is discretised using a zero-order hold with a sampling time of 10 ms.

Controllability and observability were evaluated for both the design model and the full torque-input linearisation.

### State-space analysis results

| Measurement | Result |
|---|---:|
| Number of states | 12 |
| Design-model controllability rank | 12/12 |
| Design-model observability rank | 12/12 |
| Full-model controllability rank | 12/12 |
| Full-model observability rank | 12/12 |
| Controllability condition number | 44.48 |
| Observability condition number | 130.36 |
| Maximum closed-loop pole magnitude | 0.975 |
| Maximum observer pole magnitude | 0.761 |
| Maximum open-loop pole magnitude | 1.032 |

Both models are fully controllable and observable at the analysed operating point.

The closed-loop and observer pole magnitudes are below one, indicating stable discrete-time closed-loop and observer dynamics. The full open-loop linearisation has a pole magnitude above one, indicating an unstable open-loop mode.

## 5. Camera Calibration and Object Localisation

Camera calibration was performed using a checkerboard with a 9 × 6 inner-corner pattern and 2 cm squares.

A total of 40 checkerboard views were collected, with 36 retained after outlier rejection.

Intrinsic calibration was performed using OpenCV's `calibrateCamera`. Hand-eye calibration used OpenCV's `calibrateHandEye` with the Park method, followed by nonlinear reprojection refinement.

### Calibration results

| Measurement | Result |
|---|---:|
| Checkerboard views collected | 40 |
| Views retained | 36 |
| Reprojection RMS | 0.479 px |
| Estimated focal length, fx | 582.287 px |
| Estimated focal length, fy | 581.892 px |
| Principal point, cx | 321.835 px |
| Principal point, cy | 238.320 px |
| Hand-eye translation error | 2.046 mm |
| Hand-eye rotation error | 0.222° |

Object localisation combines HSV segmentation and depth-image back-projection to estimate the cube's three-dimensional position.

The estimated position is transformed into the robot base frame using the calibrated camera transformation and encoder-based forward kinematics.

The cube's yaw is estimated from its top-face geometry using a minimum-area rectangle.

## 6. State Estimation

Two Kalman filtering approaches are implemented.

### 6.1 Joint-state observer

The robot encoders measure joint positions but do not directly measure joint velocities.

A steady-state Kalman observer combines encoder measurements with the mathematical state-space model to estimate the unmeasured joint velocities.

These estimated velocities are supplied to the feedback controller, allowing it to use both position and velocity estimates when calculating its control input.

### 6.2 Target-state Kalman filter

A constant-velocity Kalman filter estimates the cube's position and velocity using successive camera observations.

Measurement uncertainty is modelled using range-dependent covariance, with a 2.5 mm lateral uncertainty floor.

The filter is initialised from the first camera observation and incorporates later measurements at the scan and pre-grasp positions.

## 7. Feedback Control and Motion Planning

The robot uses a computed-torque controller with Linear Quadratic Regulator (LQR) feedback.

The controller operates on the estimated state rather than simulator ground truth.

The design uses:

- State weighting: \(Q=\operatorname{diag}(400I,4I)\).
- Input weighting: \(R=0.04I\).
- Sampling time: 10 ms.
- Quintic minimum-jerk joint trajectories.
- Gain scheduling at the beginning of each motion segment.

The controller combines model-based compensation with state feedback to improve trajectory tracking.

### Pick sequence

1. Move from the rest configuration to the scan position.
2. Capture camera observations and estimate the cube pose.
3. Calculate a pre-grasp position 12 cm above the estimated cube centre.
4. Re-observe the cube at the pre-grasp position and apply PBVS correction.
5. Descend towards the grasp position.
6. Close the parallel-jaw gripper and verify the grasp through finger closure.
7. Lift the cube 15 cm above its centre.

The visual correction is applied at the pre-grasp stage. Continuous image-based visual servoing during descent is not implemented.

## 8. Experimental Evaluation

The complete vision-guided pick pipeline was evaluated across 130 simulated trials and five experimental conditions.

### 8.1 Experimental results

| Condition | Trials | Grasp-and-lift success | XY error, median / p95 (mm) | Yaw error, median / p95 (°) |
|---|---:|---:|---:|---:|
| Nominal | 50 | 50/50 | 0.593 / 1.188 | 0.082 / 0.208 |
| Depth noise ×3 | 20 | 20/20 | 1.841 / 2.886 | 0.134 / 0.350 |
| Encoder noise ×4 | 20 | 20/20 | 2.401 / 4.136 | 0.123 / 0.274 |
| Heavy and slippery | 20 | 20/20 | 0.536 / 1.235 | 0.087 / 0.262 |
| Intrinsics 3% error | 20 | 20/20 | 2.383 / 3.173 | 0.115 / 0.231 |
| **Total** | **130** | **130/130** | | |

All 130 simulated trials achieved successful grasp verification and lifting.

### 8.2 Nominal performance

| Performance metric | Result |
|---|---:|
| Median XY localisation error | 0.593 mm |
| 95th percentile XY error | 1.188 mm |
| Median yaw error | 0.082° |
| 95th percentile yaw error | 0.208° |
| Median joint tracking RMS | 0.0051 rad |
| Median velocity-estimation RMS | 0.0256 rad/s |
| Median vertical position error | 1.912 mm |

### 8.3 Robustness testing

Four stress conditions were used to assess system performance:

- **Increased depth noise ×3:** Evaluates the effect of greater uncertainty in camera depth measurements.
- **Increased encoder noise ×4:** Evaluates the effect of less accurate joint position measurements on state estimation and control.
- **Heavy and slippery object:** Evaluates performance under increased cube mass and reduced friction.
- **Intrinsics 3% error:** Evaluates sensitivity to camera focal-length calibration error.

The system achieved 100% simulated grasp-and-lift success across all five conditions. However, localisation and tracking errors varied depending on the disturbance applied.

These results demonstrate robustness under the tested simulated conditions but do not guarantee equivalent performance on physical hardware.

### 8.4 Experimental figures

The `results/figures/` directory contains the following generated figures:

| Figure | Description |
|---|---|
| `fig_success_rates.png` | Success rates across experimental conditions |
| `fig_error_histograms.png` | Localisation and tracking error distributions |
| `fig_showcase_tracking.png` | True and estimated joint velocity tracking |
| `fig_poles.png` | Discrete-time pole analysis |

## 9. Limitations and Future Improvements

The project demonstrates an integrated sensing, estimation and control pipeline in simulation. However, several limitations remain:

- The UR3 model uses approximate link masses and simplified capsule geometry.
- The gripper is represented by a simplified parallel-jaw model.
- The simulated RGB-D camera uses an idealised pinhole model and simplified depth noise.
- Calibration is performed using simulated observations rather than physical hardware.
- The cube dimensions are assumed to be known.
- PBVS correction is applied at the pre-grasp stage rather than continuously throughout descent.
- Controller and observer gain scheduling is performed at the beginning of each motion segment.

Potential future improvements include:

- Physical hardware validation using a real UR3 and RGB-D camera.
- Continuous visual feedback during descent.
- More realistic camera and depth-sensor noise modelling.
- Improved modelling of robot dynamics and gripper behaviour.
- Further evaluation under unmodelled disturbances and changing operating conditions.

## 10. Repository Structure

The main implementation is contained in `track_b_vision_pick/`.

| File | Purpose |
|---|---|
| `ur3_model.py` | UR3 kinematics using standard DH parameters, MJCF model generation and simulator wrapper |
| `calibration.py` | Camera intrinsic and hand-eye calibration |
| `perception.py` | Cube segmentation, depth localisation and yaw estimation |
| `estimation.py` | Joint-state Kalman observer and target Kalman filter |
| `control.py` | State-space model, controllability and observability analysis, LQG joint controller |
| `pick_pipeline.py` | Integrated pick sequence: scan, observe, pre-grasp, PBVS, grasp and lift |
| `run_experiments.py` | Calibration and randomised robustness trials |
| `make_figures.py` | Generates analysis figures and state-space summary |
| `make_report.py` | Generates the PDF project report |
| `results/` | Experimental outputs, calibration results, figures and PDF report |

## 11. Installation and Execution

### 11.1 Software requirements

- Python 3.12
- MuJoCo
- NumPy
- SciPy
- Matplotlib
- OpenCV 4.10.0.84
- wkhtmltopdf (required for PDF report generation)

OpenCV 4.10.0.84 is used because the hand-eye calibration function `calibrateHandEye` is not available in OpenCV 5.x.

### 11.2 Install dependencies

From the project directory, install the required packages:

```powershell
python -m pip install mujoco numpy scipy matplotlib opencv-python-headless==4.10.0.84
```

### 11.3 Run the project

Run these commands from inside the `track_b_vision_pick` directory:

```powershell
python -X utf8 calibration.py
```

Run the complete experiment:

```powershell
python -X utf8 run_experiments.py
```

Run a quick test with three trials per condition:

```powershell
python -X utf8 run_experiments.py --quick
```

Generate figures and state-space analysis:

```powershell
python -X utf8 make_figures.py
```

Generate the PDF report:

```powershell
python -X utf8 make_report.py
```

PDF generation requires `wkhtmltopdf` to be installed and accessible.

### 11.4 Rendering configuration

Offscreen rendering on Linux uses OSMesa:

```bash
apt install libosmesa6
```

On a desktop with a display, MuJoCo can use GLFW by setting:

```powershell
$env:MUJOCO_GL = "glfw"
```

The scripts configure the MuJoCo rendering environment as required by the project.

## 12. Reproducibility

Each experimental condition's scene random number generator is seeded using:

`seed0 + zlib.crc32(condition name) % 100000`

This ensures that scenes are consistent across separate processes.

Python's built-in `hash()` function is deliberately not used because its value is randomised between processes.

Per-trial sensor noise is drawn from the same seeded random stream to support repeatable experimental comparisons.

## 13. Demonstration Video

**Five-minute project demonstration:** [Insert video link here.]

The demonstration will present:

- The UR3 simulation environment and system setup.
- Camera observations and cube localisation.
- The complete vision-guided pick-and-lift pipeline.
- PBVS correction and feedback control.
- State estimation and true-versus-estimated joint velocity tracking.
- Performance under a varied or disturbed trial.

## 14. Team Contributions

**Elizabeth Baker – [25844477]**

Responsible for the vision and perception components of the project. Contributions included developing the camera calibration process, implementing intrinsic and hand-eye calibration, and developing the object detection and depth-based localisation methods. Additional work involved transforming camera observations into the robot base coordinate frame, estimating the cube's position and orientation, and contributing to the integration and testing of the vision-guided picking pipeline.

**Marwa Rahmani – [25303292]**

Responsible for the robot modelling, state estimation and feedback control components of the project. Contributions included developing the UR3 state-space model, performing controllability and observability analysis, implementing the Kalman observer for joint-state estimation, and developing the LQR feedback controller with computed-torque compensation. Additional work involved evaluating controller performance and contributing to the integration and testing of the complete picking system.

**Joint Contributions**

Both team members collaborated on integrating the vision, estimation and control components into the complete UR3 picking pipeline. Joint contributions included debugging, running the 130 experimental trials, analysing quantitative results, evaluating robustness under different operating conditions, preparing the GitHub documentation and producing the final demonstration and project report.

## 15. Generative AI Declaration

ChatGPT was used as a supporting tool during project development. It assisted with explaining programming concepts, troubleshooting Python errors, interpreting technical results, improving code documentation and organising the final project report and GitHub README.

The project team reviewed and tested the implementation and checked the reported experimental results. The team retains responsibility for the accuracy, functionality and integrity of the submitted work.

## 16. References and Acknowledgements

- OpenCV documentation – camera calibration, image processing and hand-eye calibration.
- MuJoCo documentation – physics simulation and robot modelling.
- University of Technology Sydney. (2026). 41014 Industrial Robotics: Subject learning materials and assessment resources.
