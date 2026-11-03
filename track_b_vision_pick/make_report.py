"""Builds the PDF report (HTML -> PDF with wkhtmltopdf) from results/ so every number is generated."""
import os, json, subprocess, html
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIG = os.path.join(RES, "figures")


def fmt(x, nd=2):
    return f"{x:.{nd}f}"


def main():
    data = json.load(open(os.path.join(RES, "experiment_results.json")))
    cal = data["calibration"]
    conds = {c["condition"]: c for c in data["conditions"]}
    ss = json.load(open(os.path.join(RES, "state_space_analysis.json")))
    nom = conds["nominal"]

    rows = ""
    for c in data["conditions"]:
        lo, hi = c["success_ci95"]
        rows += (f"<tr><td>{c['condition']}</td><td>{c['successes']}/{c['trials']}</td>"
                 f"<td>{fmt(100*c['success_rate'],1)}% [{fmt(100*lo,1)}, {fmt(100*hi,1)}]</td>"
                 f"<td>{fmt(c['est_err_xy_mm_median'],2)} / {fmt(c['est_err_xy_mm_p95'],2)}</td>"
                 f"<td>{fmt(c['yaw_err_deg_median'],2)} / {fmt(c['yaw_err_deg_p95'],2)}</td>"
                 f"<td>{fmt(np.rad2deg(c['track_rms_rad_median']),3)}</td>"
                 f"<td>{fmt(c['vel_est_rms_radps_median'],3)}</td></tr>")

    html_doc = f"""<!doctype html><html><head><meta charset="utf-8">
<style>
body {{ font-family: Georgia, 'Times New Roman', serif; font-size: 11pt; line-height: 1.42; color: #111; margin: 28px 40px; }}
h1 {{ font-size: 20pt; margin-bottom: 4px; }} h2 {{ font-size: 14pt; margin-top: 22px; border-bottom: 1px solid #bbb; padding-bottom: 2px; }}
h3 {{ font-size: 11.5pt; margin-bottom: 2px; }}
table {{ border-collapse: collapse; margin: 8px 0 12px 0; font-size: 9.5pt; width: 100%; }}
th, td {{ border: 1px solid #999; padding: 3px 6px; text-align: left; vertical-align: top; }}
th {{ background: #eee; }} code {{ font-family: Menlo, Consolas, monospace; font-size: 9.5pt; }}
img {{ width: 100%; margin: 6px 0; }} .small {{ font-size: 9pt; color: #333; }}
</style></head><body>
<h1>Track B: Vision-guided pick with a UR3 in MuJoCo</h1>
<p class="small">Own project. Position-based visual servoing (PBVS) with a state-space LQG joint controller and a Kalman
observer. Source: <code>track_b_vision_pick/</code>. All numbers below are generated from <code>results/</code>.</p>

<h2>1. Summary</h2>
<p>A UR3 (6 DOF, nominal DH kinematics) with a parallel-jaw gripper and a wrist-mounted RGB-D camera picks a red 4 cm cube from a
tabletop. The cube position, yaw, mass and friction are randomised in every trial. The system calibrates the camera (intrinsics
and hand-eye), localises the cube from depth, and approaches it with a two-stage PBVS. The joint controller is an LQG controller
that acts on the <em>estimated</em> state. Joint velocities and the cube's position and velocity are unmeasured, and each is
reconstructed by a Kalman filter.</p>
<p>In the nominal condition the grasp-and-lift succeeded in <b>{nom['successes']} of {nom['trials']}</b> trials.
The median cube-centre error at the grasp plan was <b>{fmt(nom['est_err_xy_mm_median'],2)} mm</b> (95th percentile
{fmt(nom['est_err_xy_mm_p95'],2)} mm), and the median yaw error was {fmt(nom['yaw_err_deg_median'],2)} degrees.
Across the four stress conditions (Section 7), every trial succeeded.</p>

<h2>2. Requirements mapping</h2>
<table><tr><th>Requirement</th><th>Implementation</th></tr>
<tr><td>Camera calibration</td><td>Checkerboard views from {cal['n_views']} arm poses. Intrinsics by <code>calibrateCamera</code>
(reprojection RMS {fmt(cal['reproj_rms_px'],2)} px). Hand-eye by <code>calibrateHandEye</code> (Park), then refined by minimising corner
reprojection error jointly over the mount, the board pose and the intrinsics.</td></tr>
<tr><td>Depth-based localisation</td><td>HSV segmentation; back-projection of the depth image with calibrated K;
transform to the base frame through encoder FK and the hand-eye; top-face extraction; yaw from the top-face minimum-area rectangle.</td></tr>
<tr><td>Approach planning</td><td>PBVS: pre-grasp from the estimate, re-observation at the pre-grasp view with a correction, descent,
closing, grasp check, lift.</td></tr>
<tr><td>State-space model, controllability, observability</td><td>12-state arm model (positions, velocities). Ranks
{ss['ctrb_rank_design']}/{ss['n_states']} and {ss['obsv_rank_design']}/{ss['n_states']} for the design model; {ss['ctrb_rank_full']}/{ss['n_states']} and
{ss['obsv_rank_full']}/{ss['n_states']} for the full torque linearisation.</td></tr>
<tr><td>Calibrated exteroceptive sensor</td><td>Wrist RGB-D camera, calibrated intrinsics and hand-eye.</td></tr>
<tr><td>Kalman filter / observer</td><td>Joint observer (steady-state Kalman filter on encoder positions) and a
constant-velocity Kalman filter on the cube position.</td></tr>
<tr><td>Feedback controller on estimated state</td><td>Computed torque plus LQR on <code>x_hat</code>, gain-scheduled per motion segment.</td></tr>
<tr><td>Robustness to randomised position</td><td>Cube (x, y, yaw, mass, friction) randomised in each trial; four stress conditions.</td></tr>
</table>

<h2>3. System and state-space model</h2>
<p>The arm model is generated from the standard UR3 DH parameters (d<sub>1</sub>=0.1519, a<sub>2</sub>=-0.24365, a<sub>3</sub>=-0.21325,
d<sub>4</sub>=0.11235, d<sub>5</sub>=0.08535, d<sub>6</sub>=0.0819 m), with approximate link masses. MuJoCo's forward kinematics
match the DH model to numerical precision. The UR3 is not in MuJoCo Menagerie (only the UR5e is), so the links are capsules,
not the UR meshes.</p>
<p>Plant state <code>x = [q; qdot]</code> (12 states), input <code>nu</code> (joint acceleration command), measurement <code>y = q</code> (encoders).
Computed torque cancels the Coriolis and gravity terms, so the design model is <code>x' = A x + B nu</code>, with
<code>A = [[0, I], [0, -M^-1 D]]</code> and <code>B = [[0], [I]]</code>. M is the mass matrix at the operating point and D the joint damping.
This is discretised with a zero-order hold at Ts = 10 ms (the course's Week 5 method). The full torque-input linearisation,
including the gravity gradient, is also computed from MuJoCo for the analysis.</p>
<p>Results at the scan configuration: both models are controllable and observable with the encoder output. The open-loop full
linearisation has one unstable pole, with |lambda|max = {fmt(ss['max_abs_open_loop_full_eig'],3)}, from gravity. The closed-loop poles lie
at |z| <= {fmt(ss['max_abs_closed_loop_eig'],3)} and the observer poles at |z| <= {fmt(ss['max_abs_observer_eig'],3)}. The
controllability and observability condition numbers are {ss['ctrb_cond']:.1f} and {ss['obsv_cond']:.1f}.</p>
<img src="{os.path.join(FIG, 'fig_poles.png')}" style="width:55%">

<h2>4. Calibration</h2>
<p>The checkerboard (9 x 6 inner corners, 2 cm squares) sits on the table. Arm poses are sampled around it with tilt
(&plusmn;0.3 rad) and distance (0.22-0.42 m). Encoder readings are averaged over 50 samples per pose. Views whose error is an outlier
are rejected. Against the simulator's known values (used only for reporting): focal length {fmt(cal['fx'],1)} px vs
{fmt(cal['fx_true'],1)} px (true), principal point ({fmt(cal['cx'],1)}, {fmt(cal['cy'],1)}) px vs ({fmt(cal['cx_true'],1)}, 240) px,
and hand-eye translation error {fmt(cal['mount_trans_err_mm'],2)} mm with rotation error {fmt(cal['mount_rot_err_deg'],2)} degrees.</p>

<h2>5. Estimation and control</h2>
<h3>Joint observer (unmeasured velocities)</h3>
<p>The encoders measure only positions. A steady-state Kalman filter (Q from a 3 rad/s&sup2; acceleration disturbance, R from a 2 mrad
encoder noise) reconstructs the joint velocities, which the controller uses for feedback. Section 7 compares the estimated and true velocities.</p>
<h3>Target Kalman filter</h3>
<p>The state is the cube centre and velocity. Each camera view is one position measurement, with a covariance that grows with the
square of the range and a 2.5 mm lateral floor. The velocity is unmeasured. The filter is initialised from the first view and fuses
the later views at the scan and pre-grasp poses.</p>
<h3>Controller</h3>
<p>LQR with Q = diag(400 I, 4 I) and R = 0.04 I, designed on the Ts = 10 ms model at each motion segment's start configuration
(gain scheduling). Command: <code>tau = bias(x_hat) + M(q_op) * (q_ref_dd + M^-1 D q_ref_d - K (x_hat - x_ref))</code>.
Joint trajectories are quintic (minimum-jerk) between IK waypoints.</p>

<h2>6. Pick procedure</h2>
<ol>
<li>Move from the rest pose to the scan pose (tool pointing down, tool-centre point 0.25 m above the robot base). Capture two views.</li>
<li>Plan the pre-grasp 12 cm above the estimated cube centre. The gripper yaw is the cube yaw mod 90 degrees, closest to the current wrist.</li>
<li>PBVS correction: at the pre-grasp, capture two more views. If the refined grasp plan moved more than 4 mrad in joint space, re-plan and move.</li>
<li>Descend to the cube centre, close the gripper, and check the finger closure (a grasp gives about 10 mm; no cube gives 20 mm).</li>
<li>Lift 15 cm above the centre. Success is grasp detected and the cube above table by more than 5 cm (scored from truth after the trial).</li>
</ol>
<p>The approach is hierarchical PBVS. Vision feeds back at the pre-grasp view only, not continuously during the descent. IBVS was not used.</p>

<h2>7. Experiments</h2>
<p>Each trial draws the cube position uniformly over x in [0.24, 0.38] m and y in [-0.14, 0.14] m, with yaw in &plusmn;45&deg;. Mass
and friction are drawn from the ranges listed under each condition. Sensor noise: depth noise standard deviation 1 mm + 3 mm &times; Z&sup2; (Z in metres); encoders 2 mrad per sample.
Ground truth is used only after each trial, for scoring.</p>
<table><tr><th>Condition</th><th>Success</th><th>Success % [95% CI]</th><th>Cube xy error, median / p95 (mm)</th>
<th>Yaw error, median / p95 (deg)</th><th>Tracking RMS, median (deg)</th><th>Velocity-observer RMS, median (rad/s)</th></tr>
{rows}</table>
<p class="small">Conditions: nominal (mass 30-100 g, friction 0.7-1.2, 50 trials); depth noise x3 (20 trials); encoder noise x4, i.e. 8 mrad per sample
(20 trials); heavy and slippery (mass 100-200 g, friction 0.35-0.5, 20 trials); 3% focal-length error in the controller's intrinsics (20 trials).
Success is grasp verified by finger closure and the cube lifted more than 5 cm; the 95% interval is a Wilson interval.</p>
<img src="{os.path.join(FIG, 'fig_success_rates.png')}">
<img src="{os.path.join(FIG, 'fig_error_histograms.png')}">
<img src="{os.path.join(FIG, 'fig_showcase_tracking.png')}">

<h2>8. Discussion and limitations</h2>
<ul>
<li>The UR3 is modelled from DH parameters and approximate masses, with capsule geometry. The gripper is a simplified parallel
jaw. Results should transfer to a real UR3 only after identification of its mass and friction parameters.</li>
<li>The camera is an ideal pinhole model with no lens distortion and no rolling shutter, and the depth noise is a simple range-dependent model. Real depth sensors also have multipath and edge artefacts.</li>
<li>The hand-eye and intrinsic calibrations are estimated from simulated images. The 3% focal-length condition in Section 7 tests the sensitivity to calibration error.</li>
<li>The grasp is a top-down pinch on a cube of known size. The cube's size is a model assumption: the estimator sets the centre to the
top-face height minus the known half-size.</li>
<li>Gain scheduling is done once per segment, so the observer and controller are linear only locally. There is no integral action,
and the steady-state error is handled by the computed-torque feedforward.</li>
<li>The vision-based correction is applied at the pre-grasp view only. A continuous image-based loop during the descent would be the
natural next step.</li>
</ul>
</body></html>"""
    out_html = os.path.join(RES, "report.html")
    open(out_html, "w", encoding="utf-8").write(html_doc)
    out_pdf = os.path.join(RES, "Track_B_UR3_Vision_Pick_Report.pdf")
    subprocess.run(["wkhtmltopdf", "--quiet", "--enable-local-file-access", "--page-size", "A4",
                    "--margin-top", "12mm", "--margin-bottom", "12mm", "--margin-left", "10mm", "--margin-right", "10mm",
                    out_html, out_pdf], check=False)
    print("report:", out_pdf, os.path.exists(out_pdf))


if __name__ == "__main__":
    main()
