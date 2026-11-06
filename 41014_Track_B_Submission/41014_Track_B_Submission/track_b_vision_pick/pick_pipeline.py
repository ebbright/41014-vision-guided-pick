"""
Vision-guided pick: position-based visual servoing (PBVS) with a state-space joint controller.

Information flow (no ground truth anywhere in this file):
  camera RGB-D + calibrated intrinsics K + hand-eye T_flange_cam  -> cube 3D pose (perception)
  cube pose measurements                                           -> target Kalman filter
  encoder positions                                                -> joint-state observer (LQG)
  target estimate + nominal kinematics                             -> planned approach and grasp
  joint reference + estimated state                                -> computed torque + LQR

Sequence: rest -> scan pose -> observe -> pre-grasp -> observe & correct (PBVS) -> descend
          -> close gripper -> verify grasp -> lift.
"""
import numpy as np

from ur3_model import (fk_flange, fk_tcp, flange_from_tcp, tool_down_rotation, solve_ik, wrap_angle,
                       SCAN_TCP, TABLE_Z, CUBE_HALF, GRIPPER_OPEN, GRIPPER_CLOSED, N_JOINTS)
from control import ArmModel, StateSpaceJointController
from estimation import JointObserver, TargetKF
from perception import localise_cube, measurement_cov, add_depth_noise

CTRL_DT = 0.01
Q_REST = np.array([0.0, -1.2, 1.2, -1.6, -1.57, 0.0])
SCAN_YAW = 0.0


def quintic(q0, q1, T):
    """Joint-space quintic (minimum jerk) between q0 and q1 over T seconds."""
    dq = wrap_angle(np.asarray(q1) - np.asarray(q0))
    def f(t):
        s = np.clip(t / T, 0.0, 1.0)
        p = 10 * s**3 - 15 * s**4 + 6 * s**5
        dp = (30 * s**2 - 60 * s**3 + 30 * s**4) / T
        ddp = (60 * s - 180 * s**2 + 120 * s**3) / T**2
        return q0 + dq * p, dq * dp, dq * ddp
    return f


def hold(q, T):
    return quintic(q, q, T)


_SCAN_Q = None


def scan_configuration():
    """Joint configuration of the scan pose (solved once, deterministically, from the nominal kinematics)."""
    global _SCAN_Q
    if _SCAN_Q is None:
        seed = np.array([0.0, -1.57, 1.57, -1.57, -1.57, 0.0])
        _SCAN_Q = solve_ik(flange_from_tcp(tool_down_rotation(SCAN_YAW), SCAN_TCP), seed,
                           n_restarts=30, rng=np.random.default_rng(0))
        assert _SCAN_Q is not None, "scan pose unreachable"
    return _SCAN_Q


class PickRobot:
    def __init__(self, sim, calib, rng, cfg=None):
        cfg = {} if cfg is None else cfg
        self.sim, self.rng = sim, rng
        self.K = calib["K"] * np.array([[cfg.get("fx_scale", 1.0), 1, 1], [1, cfg.get("fx_scale", 1.0), 1], [1, 1, 1]])
        self.T_f_c = calib["T_flange_cam"]                       # calibrated hand-eye
        self.arm = ArmModel()
        self.ctl = StateSpaceJointController(self.arm, CTRL_DT, **cfg.get("lqg", {}))
        self.obs = JointObserver(self.ctl)
        self.kf = TargetKF(sigma_acc=cfg.get("target_sigma_acc", 0.005))
        self.enc_sigma = cfg.get("enc_sigma", 0.002)
        self.depth_gain = cfg.get("depth_noise_gain", 1.0)
        self.refine_iters = cfg.get("refine_iters", 2)
        self.t = 0.0
        self.nu_prev = np.zeros(N_JOINTS)
        self.yaw_meas = []                                        # (weight, yaw mod 90deg)
        self.log = dict(t=[], q_true=[], qd_true=[], qd_hat=[], q_hat=[], q_ref=[], grip=[], phase=[], q_std=[])
        self.events = []
        self.keep_log = cfg.get("keep_log", False)

    # ---------------------------------------------------------------- sensing
    def _encoders(self):
        return self.sim.measure_joints(sigma=self.enc_sigma)

    def _finger_positions(self):
        return self.sim.measure_fingers(sigma=0.0005)

    def _vision(self):
        """Capture RGB-D, localise the cube, and fuse the measurement into the target KF."""
        rgb, depth = self.sim.render_rgbd()
        depth = add_depth_noise(depth, self.rng, self.depth_gain)
        q_est = self.obs.x[:N_JOINTS]                              # filtered encoder estimate
        T_base_cam = fk_flange(q_est) @ self.T_f_c
        det = localise_cube(rgb, depth, self.K, T_base_cam)
        if det is None:
            self.events.append((self.t, "vision: cube not found"))
            return None
        R = measurement_cov(det["depth"])
        self.kf.update(det["centre"], R, self.t)
        w = 1.0 / (det["depth"] ** 2 * det["n_top"] + 1e-9)
        self.yaw_meas.append((w, det["yaw"]))
        return det

    def target_yaw(self):
        """Weighted circular mean of the cube yaw measurements (period pi/2)."""
        w = np.array([a for a, _ in self.yaw_meas])
        th = np.array([b for _, b in self.yaw_meas])
        z = np.sum(w * np.exp(1j * 4 * th))
        return float(np.angle(z) / 4) % (np.pi / 2)

    # ---------------------------------------------------------------- control
    def _follow(self, traj, T, grip, phase):
        """Run the joint-space state-space loop for T seconds along traj(t) -> (q, qd, qdd)."""
        n = int(round(T / CTRL_DT))
        for k in range(n):
            t_loc = k * CTRL_DT
            q_ref, qd_ref, qdd_ref = traj(t_loc)
            y = self._encoders()
            x_hat = self.obs.step(y, self.nu_prev)                 # correct + predict
            tau, nu = self.ctl.torque(x_hat, np.concatenate([q_ref, qd_ref]), qdd_ref, qd_ref)
            self.sim.step_control(tau, grip=grip)
            self.nu_prev = nu
            self.t += CTRL_DT

            if self.keep_log:
                self.log["t"].append(self.t)
                self.log["q_true"].append(self.sim.truth_joints())
                self.log["qd_true"].append(self.sim.truth_qdot())
                self.log["qd_hat"].append(x_hat[N_JOINTS:].copy())
                self.log["q_hat"].append(x_hat[:N_JOINTS].copy())
                self.log["q_ref"].append(q_ref.copy())

                # Record estimated joint-position uncertainty
                q_variance = np.maximum(np.diag(self.ctl.Pm)[:N_JOINTS], 0.0)
                q_std = np.sqrt(q_variance)
                self.log["q_std"].append(q_std.copy())

                self.log["grip"].append(grip if grip is not None else 0.0)
                self.log["phase"].append(phase)

    def _start_segment(self):
        """Gain scheduling: re-linearise and re-design the LQG at the current estimate."""
        self.ctl.set_operating_point(self.obs.x[:N_JOINTS])

    def _move(self, q_goal, T, phase, grip=None):
        q0 = self.obs.x[:N_JOINTS].copy()
        self._start_segment()
        self._follow(quintic(q0, q_goal, T), T, grip if grip is not None else GRIPPER_OPEN, phase)

    def _hold(self, T, phase, grip=None):
        q0 = self.obs.x[:N_JOINTS].copy()
        self._follow(hold(q0, T), T, grip if grip is not None else GRIPPER_OPEN, phase)

    # ---------------------------------------------------------------- planning
    def _grasp_yaw_flange(self, cube_yaw, ref_yaw):
        """Gripper yaw equivalent to the cube yaw (mod 90 deg) nearest to a reference."""
        k = np.round((ref_yaw - cube_yaw) / (np.pi / 2))
        return cube_yaw + k * (np.pi / 2)

    def _ik_or_none(self, tcp, yaw, seed):
        T = flange_from_tcp(tool_down_rotation(yaw), tcp)
        return solve_ik(T, seed, n_restarts=12, rng=self.rng)

    def _plan_targets(self):
        c = self.kf.position
        yaw = self.target_yaw()
        # reference: base-frame yaw of the flange x-axis at the current estimate (NOT the wrist joint angle,
        # which is a different quantity and made the equivalent-yaw choice jump by pi between planning rounds)
        R_now = fk_flange(self.obs.x[:N_JOINTS])[:3, :3]
        ref = float(np.arctan2(R_now[1, 0], R_now[0, 0]))
        psi = self._grasp_yaw_flange(yaw, ref)
        seed = self.obs.x[:N_JOINTS].copy()
        q_pre = self._ik_or_none(np.array([c[0], c[1], c[2] + 0.12]), psi, seed)
        q_grasp = self._ik_or_none(np.array([c[0], c[1], c[2] + 0.005]), psi, seed if q_pre is None else q_pre)
        q_lift = self._ik_or_none(np.array([c[0], c[1], c[2] + 0.15]), psi, seed if q_grasp is None else q_grasp)
        return q_pre, q_grasp, q_lift

    # ---------------------------------------------------------------- pick
    def run(self):
        sim = self.sim
        # initialise estimator from encoders at rest
        q_meas = self._encoders()
        self.obs.reset(q_meas)
        self.ctl.set_operating_point(q_meas)
        out = dict(success_reach=True)

        # 1. rest -> scan pose (IK from the nominal scan configuration)
        q_scan = scan_configuration()
        self._move(q_scan, 2.0, "to_scan")
        self._hold(0.5, "scan_hold")
        self._vision()
        self._hold(0.2, "scan_hold")
        self._vision()

        # 2. plan pre-grasp from the estimate, move, then PBVS refinement at the closer view
        q_pre, q_grasp, q_lift = self._plan_targets()
        if q_pre is None or q_grasp is None or q_lift is None:
            out["success_reach"] = False
            out["failure"] = "unreachable plan"
            return out
        self._move(q_pre, 2.0, "to_pregrasp")
        for it in range(self.refine_iters):
            self._hold(0.25, "pbvs_observe")
            det = self._vision()
            self._hold(0.15, "pbvs_observe")
            det2 = self._vision()
            if det is None and det2 is None:
                break
            # PBVS correction: if the refined target moved, re-plan the grasp along the estimate
            q_pre_new, q_grasp_new, q_lift_new = self._plan_targets()
            if q_pre_new is None or q_grasp_new is None:
                break
            delta = np.linalg.norm(q_pre_new - q_pre)
            q_pre, q_grasp, q_lift = q_pre_new, q_grasp_new, q_lift_new
            if delta < 0.004:
                break
            self._move(q_pre, 0.8, f"pbvs_correct_{it}")
        out["estimate_centre"] = self.kf.position.copy()
        # EVALUATION ONLY: truth snapshot at the moment the grasp is planned (not used by any controller)
        out["truth_at_plan_centre"] = self.sim.truth_cube_centre()
        out["truth_at_plan_yaw"] = self.sim.truth_cube_yaw()
        out["estimate_std"] = self.kf.position_std.copy()
        out["estimate_yaw"] = self.target_yaw()
        out["n_updates"] = self.kf.n_updates

        # 3. descend, close, verify, lift
        self._move(q_grasp, 1.2, "descend")
        self._hold(0.2, "descend")
        self._follow(hold(self.obs.x[:N_JOINTS].copy(), 0.8), 0.8, GRIPPER_CLOSED, "close")
        fingers = self._finger_positions()
        out["finger_closure"] = float(np.mean(fingers))
        out["holding"] = bool(0.004 < np.mean(fingers) < 0.017)
        self._move(q_lift, 1.5, "lift", grip=GRIPPER_CLOSED)
        self._hold(0.4, "lift_hold", grip=GRIPPER_CLOSED)
        out["finger_closure_after_lift"] = float(np.mean(self._finger_positions()))
        return out
