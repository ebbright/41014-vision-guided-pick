"""
State-space model, analysis and joint-space LQG controller for the UR3.

Plant model (per the course's state-space formulation, linearised about an operating
configuration q_op):
    x = [q ; qdot]  (12 states),   u = qdd_cmd (6 inputs, acceleration level)
    x_dot = A x + B u,   A = [[0, I], [0, -Minv D]],   B = [[0], [I]],   y = C x = q_enc

The nonlinear gravity/Coriolis terms are cancelled by model-based computed torque,
  tau = bias(q_hat, qdot_hat) + M(q_op) * nu,   nu = qdd_ref + Minv*D*qdot_ref - K (x_hat - x_ref)
so the residual plant is the linear model above. Gains are re-computed (gain scheduling)
at each motion segment's operating point.

Analysis: controllability / observability of the linearised plant (both the
acceleration-input model used for design and the full torque-input linearisation from
MuJoCo), and the discrete poles of the closed loop and of the observer.
"""
import numpy as np
import mujoco
from scipy.linalg import solve_discrete_are, expm
from scipy.signal import cont2discrete

from ur3_model import build_xml, make_checkerboard_png, N_JOINTS

JOINT_DAMPING = 2.0          # matches the MJCF joint damping


def ctrb(A, B):
    n = A.shape[0]
    cols = [B]
    for _ in range(n - 1):
        cols.append(A @ cols[-1])
    return np.hstack(cols)


def obsv(A, C):
    n = A.shape[0]
    rows = [C]
    for _ in range(n - 1):
        rows.append(rows[-1] @ A)
    return np.vstack(rows)


class ArmModel:
    """Nominal (model-based) arm dynamics from the same MJCF as the simulator, used for
    computed torque, mass matrix and linearisation. It never sees the cube or true camera."""

    def __init__(self):
        self.m = mujoco.MjModel.from_xml_string(build_xml(), {"board.png": make_checkerboard_png()})
        self.d = mujoco.MjData(self.m)
        self.qadr = np.array([self.m.joint(f"j{i+1}").qposadr[0] for i in range(N_JOINTS)])
        self.dadr = np.array([self.m.joint(f"j{i+1}").dofadr[0] for i in range(N_JOINTS)])
        self.D = np.diag(self.m.dof_damping[self.dadr])

    def _set(self, q, qd):
        self.d.qpos[self.qadr] = q
        self.d.qvel[self.dadr] = qd
        mujoco.mj_forward(self.m, self.d)

    def bias(self, q, qd):
        """C(q, qd) qd + g(q): the terms computed torque cancels."""
        self._set(q, qd)
        return self.d.qfrc_bias[self.dadr].copy()

    def mass(self, q):
        self._set(q, np.zeros(N_JOINTS))
        M = np.zeros((self.m.nv, self.m.nv))
        mujoco.mj_fullM(self.m, self.d, M)
        return M[np.ix_(self.dadr, self.dadr)]

    def full_linearisation(self, q_op, h):
        """Torque-input linearisation A_t, B_t of the full arm (incl. gravity gradient) by
        finite differences, used for the analysis section only."""
        n = 2 * N_JOINTS
        eps = 1e-6
        M = self.mass(q_op)
        Minv = np.linalg.inv(M)
        dq_bias = np.zeros((N_JOINTS, N_JOINTS))
        dv_bias = np.zeros((N_JOINTS, N_JOINTS))
        for j in range(N_JOINTS):
            e = np.zeros(N_JOINTS); e[j] = eps
            dq_bias[:, j] = (self.bias(q_op + e, np.zeros(N_JOINTS)) - self.bias(q_op - e, np.zeros(N_JOINTS))) / (2 * eps)
            dv_bias[:, j] = (self.bias(q_op, e) - self.bias(q_op, -e)) / (2 * eps)
        A = np.zeros((n, n))
        A[:N_JOINTS, N_JOINTS:] = np.eye(N_JOINTS)
        A[N_JOINTS:, :N_JOINTS] = -Minv @ dq_bias
        A[N_JOINTS:, N_JOINTS:] = -Minv @ (dv_bias + self.D)
        B = np.zeros((n, N_JOINTS))
        B[N_JOINTS:, :] = Minv
        return A, B


def discretise(A, B, Ts):
    C = np.zeros((1, A.shape[0]))
    Ad, Bd, _, _, _ = cont2discrete((A, B, C, np.zeros((1, B.shape[1]))), Ts, method="zoh")
    return Ad, Bd


class StateSpaceJointController:
    """LQG on the linearised arm: LQR state feedback on the estimated state + steady-state
    Kalman observer. The observer estimates the unmeasured joint velocities from encoder
    positions (y = q)."""

    def __init__(self, arm, Ts, q_w=400.0, v_w=4.0, r_w=0.04, sigma_acc=3.0, sigma_enc=0.002):
        self.arm, self.Ts = arm, Ts
        self.q_w, self.v_w, self.r_w = q_w, v_w, r_w
        self.sigma_acc, self.sigma_enc = sigma_acc, sigma_enc
        self.C = np.hstack([np.eye(N_JOINTS), np.zeros((N_JOINTS, N_JOINTS))])
        self.q_op = None

    def set_operating_point(self, q_op):
        """Linearise about q_op and recompute LQR and Kalman gains (gain scheduling)."""
        self.q_op = np.array(q_op, float)
        M = self.arm.mass(self.q_op)
        Minv = np.linalg.inv(M)
        n = N_JOINTS
        A = np.zeros((2 * n, 2 * n))
        A[:n, n:] = np.eye(n)
        A[n:, n:] = -Minv @ self.arm.D
        B = np.zeros((2 * n, n))
        B[n:, :] = np.eye(n)
        self.M_op = M
        self.A_c, self.B_c = A, B
        self.Ad, self.Bd = discretise(A, B, self.Ts)
        Q = np.diag(np.concatenate([np.full(n, self.q_w), np.full(n, self.v_w)]))
        R = np.eye(n) * self.r_w
        P = solve_discrete_are(self.Ad, self.Bd, Q, R)
        self.K = np.linalg.solve(R + self.Bd.T @ P @ self.Bd, self.Bd.T @ P @ self.Ad)
        Qn = self.Bd @ (self.sigma_acc ** 2 * np.eye(n)) @ self.Bd.T + 1e-10 * np.eye(2 * n)
        Rv = (self.sigma_enc ** 2) * np.eye(n)
        Pm = solve_discrete_are(self.Ad.T, self.C.T, Qn, Rv)          # prior error covariance
        self.Lf = Pm @ self.C.T @ np.linalg.inv(self.C @ Pm @ self.C.T + Rv)
        self.Pm = Pm
        return self

    def analysis(self, q_op=None):
        """Controllability / observability and pole locations of the linearised plant."""
        q_op = self.q_op if q_op is None else q_op
        A, B = self.A_c, self.B_c
        out = {}
        Co = ctrb(self.Ad, self.Bd)
        Ob = obsv(self.Ad, self.C)
        out["ctrb_rank_design"] = int(np.linalg.matrix_rank(Co))
        out["obsv_rank_design"] = int(np.linalg.matrix_rank(Ob))
        out["n_states"] = 2 * N_JOINTS
        out["ctrb_cond"] = float(np.linalg.cond(Co))
        out["obsv_cond"] = float(np.linalg.cond(Ob))
        At, Bt = self.arm.full_linearisation(q_op, self.arm.m.opt.timestep)
        Adt, Bdt = discretise(At, Bt, self.Ts)
        out["ctrb_rank_full"] = int(np.linalg.matrix_rank(ctrb(Adt, Bdt)))
        out["obsv_rank_full"] = int(np.linalg.matrix_rank(obsv(Adt, self.C)))
        out["eig_open_loop_full_max_abs"] = float(np.max(np.abs(np.linalg.eigvals(Adt))))
        out["eig_closed_loop_design"] = np.linalg.eigvals(self.Ad - self.Bd @ self.K)
        out["eig_observer_design"] = np.linalg.eigvals(self.Ad - self.Ad @ self.Lf @ self.C)
        out["eig_open_loop_design"] = np.linalg.eigvals(self.Ad)
        return out

    def torque(self, x_hat, x_ref, q_ref_acc, qd_ref):
        """Computed-torque + LQR command from the ESTIMATED state."""
        q_hat, qd_hat = x_hat[:N_JOINTS], x_hat[N_JOINTS:]
        x_ref_vec = np.concatenate([x_ref[:N_JOINTS], x_ref[N_JOINTS:]])
        nu = q_ref_acc + np.linalg.solve(self.M_op, self.arm.D @ qd_ref) - self.K @ (x_hat - x_ref_vec)
        bias = self.arm.bias(q_hat, qd_hat)
        return bias + self.M_op @ nu, nu
