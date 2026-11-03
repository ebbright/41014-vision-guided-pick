"""
Estimators.

JointObserver  : steady-state Kalman filter on the arm state x = [q; qdot]. Only the
                 encoder positions are measured, so the joint velocities are an
                 *unmeasured* state that the observer reconstructs.
TargetKF       : Kalman filter on the cube's 3D position and velocity (unmeasured
                 velocity), fed by the depth-based measurements from the camera. Each
                 new view is fused with a range-dependent measurement covariance.
"""
import numpy as np


class JointObserver:
    def __init__(self, controller):
        self.c = controller
        self.x = None
        self.P = None

    def reset(self, q0, qd0=None):
        n = len(q0)
        self.x = np.concatenate([np.asarray(q0, float), np.zeros(n) if qd0 is None else np.asarray(qd0, float)])
        self.P = np.diag(np.concatenate([np.full(n, 0.01 ** 2), np.full(n, 0.5 ** 2)]))

    def step(self, y_meas, nu_applied=None):
        """Measurement update with encoder reading y, then predict one control period ahead
        with the acceleration input that was applied (nu)."""
        C = self.c.C
        Rv = (self.c.sigma_enc ** 2) * np.eye(C.shape[0])
        # correct with the measurement (steady-state gain)
        innov = y_meas - C @ self.x
        self.x = self.x + self.c.Lf @ innov
        # predict
        nu = np.zeros(C.shape[0]) if nu_applied is None else nu_applied
        self.x = self.c.Ad @ self.x + self.c.Bd @ nu
        return self.x.copy()


class TargetKF:
    """Constant-velocity Kalman filter for the cube position (x, y, z) and velocity."""

    def __init__(self, sigma_acc=0.02):
        self.sigma_acc = sigma_acc
        self.x = None
        self.P = None
        self.t = None
        self.n_updates = 0

    def initialise(self, z, R, t):
        self.x = np.concatenate([z, np.zeros(3)])
        self.P = np.block([[R, np.zeros((3, 3))], [np.zeros((3, 3)), np.diag([0.05 ** 2] * 3)]])
        self.t = t
        self.n_updates = 1

    def update(self, z, R, t):
        if self.x is None:
            self.initialise(z, R, t)
            return
        dt = t - self.t
        self.t = t
        F = np.eye(6); F[:3, 3:] = dt * np.eye(3)
        q = self.sigma_acc ** 2
        Q = np.block([[dt ** 4 / 4 * q * np.eye(3), dt ** 3 / 2 * q * np.eye(3)],
                      [dt ** 3 / 2 * q * np.eye(3), dt ** 2 * q * np.eye(3)]])
        x_p = F @ self.x
        P_p = F @ self.P @ F.T + Q
        H = np.hstack([np.eye(3), np.zeros((3, 3))])
        S = H @ P_p @ H.T + R
        K = P_p @ H.T @ np.linalg.inv(S)
        self.x = x_p + K @ (z - H @ x_p)
        self.P = (np.eye(6) - K @ H) @ P_p
        self.n_updates += 1

    @property
    def position(self):
        return self.x[:3].copy()

    @property
    def velocity(self):
        return self.x[3:].copy()

    @property
    def position_std(self):
        return np.sqrt(np.diag(self.P)[:3])
