"""
UR3 manipulator + parallel gripper + wrist RGB-D camera in MuJoCo.

Everything that the *estimator/controller* is allowed to know (nominal DH
kinematics, camera model class) is defined here, and everything that is
ground truth (true cube pose, true camera mount) is only exposed through
explicitly named `truth_*` helpers used by the evaluation code.
"""
import os
import sys


if sys.platform == "win32":
    os.environ["MUJOCO_GL"] = "glfw"
    os.environ.pop("PYOPENGL_PLATFORM", None)
else:
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

import numpy as np
import mujoco
import cv2

# ----------------------------------------------------------------------------
# Nominal UR3 kinematics (standard DH, Universal Robots CB3 datasheet values)
# ----------------------------------------------------------------------------
DH_D = np.array([0.15190, 0.0, 0.0, 0.11235, 0.08535, 0.08190])
DH_A = np.array([0.0, -0.24365, -0.21325, 0.0, 0.0, 0.0])
DH_ALPHA = np.array([np.pi / 2, 0.0, 0.0, np.pi / 2, -np.pi / 2, 0.0])
LINK_MASS = np.array([2.00, 3.42, 1.26, 0.80, 0.80, 0.35])   # kg, approx. UR3 values
GRIPPER_MASS = 0.40
N_JOINTS = 6

TCP_OFFSET = 0.10          # flange -> fingertip centre along flange z [m]
TABLE_Z = -0.10            # table top height in the world frame [m]
CUBE_HALF = 0.02           # target cube half-size [m]
CUBE_MASS = 0.05           # nominal cube mass [kg]
CUBE_FRICTION = 1.0        # nominal sliding friction

# Wrist camera (nominal design; the *real* mount is perturbed in the sim, see Sim)
IMG_H, IMG_W = 480, 640
FOVY_DEG = 45.0
CAM_MOUNT_POS = np.array([0.06, 0.0, 0.0])   # in flange frame
CAM_MOUNT_NOMINAL = np.eye(4)

GRIPPER_OPEN = 0.0         # finger slide offset when open
GRIPPER_CLOSED = 0.02      # finger slide target when closing (contact stops it)

# Calibration board (planar checkerboard on the table, used only in calibration)
BOARD_CENTRE = np.array([0.20, -0.33, TABLE_Z + 0.002])
BOARD_SQUARES = (10, 7)     # squares (columns, rows) -> 9 x 6 inner corners
BOARD_SQUARE_M = 0.02
BOARD_PX_PER_SQUARE = 70

# Ranges where the cube is allowed to be placed
CUBE_X_RANGE = (0.24, 0.38)
CUBE_Y_RANGE = (-0.14, 0.14)


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def dh_matrix(theta, d, a, alpha):
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([[ct, -st * ca, st * sa, a * ct],
                     [st, ct * ca, -ct * sa, a * st],
                     [0, sa, ca, d],
                     [0, 0, 0, 1.0]])


def fk_flange(q):
    """Nominal forward kinematics: 4x4 pose of the flange in the base frame."""
    T = np.eye(4)
    for i in range(N_JOINTS):
        T = T @ dh_matrix(q[i], DH_D[i], DH_A[i], DH_ALPHA[i])
    return T


def fk_tcp(q):
    T = fk_flange(q)
    return T[:3, 3] + T[:3, 2] * TCP_OFFSET


def pose_tool_down(xyz_tcp, yaw):
    """Flange pose (4x4) with the tool pointing straight down and given yaw."""
    z = np.array([0.0, 0.0, -1.0])
    x = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    y = np.cross(z, x)
    R = np.column_stack([x, y, z])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = xyz_tcp - R[:, 2] * TCP_OFFSET
    return T


def jacobian_flange(q):
    """Analytic geometric Jacobian (6x6) of the flange pose w.r.t. the joints, from the DH chain."""
    J = np.zeros((6, N_JOINTS))
    T = np.eye(4)
    zs, ps = [], []
    for i in range(N_JOINTS):
        zs.append(T[:3, 2].copy()); ps.append(T[:3, 3].copy())          # joint i axis = z_{i-1}
        T = T @ dh_matrix(q[i], DH_D[i], DH_A[i], DH_ALPHA[i])
    pe = T[:3, 3]
    for i in range(N_JOINTS):
        J[:3, i] = np.cross(zs[i], pe - ps[i])
        J[3:, i] = zs[i]
    return J, T


def ik_flange(T_target, q0, iters=200, tol=1e-6, rng=None):
    """Damped-least-squares IK on the nominal kinematics (analytic Jacobian). Returns (q, ok)."""
    q = np.array(q0, dtype=float)
    lam = 1e-2
    e = np.ones(6)
    for _ in range(iters):
        J, T = jacobian_flange(q)
        ep = T_target[:3, 3] - T[:3, 3]
        eR = T_target[:3, :3] @ T[:3, :3].T
        er = np.array([eR[2, 1] - eR[1, 2], eR[0, 2] - eR[2, 0], eR[1, 0] - eR[0, 1]]) / 2.0
        e = np.concatenate([ep, er])
        if np.linalg.norm(e) < tol:
            return wrap_angle(q), True
        step = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(6), e)
        q = q + step
    return wrap_angle(q), bool(np.linalg.norm(e) < 1e-4)


def wrap_angle(q):
    return (np.asarray(q) + np.pi) % (2 * np.pi) - np.pi


# Scan pose used at the start of every trial (tool down, above the workspace), found by IK.
SCAN_TCP = np.array([0.30, 0.0, 0.25])


def flange_from_tcp(R_tool, p_tcp):
    """Flange pose (4x4) from a desired tool orientation (3x3) and TCP position."""
    T = np.eye(4)
    T[:3, :3] = R_tool
    T[:3, 3] = np.asarray(p_tcp) - R_tool[:, 2] * TCP_OFFSET
    return T


def tool_down_rotation(yaw):
    z = np.array([0.0, 0.0, -1.0])
    x = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    return np.column_stack([x, np.cross(z, x), z])


def solve_ik(T_target, q_seed, n_restarts=20, rng=None):
    """IK with random restarts; returns the solution closest to q_seed."""
    rng = np.random.default_rng() if rng is None else rng
    best, best_d = None, np.inf
    seeds = [np.asarray(q_seed, float)] + [np.asarray(q_seed, float) + rng.normal(0, 0.8, N_JOINTS)
                                           for _ in range(n_restarts)]
    if not all(np.isfinite(s).all() for s in seeds):
        return None
    for s in seeds:
        q, ok = ik_flange(T_target, s)
        if ok:
            d = np.linalg.norm(wrap_angle(q - q_seed))
            if d < best_d:
                best, best_d = q, d
    return best


# ----------------------------------------------------------------------------
# MJCF generation
# ----------------------------------------------------------------------------
def _quat_rx(a):
    return f"{np.cos(a/2):.6f} {np.sin(a/2):.6f} 0 0"


def make_checkerboard_png():
    """Checkerboard texture for the calibration board, returned as PNG bytes."""
    nx, ny = BOARD_SQUARES
    px = BOARD_PX_PER_SQUARE
    img = np.zeros((ny * px, nx * px), np.uint8)
    for r in range(ny):
        for c in range(nx):
            if (r + c) % 2 == 0:
                img[r*px:(r+1)*px, c*px:(c+1)*px] = 255
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def build_xml(cam_mount_offset=CAM_MOUNT_POS):
    links_xml = []
    # body_i carries joint i. Its child (body_{i+1}) sits at (a_i, 0, d_i) with Rx(alpha_i)
    child_pos = [(DH_A[i], 0.0, DH_D[i]) for i in range(N_JOINTS)]
    child_quat = [_quat_rx(DH_ALPHA[i]) for i in range(N_JOINTS)]

    def link_block(i, indent):
        # geometry of link i = segment from body_i origin to its child offset
        p = np.array(child_pos[i])
        L = np.linalg.norm(p) if np.linalg.norm(p) > 1e-6 else 0.03
        r = 0.045 if i < 3 else 0.035
        m = LINK_MASS[i]
        if np.linalg.norm(p) > 1e-6:
            com = p / 2
            Ixx = m * (3 * r**2 + L**2) / 12
            ine = f"{Ixx:.6f} {Ixx:.6f} {0.5*m*r**2:.6f}"
            geom = (f'<geom type="capsule" fromto="0 0 0 {p[0]:.5f} {p[1]:.5f} {p[2]:.5f}" '
                    f'size="{r:.4f}" material="arm" contype="0" conaffinity="0"/>')
        else:
            com = np.zeros(3)
            ine = f"{m*0.01:.6f} {m*0.01:.6f} {m*0.01:.6f}"
            geom = f'<geom type="sphere" size="{r:.4f}" material="arm" contype="0" conaffinity="0"/>'
        inertial = (f'<inertial pos="{com[0]:.5f} {com[1]:.5f} {com[2]:.5f}" mass="{m:.4f}" '
                    f'diaginertia="{ine}"/>')
        return geom, inertial

    def body_chain(i):
        # i: 0..5 are the joint bodies; 6 is the flange
        if i == N_JOINTS:
            return flange_xml
        geom, inertial = link_block(i, "")
        pos = "0 0 0" if i == 0 else f"{child_pos[i-1][0]:.5f} {child_pos[i-1][1]:.5f} {child_pos[i-1][2]:.5f}"
        quat = "1 0 0 0" if i == 0 else child_quat[i-1]
        inner = body_chain(i + 1)
        return (f'<body name="link{i+1}" pos="{pos}" quat="{quat}">'
                f'<joint name="j{i+1}" type="hinge" axis="0 0 1" range="-6.2832 6.2832" damping="2.0" armature="0.05"/>'
                f'{inertial}{geom}\n{inner}</body>')

    gripper_xml = (
        f'<body name="gripper" pos="0 0 0">'
        f'<inertial pos="0 0 0.02" mass="{GRIPPER_MASS}" diaginertia="0.0008 0.0008 0.0008"/>'
        f'<geom type="box" size="0.025 0.045 0.015" pos="0 0 0.03" material="gripper" contype="0" conaffinity="0"/>'
        f'<body name="finger_l" pos="0 0.034 0.04">'
        f'<joint name="finger_l" type="slide" axis="0 -1 0" range="0 0.02" damping="5"/>'
        f'<inertial pos="0 0 0.05" mass="0.05" diaginertia="0.00005 0.00005 0.00002"/>'
        f'<geom name="pad_l" type="box" size="0.006 0.004 0.04" pos="0 0 0.06" material="gripper" friction="1.2 0.02 0.001" '
        f'contype="1" conaffinity="1"/></body>'
        f'<body name="finger_r" pos="0 -0.034 0.04">'
        f'<joint name="finger_r" type="slide" axis="0 1 0" range="0 0.02" damping="5"/>'
        f'<inertial pos="0 0 0.05" mass="0.05" diaginertia="0.00005 0.00005 0.00002"/>'
        f'<geom name="pad_r" type="box" size="0.006 0.004 0.04" pos="0 0 0.06" material="gripper" friction="1.2 0.02 0.001" '
        f'contype="1" conaffinity="1"/></body>'
        f'</body>')

    cam_pos = " ".join(f"{v:.4f}" for v in cam_mount_offset)
    flange_xml = (
        f'<body name="flange" pos="{child_pos[5][0]:.5f} {child_pos[5][1]:.5f} {child_pos[5][2]:.5f}" quat="1 0 0 0">'
        f'<inertial pos="0 0 0.02" mass="0.2" diaginertia="0.0004 0.0004 0.0004"/>'
        f'<geom type="cylinder" size="0.035 0.02" pos="0 0 -0.01" material="arm" contype="0" conaffinity="0"/>'
        f'{gripper_xml}'
        f'<body name="wrist_cam_mount" pos="{cam_pos}" quat="1 0 0 0">'
        f'<geom type="box" size="0.012 0.02 0.012" material="cam" contype="0" conaffinity="0"/>'
        # MuJoCo cameras look along -z. Flange z points at the table, so the camera frame is
        # x = flange x, y = -flange y, z = -flange z (xyaxes "1 0 0  0 -1 0").
        f'<camera name="wrist_cam" pos="0 0 0" xyaxes="1 0 0 0 -1 0" fovy="{FOVY_DEG}"/>'
        f'</body></body>')
    # Note: body_chain(5) returns link6 which contains the flange; we need the flange as child of link6.
    chain = body_chain(0)

    cube_density = CUBE_MASS / (2 * CUBE_HALF) ** 3
    xml = f"""
<mujoco model="ur3_visual_pick">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" integrator="implicitfast" cone="elliptic" impratio="10"/>
  <visual><global offwidth="{IMG_W}" offheight="{IMG_H}"/><quality shadowsize="1024"/></visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.35 0.35 0.38" rgb2="0.30 0.30 0.33" width="300" height="300"/>
    <texture name="board_tex" type="2d" file="board.png"/>
    <material name="board" texture="board_tex" texrepeat="1 1" reflectance="0"/>
    <material name="grid" texture="grid" texrepeat="4 4"/>
    <material name="table" rgba="0.55 0.45 0.35 1" specular="0.05"/>
    <material name="arm" rgba="0.75 0.75 0.78 1" specular="0.1"/>
    <material name="gripper" rgba="0.25 0.25 0.28 1"/>
    <material name="cam" rgba="0.1 0.1 0.6 1"/>
    <material name="cube" rgba="0.9 0.08 0.08 1"/>
  </asset>
  <worldbody>
    <light pos="0.5 0 1.5" dir="0 0 -1" diffuse="0.45 0.45 0.45" specular="0.05 0.05 0.05" ambient="0.25 0.25 0.25"/>
    <light pos="-0.5 -1 1.2" dir="0.3 0.6 -1" diffuse="0.2 0.2 0.2" specular="0 0 0"/>
    <geom name="floor" type="plane" size="2 2 0.1" pos="0 0 -0.35" material="grid"/>
    <geom name="pedestal" type="cylinder" size="0.09 0.1" pos="0 0 -0.05" material="arm" contype="0" conaffinity="0"/>
    <geom name="table_top" type="box" size="0.45 0.5 0.05" pos="0.35 0 -0.15" material="table" friction="1.0 0.005 0.0001"/>
    <body name="calib_board" pos="{BOARD_CENTRE[0]:.4f} {BOARD_CENTRE[1]:.4f} {BOARD_CENTRE[2]:.4f}">
      <geom type="box" size="{BOARD_SQUARES[0]*BOARD_SQUARE_M/2:.4f} {BOARD_SQUARES[1]*BOARD_SQUARE_M/2:.4f} 0.002" material="board" contype="0" conaffinity="0"/>
    </body>
    <body name="base_link" pos="0 0 0">
      {chain}
    </body>
    <body name="cube" pos="0.30 0 {TABLE_Z + CUBE_HALF:.4f}">
      <freejoint name="cube_free"/>
      <geom name="cube_geom" type="box" size="{CUBE_HALF} {CUBE_HALF} {CUBE_HALF}" density="{cube_density:.3f}" material="cube" friction="{CUBE_FRICTION} 0.005 0.0001"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="m1" joint="j1" ctrlrange="-56 56" forcerange="-56 56"/>
    <motor name="m2" joint="j2" ctrlrange="-56 56" forcerange="-56 56"/>
    <motor name="m3" joint="j3" ctrlrange="-28 28" forcerange="-28 28"/>
    <motor name="m4" joint="j4" ctrlrange="-12 12" forcerange="-12 12"/>
    <motor name="m5" joint="j5" ctrlrange="-12 12" forcerange="-12 12"/>
    <motor name="m6" joint="j6" ctrlrange="-12 12" forcerange="-12 12"/>
    <position name="grip_l" joint="finger_l" kp="300" ctrlrange="0 0.02" forcerange="-25 25"/>
    <position name="grip_r" joint="finger_r" kp="300" ctrlrange="0 0.02" forcerange="-25 25"/>
  </actuator>
</mujoco>
"""
    return xml


class Sim:
    """Physics + rendering. The *controller side* only calls measure_* / render_*;
    truth_* helpers exist for evaluation and calibration checks only."""

    CTRL_DT = 0.01                # 100 Hz control / vision loop
    N_SUB = 5                     # physics dt = 0.002 s

    def __init__(self, seed=0, cam_mount_offset=None, render=True):
        self.rng = np.random.default_rng(seed)
        # the *true* camera mount is slightly different from the nominal design (unknown to the estimator)
        self._true_mount = np.array(CAM_MOUNT_POS if cam_mount_offset is None else cam_mount_offset)
        xml = build_xml(self._true_mount)
        assets = {"board.png": make_checkerboard_png()}
        self.model = mujoco.MjModel.from_xml_string(xml, assets)
        self.data = mujoco.MjData(self.model)
        self.cam_id = self.model.camera("wrist_cam").id
        self.cube_body = self.model.body("cube").id
        self.cube_geom = self.model.geom("cube_geom").id
        self._cube_qadr = self.model.joint("cube_free").qposadr[0]
        self._joint_qadr = np.array([self.model.joint(f"j{i+1}").qposadr[0] for i in range(N_JOINTS)])
        self._joint_dadr = np.array([self.model.joint(f"j{i+1}").dofadr[0] for i in range(N_JOINTS)])
        self._finger_qadr = [self.model.joint("finger_l").qposadr[0], self.model.joint("finger_r").qposadr[0]]
        self.renderer = mujoco.Renderer(self.model, IMG_H, IMG_W) if render else None
        self.nominal_mass = self.model.body_mass[self.cube_body]
        self.nominal_inertia = self.model.body_inertia[self.cube_body].copy()
        self.reset()

    # ----- actuation / sensing -------------------------------------------------
    def reset(self, q=None):
        mujoco.mj_resetData(self.model, self.data)
        if q is not None:
            self.data.qpos[self._joint_qadr] = q
        self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def set_cube(self, x, y, yaw, mass=None, friction=None):
        self.data.qpos[self._cube_qadr:self._cube_qadr + 3] = [x, y, TABLE_Z + CUBE_HALF + 1e-4]
        self.data.qpos[self._cube_qadr + 3:self._cube_qadr + 7] = [np.cos(yaw/2), 0, 0, np.sin(yaw/2)]
        self.data.qvel[:] = 0
        m = self.nominal_mass if mass is None else mass
        self.model.body_mass[self.cube_body] = m
        self.model.body_inertia[self.cube_body] = self.nominal_inertia * (m / self.nominal_mass)
        self.model.geom_friction[self.cube_geom, 0] = CUBE_FRICTION if friction is None else friction
        mujoco.mj_forward(self.model, self.data)

    def measure_joints(self, sigma=0.002):
        """Encoder reading (noisy joint angles). This is the only proprioceptive measurement."""
        return self.data.qpos[self._joint_qadr].copy() + self.rng.normal(0, sigma, N_JOINTS)

    def step_control(self, tau, grip=None, n=None):
        """Apply joint torques (and optional gripper target) for one control period."""
        self.data.ctrl[:N_JOINTS] = tau
        if grip is not None:
            self.data.ctrl[N_JOINTS:N_JOINTS + 2] = grip
        for _ in range(self.N_SUB if n is None else n):
            mujoco.mj_step(self.model, self.data)

    def measure_fingers(self, sigma=0.0005):
        """Finger slide encoders (mean opening measure is used to detect a grasp)."""
        f = np.array([self.data.qpos[a] for a in self._finger_qadr])
        return f + self.rng.normal(0, sigma, 2)

    # ----- camera ---------------------------------------------------------------
    def render_rgb(self):
        self.renderer.disable_depth_rendering()
        self.renderer.update_scene(self.data, camera=self.cam_id)
        return self.renderer.render().copy()

    def render_rgbd(self):
        """Returns (rgb uint8 HxWx3, depth float32 HxW in metres). Pure sensor output."""
        self.renderer.disable_depth_rendering()
        self.renderer.update_scene(self.data, camera=self.cam_id)
        rgb = self.renderer.render().copy()
        self.renderer.enable_depth_rendering()
        self.renderer.update_scene(self.data, camera=self.cam_id)
        depth = self.renderer.render().copy()
        self.renderer.disable_depth_rendering()
        return rgb, depth

    # ----- ground truth (EVALUATION ONLY) --------------------------------------
    def truth_cube_centre(self):
        return self.data.xpos[self.cube_body].copy()

    def truth_cube_yaw(self):
        R = self.data.xmat[self.cube_body].reshape(3, 3)
        return float(np.arctan2(R[1, 0], R[0, 0]))

    def truth_joints(self):
        return self.data.qpos[self._joint_qadr].copy()

    def truth_cam_pose(self):
        T = np.eye(4)
        T[:3, :3] = self.data.cam_xmat[self.cam_id].reshape(3, 3)
        T[:3, 3] = self.data.cam_xpos[self.cam_id]
        return T

    def truth_qdot(self):
        return self.data.qvel[self._joint_dadr].copy()

    def truth_fingers(self):
        return np.array([self.data.qpos[a] for a in self._finger_qadr])


def intrinsics_from_fovy():
    """Nominal pinhole intrinsics implied by the MuJoCo camera (used only to *initialise* the calibration)."""
    fy = (IMG_H / 2) / np.tan(np.deg2rad(FOVY_DEG) / 2)
    return np.array([[fy, 0, IMG_W / 2], [0, fy, IMG_H / 2], [0, 0, 1.0]])
