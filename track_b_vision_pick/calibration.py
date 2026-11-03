"""
Camera calibration for the wrist RGB-D camera.

1. Intrinsics: a checkerboard is imaged from many arm poses; cv2.calibrateCamera
   estimates fx, fy, cx, cy (and reports the reprojection RMS).
2. Hand-eye: for each pose, the flange pose comes from encoder readings + nominal DH
   kinematics and the board pose from PnP. cv2.calibrateHandEye (Tsai) returns the
   fixed transform T_flange_cam (the camera mount), which the estimator then uses.

Only images and joint encoders are used. The true mount / intrinsics are used
solely to *report* calibration accuracy.
"""
import numpy as np
import cv2

from ur3_model import (Sim, BOARD_SQUARES, BOARD_SQUARE_M, fk_flange, flange_from_tcp,
                       tool_down_rotation, solve_ik, rot_x, rot_y, SCAN_TCP, BOARD_CENTRE,
                       intrinsics_from_fovy, IMG_H, IMG_W, wrap_angle)

PATTERN = (BOARD_SQUARES[0] - 1, BOARD_SQUARES[1] - 1)     # inner corners (9, 6)


def board_object_points():
    nx, ny = PATTERN
    pts = np.zeros((nx * ny, 3), np.float32)
    grid = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
    pts[:, :2] = grid * BOARD_SQUARE_M
    return pts


def _rot_about(R, rx, ry):
    return R @ rot_x(rx) @ rot_y(ry)


def calibration_poses(rng, n_views=40):
    """Random tool-down poses that look at the calibration board (with tilt, so the
    rotation axes of the hand-eye problem are not all parallel)."""
    poses = []
    seed_q = solve_ik(flange_from_tcp(tool_down_rotation(0.0), SCAN_TCP), np.array([0, -1.57, 1.57, -1.57, -1.57, 0.0]))
    target_xy = BOARD_CENTRE[:2]
    attempts = 0
    while len(poses) < n_views and attempts < 6000:
        attempts += 1
        p = np.array([target_xy[0] + rng.uniform(-0.08, 0.08),
                      target_xy[1] + rng.uniform(-0.07, 0.07),
                      BOARD_CENTRE[2] + rng.uniform(0.22, 0.42)])
        R = _rot_about(tool_down_rotation(rng.uniform(-0.6, 0.6)), rng.uniform(-0.30, 0.30), rng.uniform(-0.30, 0.30))
        q = solve_ik(flange_from_tcp(R, p), seed_q, rng=rng)
        if q is None:
            continue
        poses.append(q)
    return poses


def collect_views(sim, n_views=40, rng=None, enc_sigma=0.002, n_avg=50):
    """Move the arm (kinematically, images only) to calibration poses and detect the board."""
    rng = np.random.default_rng(0) if rng is None else rng
    obj = board_object_points()
    views = []
    for q_true in calibration_poses(rng, n_views * 3):
        if len(views) >= n_views:
            break
        sim.reset(q_true)
        rgb = sim.render_rgb()
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        found, corners = cv2.findChessboardCornersSB(gray, PATTERN, cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE)
        if not found:
            continue
        corners = corners.reshape(-1, 2).astype(np.float32)
        # encoder readings averaged over n_avg samples while the arm is held still (standard procedure)
        q_meas = q_true + rng.normal(0, enc_sigma / np.sqrt(n_avg), 6)
        views.append(dict(q=q_meas, corners=corners))
    return views, obj


def calibrate(views, obj, img_size=(IMG_W, IMG_H)):
    """Intrinsics + hand-eye from the collected views. Returns a dict of estimates."""
    K0 = intrinsics_from_fovy().astype(np.float64)
    obj_pts = [obj for _ in views]
    img_pts = [v["corners"].reshape(-1, 1, 2) for v in views]
    flags = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3
             | cv2.CALIB_ZERO_TANGENT_DIST)
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, img_size, K0.copy(), np.zeros(5), flags=flags)
    # outlier rejection: drop views whose mean reprojection error is far above the rest
    errs = []
    for o, i, r, t in zip(obj_pts, img_pts, rvecs, tvecs):
        p, _ = cv2.projectPoints(o, r, t, K, np.zeros(5))
        errs.append(float(np.sqrt(((p.reshape(-1, 2) - i.reshape(-1, 2)) ** 2).sum(1)).mean()))
    keep = [k for k, e in enumerate(errs) if e < max(3 * np.median(errs), 0.5)]
    if len(keep) < len(views):
        views = [views[k] for k in keep]
        obj_pts = [obj for _ in views]; img_pts = [v["corners"].reshape(-1, 1, 2) for v in views]
        rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, img_size, K.copy(), np.zeros(5), flags=flags)

    R_g2b, t_g2b, R_t2c, t_t2c = [], [], [], []
    for v, rvec, tvec in zip(views, rvecs, tvecs):
        T_b_f = fk_flange(v["q"])                               # nominal kinematics + encoders
        R_g2b.append(T_b_f[:3, :3]); t_g2b.append(T_b_f[:3, 3].reshape(3, 1))
        R_t2c.append(cv2.Rodrigues(rvec)[0]); t_t2c.append(tvec.reshape(3, 1))
    R_c2f, t_c2f = cv2.calibrateHandEye(R_g2b, t_g2b, R_t2c, t_t2c, method=cv2.CALIB_HAND_EYE_PARK)
    T_f_c = np.eye(4)
    T_f_c[:3, :3] = R_c2f
    T_f_c[:3, 3] = t_c2f.ravel()
    init = dict(K=K, dist=dist, rms=float(rms), T_flange_cam=T_f_c, n_views=len(views))
    return refine_calibration(views, obj, init)


def _pose_from_vec(p):
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(np.asarray(p[:3]))[0]
    T[:3, 3] = p[3:6]
    return T


def refine_calibration(views, obj, init, fix_intrinsics=True):
    """Joint nonlinear refinement (bundle-adjustment style) of the camera mount, the board
    pose in the base frame and the pinhole intrinsics, minimising corner reprojection error."""
    from scipy.optimize import least_squares

    T_b_f = [fk_flange(v["q"]) for v in views]
    # initial board-in-base pose: average over views of T_b_f * T_f_c * T_c_board
    T_m0 = init["T_flange_cam"]
    T_board0 = []
    for v, Tbf in zip(views, T_b_f):
        ok, rvec, tvec = cv2.solvePnP(obj, v["corners"], init["K"], np.zeros(5), flags=cv2.SOLVEPNP_ITERATIVE)
        T_c_board = np.eye(4); T_c_board[:3, :3] = cv2.Rodrigues(rvec)[0]; T_c_board[:3, 3] = tvec.ravel()
        T_board0.append(Tbf @ T_m0 @ np.linalg.inv(np.eye(4)) @ np.linalg.inv(np.linalg.inv(T_c_board)) if False
                        else Tbf @ T_m0 @ T_c_board)
    Tb = np.mean([T for T in T_board0], axis=0)
    # re-orthonormalise the rotation part
    U, _, Vt = np.linalg.svd(Tb[:3, :3]); Tb[:3, :3] = U @ Vt
    K_init = init["K"]
    x0 = np.concatenate([
        cv2.Rodrigues(T_m0[:3, :3])[0].ravel(), T_m0[:3, 3],
        cv2.Rodrigues(Tb[:3, :3])[0].ravel(), Tb[:3, 3],
        [K_init[0, 0], K_init[1, 1], K_init[0, 2], K_init[1, 2]]])

    def residuals(x):
        T_m = _pose_from_vec(x[0:6]); T_B = _pose_from_vec(x[6:12])
        fx, fy, cx, cy = K_init[0, 0], K_init[1, 1], K_init[0, 2], K_init[1, 2] if fix_intrinsics else x[12:16]
        res = []
        for v, Tbf in zip(views, T_b_f):
            T_b_c = Tbf @ T_m
            Xb = (T_B[:3, :3] @ obj.T + T_B[:3, 3:4]).T                     # board pts in base
            Xc = (np.linalg.inv(T_b_c)[:3, :3] @ Xb.T + np.linalg.inv(T_b_c)[:3, 3:4]).T
            u = fx * Xc[:, 0] / Xc[:, 2] + cx
            w = fy * Xc[:, 1] / Xc[:, 2] + cy
            res.append(np.column_stack([u, w]) - v["corners"])
        return np.concatenate(res).ravel()

    sol = least_squares(residuals, x0, method="lm")
    x = sol.x
    rms = float(np.sqrt(np.mean(sol.fun ** 2)))
    if fix_intrinsics:
        K = K_init.copy()
    else:
        K = np.array([[x[12], 0, x[14]], [0, x[13], x[15]], [0, 0, 1.0]])
    return dict(K=K, dist=np.zeros(5), rms=rms, T_flange_cam=_pose_from_vec(x[0:6]),
                T_board_base=_pose_from_vec(x[6:12]), n_views=len(views), init_rms=init["rms"])


def mount_truth(sim):
    """EVALUATION ONLY: the true flange->camera transform in the OpenCV camera convention."""
    T_b_c_mj = sim.truth_cam_pose()
    flip = np.diag([1.0, -1.0, -1.0, 1.0])                    # MuJoCo cam (z back) -> OpenCV (z forward)
    T_b_c = T_b_c_mj @ flip
    T_b_f = np.eye(4)
    T_b_f[:3, :3] = sim.data.body("flange").xmat.reshape(3, 3)
    T_b_f[:3, 3] = sim.data.body("flange").xpos
    return np.linalg.inv(T_b_f) @ T_b_c


def run_calibration(seed=123, n_views=40, verbose=True, sim=None):
    sim = Sim(seed=seed) if sim is None else sim
    rng = np.random.default_rng(seed)
    views, obj = collect_views(sim, n_views, rng)
    est = calibrate(views, obj)
    T_true = mount_truth(sim)
    K_true = intrinsics_from_fovy()
    dt = np.linalg.norm(est["T_flange_cam"][:3, 3] - T_true[:3, 3]) * 1000
    dR = np.degrees(np.linalg.norm(cv2.Rodrigues(est["T_flange_cam"][:3, :3] @ T_true[:3, :3].T)[0]))
    report = dict(
        n_views=est["n_views"], reproj_rms_px=est["rms"],
        fx=float(est["K"][0, 0]), fy=float(est["K"][1, 1]), cx=float(est["K"][0, 2]), cy=float(est["K"][1, 2]),
        fx_true=float(K_true[0, 0]), cx_true=float(K_true[0, 2]),
        mount_trans_err_mm=float(dt), mount_rot_err_deg=float(dR),
    )
    if verbose:
        print("Calibration:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in report.items()})
    return est, report, sim


if __name__ == "__main__":
    import json, os
    est, report, _ = run_calibration()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(out, exist_ok=True)
    np.savez(os.path.join(out, "calibration.npz"), K=est["K"], dist=est["dist"], T_flange_cam=est["T_flange_cam"])
    json.dump(report, open(os.path.join(out, "calibration_report.json"), "w"), indent=2)
