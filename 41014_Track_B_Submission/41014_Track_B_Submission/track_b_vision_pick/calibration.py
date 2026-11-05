
"""
Camera calibration for the wrist RGB-D camera.

1. Intrinsics: a checkerboard is imaged from many arm poses; cv2.calibrateCamera
   estimates fx, fy, cx, cy (and reports the reprojection RMS).
2. Hand-eye: for each pose, the flange pose comes from encoder readings + nominal DH
   kinematics and the board pose from PnP. cv2.calibrateHandEye estimates the
   fixed transform T_flange_cam.

Only images and joint encoders are used. The true mount / intrinsics are used
solely to report calibration accuracy.
"""

import numpy as np
import cv2

from ur3_model import (
    Sim,
    BOARD_SQUARES,
    BOARD_SQUARE_M,
    fk_flange,
    flange_from_tcp,
    tool_down_rotation,
    solve_ik,
    rot_x,
    rot_y,
    SCAN_TCP,
    BOARD_CENTRE,
    intrinsics_from_fovy,
    IMG_H,
    IMG_W,
    wrap_angle,
    CAM_MOUNT_POS,
    TCP_OFFSET,
)


PATTERN = (BOARD_SQUARES[0] - 1, BOARD_SQUARES[1] - 1)


def board_object_points():
    nx, ny = PATTERN

    pts = np.zeros((nx * ny, 3), np.float32)
    grid = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    pts[:, :2] = grid * BOARD_SQUARE_M

    return pts


def _rot_about(R, rx, ry):
    return R @ rot_x(rx) @ rot_y(ry)


def calibration_poses(rng, n_views=40):
    """
    Generate calibration poses with the wrist camera directed towards
    the checkerboard.

    Both the camera mounting offset and TCP offset are accounted for.
    """

    poses = []

    seed_q = solve_ik(
        flange_from_tcp(
            tool_down_rotation(0.0),
            SCAN_TCP
        ),
        np.array([0, -1.57, 1.57, -1.57, -1.57, 0.0])
    )

    if seed_q is None:
        raise RuntimeError("Could not generate initial calibration IK pose.")

    target_xy = BOARD_CENTRE[:2]

    attempts = 0

    while len(poses) < n_views and attempts < 6000:

        attempts += 1

        yaw = rng.uniform(-0.3, 0.3)

        R = _rot_about(
            tool_down_rotation(yaw),
            rng.uniform(-0.10, 0.10),
            rng.uniform(-0.10, 0.10)
        )

        # Desired camera position above the checkerboard.
        camera_target = np.array([
            target_xy[0] + rng.uniform(-0.015, 0.015),
            target_xy[1] + rng.uniform(-0.015, 0.015),
            BOARD_CENTRE[2] + rng.uniform(0.28, 0.38)
        ])

        # Convert desired camera position into flange position.
        flange_position = camera_target - R @ CAM_MOUNT_POS

        # Convert flange position into the TCP position expected by
        # flange_from_tcp().
        tcp_position = flange_position + R[:, 2] * TCP_OFFSET

        q = solve_ik(
            flange_from_tcp(R, tcp_position),
            seed_q,
            rng=rng
        )

        if q is None:
            continue

        poses.append(q)

    if len(poses) < n_views:
        raise RuntimeError(
            f"Could only generate {len(poses)} of {n_views} "
            "calibration poses. Check the robot workspace."
        )

    return poses


def collect_views(
    sim,
    n_views=40,
    rng=None,
    enc_sigma=0.002,
    n_avg=50
):
    """
    Move the arm to calibration poses and detect the checkerboard.
    """

    rng = np.random.default_rng(0) if rng is None else rng

    obj = board_object_points()

    views = []

    candidate_poses = calibration_poses(rng, n_views * 3)

    for index, q_true in enumerate(candidate_poses):

        if len(views) >= n_views:
            break

        sim.reset(q_true)

        rgb = sim.render_rgb()

        gray = cv2.cvtColor(
            rgb,
            cv2.COLOR_RGB2GRAY
        )

        found, corners = cv2.findChessboardCorners(
                gray,
                PATTERN,
                cv2.CALIB_CB_ADAPTIVE_THRESH |
                cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        if not found:
            continue

        corners = corners.reshape(-1, 2).astype(np.float32)

        # Average encoder readings while the arm is held still.
        q_meas = (
            q_true +
            rng.normal(0, enc_sigma / np.sqrt(n_avg), 6)
        )

        views.append({
            "q": q_meas,
            "corners": corners
        })

        if len(views) % 5 == 0:
            print(
                f"Collected {len(views)}/{n_views} calibration views",
                flush=True
            )

    if len(views) < n_views:
        raise RuntimeError(
            f"Only collected {len(views)} valid checkerboard views "
            f"out of {n_views} required."
        )

    return views, obj


def calibrate(views, obj, img_size=(IMG_W, IMG_H)):
    """
    Estimate camera intrinsics and hand-eye transformation.
    """

    K0 = intrinsics_from_fovy().astype(np.float64)

    obj_pts = [obj for _ in views]

    img_pts = [
        v["corners"].reshape(-1, 1, 2)
        for v in views
    ]

    flags = (
        cv2.CALIB_USE_INTRINSIC_GUESS
        | cv2.CALIB_FIX_K1
        | cv2.CALIB_FIX_K2
        | cv2.CALIB_FIX_K3
        | cv2.CALIB_ZERO_TANGENT_DIST
    )

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_pts,
        img_pts,
        img_size,
        K0.copy(),
        np.zeros(5),
        flags=flags
    )

    # Reprojection error for each view.
    errs = []

    for o, i, r, t in zip(
        obj_pts,
        img_pts,
        rvecs,
        tvecs
    ):

        projected, _ = cv2.projectPoints(
            o,
            r,
            t,
            K,
            np.zeros(5)
        )

        error = np.sqrt(
            (
                (
                    projected.reshape(-1, 2)
                    - i.reshape(-1, 2)
                ) ** 2
            ).sum(1)
        ).mean()

        errs.append(float(error))

    # Reject unusually poor views.
    threshold = max(3 * np.median(errs), 0.5)

    keep = [
        k for k, error in enumerate(errs)
        if error < threshold
    ]

    if len(keep) < len(views):

        views = [views[k] for k in keep]

        obj_pts = [obj for _ in views]

        img_pts = [
            v["corners"].reshape(-1, 1, 2)
            for v in views
        ]

        if len(views) < 3:
            raise RuntimeError(
                "Too few calibration views remain after outlier rejection."
            )

        rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            obj_pts,
            img_pts,
            img_size,
            K.copy(),
            np.zeros(5),
            flags=flags
        )

    # Hand-eye calibration.
    R_g2b, t_g2b = [], []
    R_t2c, t_t2c = [], []

    for v, rvec, tvec in zip(
        views,
        rvecs,
        tvecs
    ):

        T_b_f = fk_flange(v["q"])

        R_g2b.append(T_b_f[:3, :3])

        t_g2b.append(
            T_b_f[:3, 3].reshape(3, 1)
        )

        R_t2c.append(
            cv2.Rodrigues(rvec)[0]
        )

        t_t2c.append(
            tvec.reshape(3, 1)
        )

    R_c2f, t_c2f = cv2.calibrateHandEye(
        R_g2b,
        t_g2b,
        R_t2c,
        t_t2c,
        method=cv2.CALIB_HAND_EYE_PARK
    )

    T_f_c = np.eye(4)

    T_f_c[:3, :3] = R_c2f
    T_f_c[:3, 3] = t_c2f.ravel()

    init = {
        "K": K,
        "dist": dist,
        "rms": float(rms),
        "T_flange_cam": T_f_c,
        "n_views": len(views)
    }

    return refine_calibration(
        views,
        obj,
        init
    )


def _pose_from_vec(p):

    T = np.eye(4)

    T[:3, :3] = cv2.Rodrigues(
        np.asarray(p[:3])
    )[0]

    T[:3, 3] = p[3:6]

    return T


def refine_calibration(
    views,
    obj,
    init,
    fix_intrinsics=True
):
    """
    Joint nonlinear refinement of camera mount, board pose and
    optionally camera intrinsics.
    """

    from scipy.optimize import least_squares

    T_b_f = [
        fk_flange(v["q"])
        for v in views
    ]

    T_m0 = init["T_flange_cam"]

    # Initial estimate of board pose in the base frame.
    T_board0 = []

    for v, Tbf in zip(views, T_b_f):

        ok, rvec, tvec = cv2.solvePnP(
            obj,
            v["corners"],
            init["K"],
            np.zeros(5),
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not ok:
            continue

        T_c_board = np.eye(4)

        T_c_board[:3, :3] = cv2.Rodrigues(rvec)[0]

        T_c_board[:3, 3] = tvec.ravel()

        T_board0.append(
            Tbf @ T_m0 @ T_c_board
        )

    if not T_board0:
        raise RuntimeError(
            "Could not initialise checkerboard pose refinement."
        )

    Tb = np.mean(T_board0, axis=0)

    # Re-orthonormalise rotation.
    U, _, Vt = np.linalg.svd(Tb[:3, :3])

    Tb[:3, :3] = U @ Vt

    K_init = init["K"]

    x0 = np.concatenate([
        cv2.Rodrigues(T_m0[:3, :3])[0].ravel(),
        T_m0[:3, 3],
        cv2.Rodrigues(Tb[:3, :3])[0].ravel(),
        Tb[:3, 3]
    ])

    if not fix_intrinsics:
        x0 = np.concatenate([
            x0,
            [
                K_init[0, 0],
                K_init[1, 1],
                K_init[0, 2],
                K_init[1, 2]
            ]
        ])

    def residuals(x):

        T_m = _pose_from_vec(x[0:6])

        T_B = _pose_from_vec(x[6:12])

        if fix_intrinsics:

            fx = K_init[0, 0]
            fy = K_init[1, 1]
            cx = K_init[0, 2]
            cy = K_init[1, 2]

        else:

            fx, fy, cx, cy = x[12:16]

        res = []

        Xb = (
            T_B[:3, :3] @ obj.T
            + T_B[:3, 3:4]
        ).T

        for v, Tbf in zip(views, T_b_f):

            T_b_c = Tbf @ T_m

            T_c_b = np.linalg.inv(T_b_c)

            Xc = (
                T_c_b[:3, :3] @ Xb.T
                + T_c_b[:3, 3:4]
            ).T

            z = np.maximum(Xc[:, 2], 1e-8)

            u = fx * Xc[:, 0] / z + cx

            w = fy * Xc[:, 1] / z + cy

            res.append(
                np.column_stack([u, w]) - v["corners"]
            )

        return np.concatenate(res).ravel()

    sol = least_squares(
        residuals,
        x0,
        method="lm"
    )

    x = sol.x

    rms = float(
        np.sqrt(np.mean(sol.fun ** 2))
    )

    if fix_intrinsics:

        K = K_init.copy()

    else:

        K = np.array([
            [x[12], 0, x[14]],
            [0, x[13], x[15]],
            [0, 0, 1.0]
        ])

    return {
        "K": K,
        "dist": np.zeros(5),
        "rms": rms,
        "T_flange_cam": _pose_from_vec(x[0:6]),
        "T_board_base": _pose_from_vec(x[6:12]),
        "n_views": len(views),
        "init_rms": init["rms"]
    }


def mount_truth(sim):
    """
    Evaluation only: true flange-to-camera transform.
    """

    T_b_c_mj = sim.truth_cam_pose()

    flip = np.diag([
        1.0, -1.0, -1.0, 1.0
    ])

    T_b_c = T_b_c_mj @ flip

    T_b_f = np.eye(4)

    T_b_f[:3, :3] = (
        sim.data.body("flange").xmat.reshape(3, 3)
    )

    T_b_f[:3, 3] = (
        sim.data.body("flange").xpos
    )

    return np.linalg.inv(T_b_f) @ T_b_c


def run_calibration(
    seed=123,
    n_views=40,
    verbose=True,
    sim=None
):

    sim = Sim(seed=seed) if sim is None else sim

    rng = np.random.default_rng(seed)

    print("Generating calibration views...", flush=True)

    views, obj = collect_views(
        sim,
        n_views,
        rng
    )

    print(
        f"Successfully collected {len(views)} views.",
        flush=True
    )

    est = calibrate(
        views,
        obj
    )

    T_true = mount_truth(sim)

    K_true = intrinsics_from_fovy()

    dt = (
        np.linalg.norm(
            est["T_flange_cam"][:3, 3]
            - T_true[:3, 3]
        ) * 1000
    )

    dR = np.degrees(
        np.linalg.norm(
            cv2.Rodrigues(
                est["T_flange_cam"][:3, :3]
                @ T_true[:3, :3].T
            )[0]
        )
    )

    report = {
        "n_views": est["n_views"],
        "reproj_rms_px": est["rms"],
        "fx": float(est["K"][0, 0]),
        "fy": float(est["K"][1, 1]),
        "cx": float(est["K"][0, 2]),
        "cy": float(est["K"][1, 2]),
        "fx_true": float(K_true[0, 0]),
        "cx_true": float(K_true[0, 2]),
        "mount_trans_err_mm": float(dt),
        "mount_rot_err_deg": float(dR)
    }

    if verbose:

        print(
            "Calibration:",
            {
                k: round(v, 3) if isinstance(v, float) else v
                for k, v in report.items()
            }
        )

    return est, report, sim


if __name__ == "__main__":

    import json
    import os

    est, report, _ = run_calibration()

    out = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "results"
    )

    os.makedirs(out, exist_ok=True)

    np.savez(
        os.path.join(out, "calibration.npz"),
        K=est["K"],
        dist=est["dist"],
        T_flange_cam=est["T_flange_cam"]
    )

    with open(
        os.path.join(out, "calibration_report.json"),
        "w"
    ) as f:

        json.dump(
            report,
            f,
            indent=2
        )
