"""
Vision: colour segmentation of the target, depth-based 3D localisation in the robot
base frame, and cube yaw from the top-face outline. Inputs are sensor data only:
RGB, depth, calibrated intrinsics K, and the calibrated camera pose (from encoders
+ hand-eye). No ground truth is used here.
"""
import numpy as np
import cv2

CUBE_HALF_EST = 0.02        # known object size (a model parameter, not measured)
TOP_BAND = 0.005            # points within 5 mm of the highest surface are the top face


def add_depth_noise(depth, rng, gain=1.0):
    """Simulated time-of-flight/stereo noise, growing with range (applied to sim depth)."""
    sig = gain * (0.001 + 0.003 * depth ** 2)
    out = depth + rng.normal(0, 1, depth.shape) * sig
    return np.where(depth > 0, out, depth).astype(np.float32)


def cube_mask(rgb):
    """Largest red blob in the image (HSV threshold). Returns boolean mask or None."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    red = (((h <= 8) | (h >= 170)) & (s >= 110) & (v >= 50)).astype(np.uint8)
    red = cv2.morphologyEx(red, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(red)
    if n <= 1:
        return None
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[k, cv2.CC_STAT_AREA] < 40:
        return None
    return lab == k


def backproject(depth, mask, K):
    """Pixels in mask -> 3D points in the OpenCV camera frame (x right, y down, z forward)."""
    v, u = np.nonzero(mask & (depth > 0.02))
    Z = depth[v, u].astype(np.float64)
    X = (u - K[0, 2]) / K[0, 0] * Z
    Y = (v - K[1, 2]) / K[1, 1] * Z
    return np.column_stack([X, Y, Z]), (u, v)


def localise_cube(rgb, depth, K, T_base_cam):
    """Cube centre (3,), yaw mod 90 deg (rad), and diagnostics, or None if not found."""
    mask = cube_mask(rgb)
    if mask is None:
        return None
    Pc, (u, v) = backproject(depth, mask, K)
    if len(Pc) < 30:
        return None
    Pb = (T_base_cam[:3, :3] @ Pc.T + T_base_cam[:3, 3:4]).T          # base frame
    z_max = np.percentile(Pb[:, 2], 98)
    top = Pb[Pb[:, 2] > z_max - TOP_BAND]
    if len(top) < 15:
        return None
    xy = np.median(top[:, :2], axis=0)
    z_top = float(np.median(top[:, 2]))
    # yaw of the top face: minimum-area rectangle of the top-face outline (mod 90 deg)
    pts = (top[:, :2] * 1000.0).astype(np.float32)
    (_, _), (_, _), ang = cv2.minAreaRect(pts)
    yaw = np.deg2rad(ang) % (np.pi / 2)
    # approximate depth of the centroid, used to size the measurement noise
    Zc = float(np.median(Pc[:, 2]))
    return dict(centre=np.array([xy[0], xy[1], z_top - CUBE_HALF_EST]), yaw=float(yaw),
                z_top=z_top, n_top=int(len(top)), depth=Zc, uv=(float(u.mean()), float(v.mean())),
                n_pix=int(mask.sum()))


def measurement_cov(depth_m):
    """Covariance of a single cube-centre measurement, growing with range (2.5 mm lateral floor)."""
    sz = 0.001 + 0.003 * depth_m ** 2
    return np.diag([0.0025 ** 2, 0.0025 ** 2, sz ** 2])


def overlay(rgb, det):
    """Debug image: mask outline and estimated centroid."""
    img = rgb.copy()
    if det is not None:
        u, v = det["uv"]
        cv2.circle(img, (int(u), int(v)), 6, (0, 255, 0), 2)
    return img
