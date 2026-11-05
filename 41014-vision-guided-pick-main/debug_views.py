import sys
import time
import numpy as np
import cv2

sys.path.insert(0, r".\41014-vision-guided-pick-main\track_b_vision_pick")

import calibration as C
from ur3_model import Sim

sim = Sim(seed=7, render=True)
rng = np.random.default_rng(123)

print("Generating poses...", flush=True)
poses = C.calibration_poses(rng, 15)

for i, q in enumerate(poses):
    print(f"\nVIEW {i+1}/15", flush=True)

    t = time.time()
    sim.reset(q)
    print("Reset:", round(time.time()-t, 2), "s", flush=True)

    t = time.time()
    rgb = sim.render_rgb()
    print("Render:", round(time.time()-t, 2), "s", flush=True)

    t = time.time()
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    found, corners = cv2.findChessboardCornersSB(
        gray,
        C.PATTERN,
        cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    print("Detection:", found, round(time.time()-t, 2), "s", flush=True)

    if found:
        print("Checkerboard successfully detected.", flush=True)

print("\nDIAGNOSTIC COMPLETE", flush=True)
