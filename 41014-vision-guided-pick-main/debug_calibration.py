import sys
import cv2
import numpy as np

sys.path.insert(0, r".\41014-vision-guided-pick-main\track_b_vision_pick")

print("1. Importing calibration...", flush=True)

import calibration as C
from ur3_model import Sim

print("2. Creating simulation...", flush=True)

sim = Sim(seed=7, render=True)

print("3. Generating calibration pose...", flush=True)

q = C.calibration_poses(np.random.default_rng(123), 1)[0]

print("4. Resetting robot...", flush=True)

sim.reset(q)

print("5. Capturing camera image...", flush=True)

rgb = sim.render_rgb()

print("6. Saving image...", flush=True)

cv2.imwrite(
    "calibration_debug2.png",
    cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
)

print("7. Detecting checkerboard...", flush=True)

gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

found, corners = cv2.findChessboardCornersSB(
    gray,
    C.PATTERN,
    cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE
)

print("Checkerboard detected:", found, flush=True)
print("Image saved: calibration_debug2.png", flush=True)
