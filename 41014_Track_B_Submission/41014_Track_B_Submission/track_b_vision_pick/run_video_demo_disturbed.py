"""Run one live depth-noise-disturbed MuJoCo UR3 pick for video recording.

Place this file beside run_experiments.py, pick_pipeline.py and ur3_model.py.
It reuses results/calibration.npz and reproduces the first depth_noise_x3 trial
from the experiment configuration. It does NOT rerun the 130-trial experiment.
"""
import os
import time
import zlib
import numpy as np
import mujoco.viewer

from ur3_model import Sim, CUBE_X_RANGE, CUBE_Y_RANGE
from pick_pipeline import PickRobot, Q_REST

HERE = os.path.dirname(os.path.abspath(__file__))
CALIBRATION_FILE = os.path.join(HERE, "results", "calibration.npz")


def main():
    if not os.path.isfile(CALIBRATION_FILE):
        raise FileNotFoundError(
            "Could not find results/calibration.npz. Restore the supplied results folder "
            "or run the calibration workflow first."
        )

    saved = np.load(CALIBRATION_FILE)
    calib = {"K": saved["K"], "T_flange_cam": saved["T_flange_cam"]}

    # Match the first depth_noise_x3 trial in run_experiments.py.
    condition = "depth_noise_x3"
    seed0 = 1000
    scene_rng = np.random.default_rng(seed0 + zlib.crc32(condition.encode()) % 100000)
    scene = {
        "x": float(scene_rng.uniform(*CUBE_X_RANGE)),
        "y": float(scene_rng.uniform(*CUBE_Y_RANGE)),
        "yaw": float(scene_rng.uniform(-np.pi / 4, np.pi / 4)),
        "mass": float(scene_rng.uniform(0.03, 0.10)),
        "friction": float(scene_rng.uniform(0.7, 1.2)),
    }

    sim = Sim(seed=7, render=True)
    sim.reset(Q_REST)
    sim.set_cube(
        scene["x"], scene["y"], scene["yaw"],
        mass=scene["mass"], friction=scene["friction"]
    )

    print("Opening live MuJoCo viewer...")
    print("Disturbance: depth sensor noise increased to 3x nominal.")
    print("Scene:", scene)
    print("Keep the viewer open while the robot performs the pick.")

    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        original_step_control = sim.step_control

        def visible_step_control(tau, grip=None, n=None):
            original_step_control(tau, grip=grip, n=n)
            if viewer.is_running():
                viewer.sync()
                time.sleep(0.012)

        sim.step_control = visible_step_control
        robot = PickRobot(
            sim,
            calib,
            np.random.default_rng(seed0),
            {"keep_log": True, "depth_noise_gain": 3.0},
        )
        result = robot.run()

        print("\\nDISTURBED DEMO RESULT (depth_noise_x3)")
        print("Reach/planning succeeded:", result.get("success_reach", False))
        print("Gripper holding cube:", result.get("holding", False))
        if "failure" in result:
            print("Failure:", result["failure"])
        if "estimate_centre" in result:
            print("Estimated cube centre (m):", np.round(result["estimate_centre"], 4))
            print("Estimated position std (m):", np.round(result["estimate_std"], 5))
        print("\\nLeave the viewer open while recording. Press Enter here to close it.")
        input()


if __name__ == "__main__":
    main()
