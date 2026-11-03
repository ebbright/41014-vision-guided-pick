"""
Calibration + randomised robustness trials for the vision-guided UR3 pick.

Each trial randomises the cube position (and yaw, mass and friction) and runs the full
pick with sensor noise. Ground truth is read ONLY after the trial, for scoring.
Usage:  python3 run_experiments.py [--trials N] [--quick]
"""
import os, json, time, argparse, zlib
import numpy as np

import calibration as C
from ur3_model import (Sim, CUBE_X_RANGE, CUBE_Y_RANGE, CUBE_MASS, CUBE_FRICTION, TABLE_Z, CUBE_HALF,
                       wrap_angle)
from pick_pipeline import PickRobot, Q_REST

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")

CONDITIONS = {
    "nominal": dict(n=50, mass=(0.03, 0.10), friction=(0.7, 1.2), cfg={}),
    "depth_noise_x3": dict(n=20, mass=(0.03, 0.10), friction=(0.7, 1.2), cfg=dict(depth_noise_gain=3.0)),
    "encoder_noise_x4": dict(n=20, mass=(0.03, 0.10), friction=(0.7, 1.2), cfg=dict(enc_sigma=0.008)),
    "heavy_slippery": dict(n=20, mass=(0.10, 0.20), friction=(0.35, 0.5), cfg={}),
    "intrinsics_3pct_error": dict(n=20, mass=(0.03, 0.10), friction=(0.7, 1.2), cfg=dict(fx_scale=1.03)),
}


def sample_scene(rng, mass_rng, fr_rng):
    return dict(x=float(rng.uniform(*CUBE_X_RANGE)), y=float(rng.uniform(*CUBE_Y_RANGE)),
                yaw=float(rng.uniform(-np.pi / 4, np.pi / 4)),
                mass=float(rng.uniform(*mass_rng)), friction=float(rng.uniform(*fr_rng)))


def run_trial(sim, calib, scene, seed, cfg, keep_log=False):
    sim.reset(Q_REST)
    sim.set_cube(scene["x"], scene["y"], scene["yaw"], mass=scene["mass"], friction=scene["friction"])
    z0 = sim.truth_cube_centre()[2]
    rng = np.random.default_rng(seed)
    robot = PickRobot(sim, calib, rng, dict(cfg, keep_log=keep_log))
    out = robot.run()
    # ---- scoring (ground truth, after the trial) ----
    res = dict(scene=scene, reach_ok=bool(out.get("success_reach", False)), failure=out.get("failure", ""))
    if out.get("success_reach"):
        truth_c = out["truth_at_plan_centre"]                 # snapshot taken when the grasp was planned
        truth_yaw = np.mod(out["truth_at_plan_yaw"], np.pi / 2)
        est_c = out["estimate_centre"]
        est_yaw = out["estimate_yaw"]
        yaw_err = np.degrees(abs(wrap_angle(4 * (est_yaw - truth_yaw)) / 4))
        lift = float(sim.truth_cube_centre()[2] - z0)        # cube height after the lift (truth, scoring only)
        res.update(
            est_err_xy_mm=float(1000 * np.linalg.norm(est_c[:2] - truth_c[:2])),
            est_err_z_mm=float(1000 * (est_c[2] - truth_c[2])),
            est_err_vec_mm=(1000 * (est_c - truth_c)).tolist(),
            yaw_err_deg=float(yaw_err),
            est_std_mm=float(1000 * np.linalg.norm(out["estimate_std"][:2])),
            n_kf_updates=int(out["n_updates"]),
            holding=bool(out["holding"]),
            lift_m=float(lift),
            success=bool(out["holding"] and lift > 0.05),
            finger_closure=float(out["finger_closure"]),
        )
        if len(robot.log["t"]):
            L = {k: np.array(v) for k, v in robot.log.items() if k in ("q_true", "q_ref", "qd_true", "qd_hat", "q_hat", "t")}
            moving = np.array([p.startswith(("to_", "pbvs_correct", "descend", "lift")) for p in robot.log["phase"]])
            if moving.any():
                res["track_rms_rad"] = float(np.sqrt(np.mean((L["q_true"][moving] - L["q_ref"][moving]) ** 2)))
            res["vel_est_rms_radps"] = float(np.sqrt(np.mean((L["qd_hat"] - L["qd_true"]) ** 2)))
    else:
        res.update(success=False)
    return res, (robot.log if keep_log else None)


def run_condition(name, spec, sim, calib, n_override=None, seed0=1000, keep_first=False):
    n = spec["n"] if n_override is None else n_override
    mass_rng, fr_rng = spec["mass"], spec["friction"]
    rng = np.random.default_rng(seed0 + zlib.crc32(name.encode()) % 100000)   # reproducible across processes
    results, logs = [], []
    t0 = time.time()
    for i in range(n):
        scene = sample_scene(rng, mass_rng, fr_rng)
        # every trial logs its trajectory (tracking and observer metrics); only the first is kept for plots
        res, log = run_trial(sim, calib, scene, seed=seed0 + i, cfg=spec["cfg"], keep_log=True)
        results.append(res)
        if keep_first and i == 0:
            logs.append(log)
    succ = [r["success"] for r in results]
    summary = dict(condition=name, trials=n, successes=int(sum(succ)), success_rate=float(np.mean(succ)),
                   reach_failures=int(sum(not r["reach_ok"] for r in results)),
                   wall_s=round(time.time() - t0, 1))
    for key in ["est_err_xy_mm", "yaw_err_deg", "lift_m", "track_rms_rad", "vel_est_rms_radps", "est_err_z_mm"]:
        vals = [r[key] for r in results if key in r]
        if vals:
            summary[key + "_median"] = float(np.median(vals))
            summary[key + "_p95"] = float(np.percentile(np.abs(vals), 95))
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in summary.items()}))
    return summary, results, (logs[0] if logs else None)


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(c - h), float(c + h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="few trials per condition (smoke test)")
    args = ap.parse_args()
    os.makedirs(RES, exist_ok=True)
    sim = Sim(seed=7)
    est, rep, _ = C.run_calibration(seed=123, n_views=40, verbose=True, sim=sim)
    calib = dict(K=est["K"], T_flange_cam=est["T_flange_cam"])
    np.savez(os.path.join(RES, "calibration.npz"), K=est["K"], dist=est["dist"], T_flange_cam=est["T_flange_cam"])
    json.dump(rep, open(os.path.join(RES, "calibration_report.json"), "w"), indent=2)

    summaries, all_results, showcase = [], {}, None
    for name, spec in CONDITIONS.items():
        n_over = 3 if args.quick else None
        s, rs, log = run_condition(name, spec, sim, calib, n_override=n_over, keep_first=(name == "nominal"))
        lo, hi = wilson(s["successes"], s["trials"])
        s["success_ci95"] = [round(lo, 3), round(hi, 3)]
        summaries.append(s); all_results[name] = rs
        if log is not None and showcase is None:
            showcase = log
    json.dump(dict(calibration=rep, conditions=summaries, trials=all_results),
              open(os.path.join(RES, "experiment_results.json"), "w"), indent=1, default=float)
    if showcase is not None:
        np.savez(os.path.join(RES, "showcase_log.npz"), **{k: np.array(v) for k, v in showcase.items()})
    print("saved", RES)


if __name__ == "__main__":
    main()
