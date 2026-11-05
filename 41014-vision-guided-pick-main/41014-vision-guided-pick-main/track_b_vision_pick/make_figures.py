"""Figures and tables for the report, from results/ (run run_experiments.py first)."""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIG = os.path.join(RES, "figures")


def fig_showcase():
    L = np.load(os.path.join(RES, "showcase_log.npz"))
    t = L["t"]
    fig, ax = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    for j in range(6):
        ax[0].plot(t, L["q_ref"][:, j], "--", lw=1, color=f"C{j}")
        ax[0].plot(t, L["q_true"][:, j], "-", lw=1.2, color=f"C{j}", label=f"joint {j+1}")
    ax[0].set_ylabel("joint angle [rad]")
    ax[0].set_title("Showcase trial: reference (dashed) vs true joint angle (solid)")
    ax[0].legend(ncol=6, fontsize=7, loc="lower right")
    for j in range(6):
        ax[1].plot(t, L["qd_true"][:, j], "-", lw=1, color=f"C{j}")
        ax[1].plot(t, L["qd_hat"][:, j], ":", lw=1.4, color=f"C{j}")
    ax[1].set_ylabel("joint velocity [rad/s]")
    ax[1].set_title("Unmeasured joint velocities: truth (solid) vs Kalman observer (dotted)")
    err = L["q_hat"] - L["q_true"]
    ax[2].plot(t, np.rad2deg(err), lw=0.8)
    ax[2].set_ylabel("observer position error [deg]")
    ax[2].set_xlabel("time [s]")
    ax[2].set_title("Estimate error of encoder-fused joint state")
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_showcase_tracking.png"), dpi=130)
    plt.close(fig)


def fig_success(summaries):
    names = [s["condition"] for s in summaries]
    rates = np.array([s["success_rate"] for s in summaries]) * 100
    lo = np.array([s["success_ci95"][0] for s in summaries]) * 100
    hi = np.array([s["success_ci95"][1] for s in summaries]) * 100
    fig, ax = plt.subplots(figsize=(9, 4.2))
    y = np.arange(len(names))
    ax.barh(y, rates, xerr=[rates - lo, hi - rates], color="C2", alpha=0.8, capsize=4)
    for i, s in enumerate(summaries):
        ax.text(max(hi[i], rates[i]) + 1.5, i, f"{s['successes']}/{s['trials']}", va="center", fontsize=9)
    ax.set_yticks(y); ax.set_yticklabels(names)
    ax.set_xlim(0, 110); ax.invert_yaxis()
    ax.set_xlabel("grasp-and-lift success rate [%] (95% Wilson interval)")
    ax.set_title("Robustness over randomised cube pose, mass and friction")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_success_rates.png"), dpi=130); plt.close(fig)


def fig_errors(results):
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.8))
    for name, rs in results.items():
        xy = [r["est_err_xy_mm"] for r in rs if "est_err_xy_mm" in r]
        yaw = [r["yaw_err_deg"] for r in rs if "yaw_err_deg" in r]
        tr = [r["track_rms_rad"] for r in rs if "track_rms_rad" in r]
        ax[0].hist(xy, bins=15, alpha=0.5, label=name)
        ax[1].hist(yaw, bins=15, alpha=0.5, label=name)
        ax[2].hist(np.rad2deg(tr), bins=15, alpha=0.5, label=name)
    ax[0].set_xlabel("final cube-centre error (xy) [mm]")
    ax[1].set_xlabel("cube yaw error [deg]")
    ax[2].set_xlabel("joint tracking RMS [deg]")
    for a in ax:
        a.grid(alpha=0.3); a.legend(fontsize=7)
    fig.suptitle("Localisation and tracking errors across trials (truth used for scoring only)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_error_histograms.png"), dpi=130); plt.close(fig)


def fig_poles():
    import sys
    sys.path.insert(0, HERE)
    from control import ArmModel, StateSpaceJointController
    arm = ArmModel()
    q = np.load(os.path.join(RES, "showcase_log.npz"))["q_true"][0]
    ctl = StateSpaceJointController(arm, 0.01).set_operating_point(q)
    a = ctl.analysis()
    fig, ax = plt.subplots(figsize=(5.5, 5))
    th = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(th), np.sin(th), "k:", lw=1)
    ax.scatter(a["eig_open_loop_design"].real, a["eig_open_loop_design"].imag, marker="x", c="gray", label="open loop A_d")
    ax.scatter(a["eig_closed_loop_design"].real, a["eig_closed_loop_design"].imag, marker="o", facecolors="none", edgecolors="C0", label="state feedback A_d - B_d K")
    ax.scatter(a["eig_observer_design"].real, a["eig_observer_design"].imag, marker="s", facecolors="none", edgecolors="C3", label="observer A_d - L C")
    ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.legend(fontsize=7, loc="lower left")
    ax.set_xlabel("Re z"); ax.set_ylabel("Im z"); ax.set_title("Discrete poles (Ts = 10 ms)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_poles.png"), dpi=130); plt.close(fig)
    return a


def main():
    os.makedirs(FIG, exist_ok=True)
    data = json.load(open(os.path.join(RES, "experiment_results.json")))
    fig_showcase()
    fig_success(data["conditions"])
    fig_errors(data["trials"])
    a = fig_poles()
    summary = dict(
        ctrb_rank_design=a["ctrb_rank_design"], obsv_rank_design=a["obsv_rank_design"],
        ctrb_rank_full=a["ctrb_rank_full"], obsv_rank_full=a["obsv_rank_full"], n_states=a["n_states"],
        ctrb_cond=a["ctrb_cond"], obsv_cond=a["obsv_cond"],
        max_abs_closed_loop_eig=float(np.max(np.abs(a["eig_closed_loop_design"]))),
        max_abs_observer_eig=float(np.max(np.abs(a["eig_observer_design"]))),
        max_abs_open_loop_full_eig=a["eig_open_loop_full_max_abs"])
    json.dump(summary, open(os.path.join(RES, "state_space_analysis.json"), "w"), indent=2)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
