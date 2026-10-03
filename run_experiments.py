"""Run all experiments and write results/ (CSV, tables, figures).

Usage:  python run_experiments.py            # full run (~2-3 minutes)
        python run_experiments.py --seeds 3  # quick run
"""
import argparse
import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.tree import export_text  # noqa: E402

from osml.policies import FEATURES, FIFO, LRU, OPT, Learned  # noqa: E402
from osml.sim import phase_metrics, simulate, train_static_model  # noqa: E402
from osml.traces import gen_trace  # noqa: E402

N_PHASE, V, H, W = 10_000, 256, 64, 256
FRAMES_MAIN = 32
FRAMES_SWEEP = [16, 32, 64]
ORDER = ["FIFO", "LRU", "OPT", "Learned-static", "Learned-adaptive"]
# Okabe-Ito colour-blind-safe palette
COLORS = {"FIFO": "#E69F00", "LRU": "#0072B2", "OPT": "#000000",
          "Learned-static": "#D55E00", "Learned-adaptive": "#009E73"}
OUT = "results"


def make_policies(model):
    return [FIFO(), LRU(), OPT(),
            Learned(model, adaptive=False, H=H, W=W),
            Learned(model, adaptive=True, H=H, W=W)]


def rolling(hits, win=500):
    k = np.ones(win) / win
    return np.convolve(hits.astype(float), k, mode="valid")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    seeds = list(range(args.seeds))

    rows = []                      # raw per-run rows
    roll = {p: [] for p in ORDER}  # rolling hit ratio at F=FRAMES_MAIN
    model_info = {}
    retrains = []

    for F in FRAMES_SWEEP:
        model, info = train_static_model(F, H=H, W=W, n_phase=N_PHASE, V=V)
        model_info[F] = dict(train_rows=info["rows"], positive_rate=info["positive_rate"],
                             importances=dict(zip(FEATURES, map(float, model.feature_importances_))))
        if F == FRAMES_MAIN:
            with open(f"{OUT}/static_tree.txt", "w") as fh:
                fh.write(export_text(model, feature_names=FEATURES))
        for s in seeds:
            trace, b = gen_trace(s, n_phase=N_PHASE, V=V)
            for pol in make_policies(model):
                hits = simulate(trace, F, pol)
                m = phase_metrics(hits, b)
                for ph in ("before", "after"):
                    rows.append(dict(seed=s, frames=F, policy=pol.name, phase=ph, **m[ph]))
                if F == FRAMES_MAIN:
                    roll[pol.name].append(rolling(hits))
                    if pol.name == "Learned-adaptive":
                        retrains.append(pol.retrains)
        print(f"done F={F}")

    # ---- raw CSV
    with open(f"{OUT}/raw_runs.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["seed", "frames", "policy", "phase", "hits", "faults", "hit_ratio"])
        w.writeheader()
        w.writerows(rows)

    def agg(F, pol, ph, key):
        v = np.array([r[key] for r in rows if r["frames"] == F and r["policy"] == pol and r["phase"] == ph], float)
        return v.mean(), v.std(ddof=1) if len(v) > 1 else 0.0

    # ---- summary CSV
    with open(f"{OUT}/summary.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frames", "policy", "phase", "hit_ratio_mean", "hit_ratio_std",
                    "faults_mean", "faults_std"])
        for F in FRAMES_SWEEP:
            for p in ORDER:
                for ph in ("before", "after"):
                    hm, hs = agg(F, p, ph, "hit_ratio")
                    fm, fs = agg(F, p, ph, "faults")
                    w.writerow([F, p, ph, f"{hm:.4f}", f"{hs:.4f}", f"{fm:.1f}", f"{fs:.1f}"])

    # ---- main table (F = FRAMES_MAIN)
    F = FRAMES_MAIN
    lines_md = ["| Policy | Hit% before | Hit% after | Drop (pp) | Faults before | Faults after | Gap to OPT after (pp) |",
                "|---|---|---|---|---|---|---|"]
    lines_tex = []
    opt_after = agg(F, "OPT", "after", "hit_ratio")[0]
    summary_main = {}
    for p in ORDER:
        hb, hbs = agg(F, p, "before", "hit_ratio")
        ha, has = agg(F, p, "after", "hit_ratio")
        fb, fbs = agg(F, p, "before", "faults")
        fa, fas = agg(F, p, "after", "faults")
        drop, gap = (hb - ha) * 100, (opt_after - ha) * 100
        summary_main[p] = dict(hit_before=hb, hit_after=ha, drop_pp=drop, faults_before=fb,
                               faults_after=fa, gap_to_opt_after_pp=gap)
        lines_md.append(f"| {p} | {hb*100:.1f} ± {hbs*100:.1f} | {ha*100:.1f} ± {has*100:.1f} | {drop:.1f} | "
                        f"{fb:.0f} ± {fbs:.0f} | {fa:.0f} ± {fas:.0f} | {gap:.1f} |")
        lines_tex.append(f"{p} & {hb*100:.1f} $\\pm$ {hbs*100:.1f} & {ha*100:.1f} $\\pm$ {has*100:.1f} & "
                         f"{drop:.1f} & {fb:.0f} & {fa:.0f} & {gap:.1f} \\\\")
    open(f"{OUT}/table_main.md", "w").write("\n".join(lines_md) + "\n")
    header = ("\\begin{tabular}{lcccccc}\n\\toprule\n"
              "Policy & Hit\\% before & Hit\\% after & Drop (pp) & Faults before & Faults after & Gap (pp) \\\\\n"
              "\\midrule\n")
    open(f"{OUT}/table_main.tex", "w").write(header + "\n".join(lines_tex) + "\n\\bottomrule\n\\end{tabular}\n")

    # ---- sensitivity: retraining interval of the adaptive learner (F = FRAMES_MAIN)
    model, _ = train_static_model(FRAMES_MAIN, H=H, W=W, n_phase=N_PHASE, V=V)
    sens = {}
    for R in (500, 1000, 2000):
        ha, hb = [], []
        for s in seeds:
            trace, b = gen_trace(s, n_phase=N_PHASE, V=V)
            m = phase_metrics(simulate(trace, FRAMES_MAIN, Learned(model, True, H=H, W=W, retrain_every=R)), b)
            hb.append(m["before"]["hit_ratio"])
            ha.append(m["after"]["hit_ratio"])
        sens[R] = dict(hit_before=float(np.mean(hb)), hit_after=float(np.mean(ha)))
    json.dump(dict(main=summary_main, model_info=model_info, adaptive_retrains_mean=float(np.mean(retrains)),
                   retrain_interval_sensitivity=sens, seeds=len(seeds),
                   params=dict(n_phase=N_PHASE, V=V, H=H, W=W, frames_main=FRAMES_MAIN)),
              open(f"{OUT}/summary.json", "w"), indent=2)

    # ---- Figure 1: hit ratio before / after the shift
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    x = np.arange(len(ORDER))
    hb = [summary_main[p]["hit_before"] * 100 for p in ORDER]
    ha = [summary_main[p]["hit_after"] * 100 for p in ORDER]
    ax.bar(x - 0.2, hb, 0.4, label="Before shift (locality + scan)", color="#9ecae1", edgecolor="k", lw=0.6)
    ax.bar(x + 0.2, ha, 0.4, label="After shift (random / bursty)", color="#fdae6b", edgecolor="k", lw=0.6)
    for xi, a, b_ in zip(x, hb, ha):
        ax.text(xi - 0.2, a + 0.8, f"{a:.1f}", ha="center", fontsize=7)
        ax.text(xi + 0.2, b_ + 0.8, f"{b_:.1f}", ha="center", fontsize=7)
    ax.set_xticks(x, ORDER, fontsize=8)
    ax.set_ylabel("Hit ratio (%)")
    ax.set_ylim(0, 85)
    ax.set_title(f"Hit ratio per phase ({FRAMES_MAIN} frames, mean of {len(seeds)} traces)", fontsize=9)
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_phase_hit_ratio.png", dpi=200)
    plt.close(fig)

    # ---- Figure 2: rolling hit ratio over time
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    for p in ORDER:
        y = np.mean(roll[p], axis=0) * 100
        ax.plot(np.arange(len(y)) + 500, y, label=p, color=COLORS[p], lw=1.4,
                ls="--" if p == "OPT" else "-")
    ax.axvline(N_PHASE, color="grey", ls=":", lw=1)
    ax.text(N_PHASE + 150, 8, "workload shift", fontsize=7, color="grey")
    ax.set_xlabel("Reference index")
    ax.set_ylabel("Hit ratio (%), 500-access window")
    ax.set_title("Rolling hit ratio across the shift", fontsize=9)
    ax.legend(fontsize=7, frameon=False, ncol=3, loc="upper right")
    ax.set_ylim(0, 90)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_rolling_hit_ratio.png", dpi=200)
    plt.close(fig)

    # ---- Figure 3: effect of memory size
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.0), sharey=True)
    for ax, ph, ttl in zip(axes, ("before", "after"), ("Before shift", "After shift")):
        for p in ORDER:
            y = [agg(Fv, p, ph, "hit_ratio")[0] * 100 for Fv in FRAMES_SWEEP]
            ax.plot(FRAMES_SWEEP, y, marker="o", ms=3.5, label=p, color=COLORS[p],
                    ls="--" if p == "OPT" else "-")
        ax.set_xscale("log", base=2)
        ax.set_xticks(FRAMES_SWEEP, [str(f) for f in FRAMES_SWEEP])
        ax.set_xlabel("Frames")
        ax.set_title(ttl, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Hit ratio (%)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=7, frameon=False, ncol=5, loc="lower center")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    axes[1].annotate("FIFO and LRU nearly overlap here", xy=(0.5, 0.06), xycoords="axes fraction",
                     fontsize=6.5, color="grey", ha="center")
    fig.savefig(f"{OUT}/fig_frame_sweep.png", dpi=200)
    plt.close(fig)

    print(open(f"{OUT}/table_main.md").read())
    print(json.dumps(dict(importances=model_info[FRAMES_MAIN]["importances"],
                          retrains=float(np.mean(retrains)), sens=sens), indent=1))


if __name__ == "__main__":
    main()
