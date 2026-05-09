"""
visualize_v2.py
---------------
Updated visualizations covering Models A, B, C and the reputation system.
This is run AFTER train.py, model_c.py, benchmark_all.py, and reputation.py.

Figures:
  fig_adversarial_comparison.png   Model A/B/C adversarial detection rates,
                                   by technique
  fig_model_comparison.png         Side-by-side F1 / detection / FPR table
                                   across all three models on key metrics
  fig_reputation_scatter.png       Reputation score (history) vs future
                                   toxic rate, with regression line
  fig_reputation_calibration.png   Reputation deciles vs actual future
                                   toxic rate
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

FIGURES_DIR = "figures"


def _save(fig, name: str) -> str:
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_adversarial_comparison() -> str:
    df = pd.read_csv(os.path.join(FIGURES_DIR, "adversarial_all_models.csv"))
    pos = df[df["expected_toxic"] == 1]
    by_tech_model = (
        pos.groupby(["technique", "model"])["pred_toxic"].mean().reset_index()
    )

    techniques = [
        "canonical", "spacing", "leet", "repeat", "symbols", "combined",
        "spacing_in_sentence", "leet_in_sentence", "repeat_in_sentence",
        "symbols_in_sentence", "combined_in_sentence",
    ]
    by_tech_model = by_tech_model[by_tech_model["technique"].isin(techniques)]
    by_tech_model["technique"] = pd.Categorical(
        by_tech_model["technique"], categories=techniques, ordered=True
    )

    fig, ax = plt.subplots(figsize=(13, 6))
    sns.barplot(
        data=by_tech_model,
        x="technique", y="pred_toxic", hue="model",
        palette={"Model A": "#4C72B0", "Model B": "#DD8452", "Model C": "#55A868"},
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Detection rate (higher is better)")
    ax.set_title(
        "Adversarial robustness by obfuscation technique\n"
        "Model C uses normalization preprocessor; A and B see raw text",
        fontsize=11,
    )
    ax.set_ylim(0, 1.1)
    ax.tick_params(axis="x", rotation=35)
    for tick in ax.get_xticklabels():
        tick.set_horizontalalignment("right")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return _save(fig, "fig_adversarial_comparison.png")


def plot_model_comparison() -> str:
    """Three-panel summary: in-domain F1, adversarial detection, identity bias."""
    in_dom = pd.read_csv(os.path.join(FIGURES_DIR, "in_domain_all_models.csv"))
    adv = pd.read_csv(os.path.join(FIGURES_DIR, "adversarial_all_models.csv"))
    probes = pd.read_csv(os.path.join(FIGURES_DIR, "probes_all_models.csv"))

    # Compute summary stats
    adv_pos = adv[adv["expected_toxic"] == 1].groupby("model")["pred_toxic"].mean()
    identity_fpr = probes[probes["probe_set"] == "identity"].set_index("model")[
        "false_positive_rate_on_clean"
    ]
    f1 = in_dom.set_index("model")["f1"]

    models = ["Model A", "Model B", "Model C"]
    palette = ["#4C72B0", "#DD8452", "#55A868"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    metrics = [
        ("In-domain F1\n(CONDA test set, binary)", f1, "higher is better"),
        ("Adversarial detection\n(obfuscated slurs)", adv_pos, "higher is better"),
        ("Identity-bias FPR\n(false flags on benign identity statements)",
         identity_fpr, "lower is better"),
    ]
    for ax, (title, series, hint) in zip(axes, metrics):
        values = [series.get(m, 0) for m in models]
        bars = ax.bar(models, values, color=palette)
        ax.set_title(title, fontsize=10.5)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel(hint, fontsize=9, color="gray", style="italic")
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                    f"{v:.2f}", ha="center", fontsize=10)

    fig.suptitle(
        "Model A / Model B / Model C: where each one wins and loses",
        fontsize=12, y=1.03,
    )
    fig.tight_layout()
    return _save(fig, "fig_model_comparison.png")


def plot_reputation_scatter() -> str:
    df = pd.read_csv(os.path.join(FIGURES_DIR, "reputation_table.csv"))

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(
        df["reputation"], df["future_toxic_rate"],
        alpha=0.3, s=18, c="#4C72B0", edgecolor="none",
    )
    # Regression line
    z = np.polyfit(df["reputation"], df["future_toxic_rate"], 1)
    xs = np.linspace(df["reputation"].min(), df["reputation"].max(), 50)
    ax.plot(xs, np.polyval(z, xs), color="#C44E52", linewidth=2,
            label=f"linear fit (slope={z[0]:.2f})")

    from scipy.stats import spearmanr
    rho, p = spearmanr(df["reputation"], df["future_toxic_rate"])
    ax.set_xlabel("Reputation score (computed from first 70% of player's messages)")
    ax.set_ylabel("Toxicity rate in last 30% of messages (held out)")
    ax.set_title(
        f"Reputation predicts future behavior\n"
        f"Spearman ρ = {rho:.2f}  (p = {p:.1e}, n = {len(df)} players)",
        fontsize=11,
    )
    ax.legend(loc="upper right")
    ax.grid(alpha=0.25)
    return _save(fig, "fig_reputation_scatter.png")


def plot_reputation_calibration() -> str:
    df = pd.read_csv(os.path.join(FIGURES_DIR, "reputation_table.csv"))
    df["decile"] = pd.qcut(
        df["reputation"], q=10, labels=False, duplicates="drop"
    )
    grp = df.groupby("decile").agg(
        n=("future_toxic_rate", "size"),
        mean_rep=("reputation", "mean"),
        mean_future_toxic=("future_toxic_rate", "mean"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        grp["decile"].astype(str), grp["mean_future_toxic"],
        color=plt.cm.RdYlGn_r(grp["mean_future_toxic"] / max(grp["mean_future_toxic"].max(), 0.01)),
        edgecolor="black", linewidth=0.5,
    )
    for bar, n in zip(bars, grp["n"]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"n={n}", ha="center", fontsize=8, color="gray")
    ax.set_xlabel("Reputation decile (0 = worst, 9 = best)")
    ax.set_ylabel("Mean future toxicity rate")
    ax.set_title(
        "Lower-reputation players are reliably more toxic in held-out messages",
        fontsize=11,
    )
    ax.grid(alpha=0.25, axis="y")
    return _save(fig, "fig_reputation_calibration.png")


def make_all() -> list[str]:
    paths = []
    print("Generating fig_adversarial_comparison.png ...")
    paths.append(plot_adversarial_comparison())
    print("Generating fig_model_comparison.png ...")
    paths.append(plot_model_comparison())
    print("Generating fig_reputation_scatter.png ...")
    paths.append(plot_reputation_scatter())
    print("Generating fig_reputation_calibration.png ...")
    paths.append(plot_reputation_calibration())
    print("\nDone:")
    for p in paths:
        print(f"  {p}")
    return paths


if __name__ == "__main__":
    make_all()
