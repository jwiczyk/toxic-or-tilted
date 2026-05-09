"""
visualize.py
------------
Generates every figure used in the report. Run as a module to write
all PNGs under figures/.

Figures produced:

  fig_confusion_matrices.png   2x2 grid of confusion matrices, one per
                               (model, test_set) cell.

  fig_fpr_comparison.png       Bar chart of in-domain vs cross-domain
                               false-positive rate for each model.
                               This is the headline figure of the report.

  fig_top_coefficients.png     Top-15 most-toxic and least-toxic words
                               by logistic regression coefficient for
                               each model side by side. Shows which
                               words each model has come to associate
                               with toxicity.

  fig_probe_heatmap.png        Heatmap of P(toxic) across probe sets
                               and models. Identity, gamer banter, and
                               genuine toxicity probes shown as rows.

All figures use a consistent palette and are sized for an 8.5x11
report page.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from src.data_loader import make_splits
from src.probes import (
    GAMER_BANTER_PROBES,
    GENUINE_TOXICITY_PROBES,
    IDENTITY_PROBES,
    score_probes,
)
from src.train import load_pipeline

FIGURES_DIR = "figures"
PALETTE = {"a": "#4C72B0", "b": "#DD8452", "neutral": "#55A868"}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _ensure_dir() -> None:
    os.makedirs(FIGURES_DIR, exist_ok=True)


def _save(fig: plt.Figure, name: str) -> str:
    _ensure_dir()
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


# -----------------------------------------------------------------------------
# 1. Confusion matrices (2x2 grid)
# -----------------------------------------------------------------------------
def plot_confusion_matrices() -> str:
    splits = make_splits()
    models = {
        "Model A (Davidson)": load_pipeline("model_a_davidson"),
        "Model B (CONDA)": load_pipeline("model_b_conda"),
    }
    test_names = ["davidson_test", "conda_test"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for i, (mname, model) in enumerate(models.items()):
        for j, tname in enumerate(test_names):
            test = splits[tname]
            y_pred = model.predict(test.X)
            cm = confusion_matrix(test.y, y_pred, labels=[0, 1])
            cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)

            ax = axes[i, j]
            sns.heatmap(
                cm_norm,
                annot=np.array(
                    [[f"{cm[r,c]}\n({cm_norm[r,c]:.1%})" for c in range(2)] for r in range(2)]
                ),
                fmt="",
                cmap="Blues",
                cbar=False,
                xticklabels=["pred: not toxic", "pred: toxic"],
                yticklabels=["true: not toxic", "true: toxic"],
                ax=ax,
                vmin=0,
                vmax=1,
                annot_kws={"size": 11},
            )
            in_domain = (i == 0 and j == 0) or (i == 1 and j == 1)
            tag = " (in-domain)" if in_domain else " (cross-domain)"
            ax.set_title(f"{mname} on {tname}{tag}", fontsize=11)
    fig.suptitle(
        "Cross-domain confusion matrices: rows are models, columns are test sets",
        fontsize=12,
        y=1.00,
    )
    fig.tight_layout()
    return _save(fig, "fig_confusion_matrices.png")


# -----------------------------------------------------------------------------
# 2. False positive rate comparison
# -----------------------------------------------------------------------------
def plot_fpr_comparison() -> str:
    df = pd.read_csv(os.path.join(FIGURES_DIR, "eval_matrix.csv"))
    # Tag in-domain vs cross-domain
    is_in_domain = []
    for _, r in df.iterrows():
        a_match = r["model_name"].startswith("Model A") and r["test_set"] == "davidson_test"
        b_match = r["model_name"].startswith("Model B") and r["test_set"] == "conda_test"
        is_in_domain.append("in-domain" if (a_match or b_match) else "cross-domain")
    df["domain"] = is_in_domain

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data=df,
        x="model_name",
        y="fpr",
        hue="domain",
        palette={"in-domain": PALETTE["neutral"], "cross-domain": "#C44E52"},
        ax=ax,
    )
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=3, fontsize=10)
    ax.set_ylabel("False Positive Rate (lower is better)")
    ax.set_xlabel("")
    ax.set_title(
        "Cross-domain deployment multiplies false-positive rates\n"
        "(False positives = innocent messages wrongly flagged as toxic)",
        fontsize=11,
    )
    ax.legend(title="")
    ax.set_ylim(0, max(df["fpr"]) * 1.25)
    return _save(fig, "fig_fpr_comparison.png")


# -----------------------------------------------------------------------------
# 3. Top-coefficient comparison
# -----------------------------------------------------------------------------
def _top_features(model, k: int = 15):
    vec = model.named_steps["tfidf"]
    clf = model.named_steps["clf"]
    feature_names = vec.get_feature_names_out()
    coefs = clf.coef_.ravel()
    top_pos_idx = np.argsort(coefs)[-k:][::-1]
    top_neg_idx = np.argsort(coefs)[:k]
    return (
        [(feature_names[i], coefs[i]) for i in top_pos_idx],
        [(feature_names[i], coefs[i]) for i in top_neg_idx],
    )


def plot_top_coefficients() -> str:
    model_a = load_pipeline("model_a_davidson")
    model_b = load_pipeline("model_b_conda")

    fig, axes = plt.subplots(1, 2, figsize=(13, 7))
    for ax, model, name, color in [
        (axes[0], model_a, "Model A (Davidson)", PALETTE["a"]),
        (axes[1], model_b, "Model B (CONDA)", PALETTE["b"]),
    ]:
        pos, _ = _top_features(model, k=15)
        words = [w for w, _ in pos][::-1]
        scores = [c for _, c in pos][::-1]
        ax.barh(words, scores, color=color)
        ax.set_title(f"{name}: top 15 toxic-leaning features", fontsize=11)
        ax.set_xlabel("logistic regression coefficient")
        ax.tick_params(axis="y", labelsize=9)
    fig.suptitle(
        "Each model has learned a different vocabulary of toxicity",
        fontsize=12,
        y=1.00,
    )
    fig.tight_layout()
    return _save(fig, "fig_top_coefficients.png")


# -----------------------------------------------------------------------------
# 4. Probe heatmap
# -----------------------------------------------------------------------------
def plot_probe_heatmap() -> str:
    model_a = load_pipeline("model_a_davidson")
    model_b = load_pipeline("model_b_conda")
    all_probes = (
        [("identity", p) for p in IDENTITY_PROBES]
        + [("banter", p) for p in GAMER_BANTER_PROBES]
        + [("toxic", p) for p in GENUINE_TOXICITY_PROBES]
    )
    rows = []
    for set_name, p in all_probes:
        row = {
            "set": set_name,
            "text": p.text,
            "expected": p.expected_toxic,
        }
        rows.append(row)
    base = pd.DataFrame(rows)

    df_a = score_probes(model_a, [p for _, p in all_probes])
    df_b = score_probes(model_b, [p for _, p in all_probes])
    base["Model A"] = df_a["p_toxic"].values
    base["Model B"] = df_b["p_toxic"].values

    # Order: identity → banter → toxic
    base = pd.concat(
        [base[base["set"] == s] for s in ("identity", "banter", "toxic")],
        ignore_index=True,
    )
    matrix = base[["Model A", "Model B"]].to_numpy()

    fig_h = max(8, 0.32 * len(base))
    fig, ax = plt.subplots(figsize=(7, fig_h))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn_r",
        vmin=0,
        vmax=1,
        cbar_kws={"label": "P(toxic)"},
        yticklabels=base["text"].values,
        xticklabels=["Model A\n(Davidson)", "Model B\n(CONDA)"],
        ax=ax,
        linewidths=0.3,
        linecolor="#eeeeee",
    )

    # Section dividers
    boundaries = [
        len(IDENTITY_PROBES),
        len(IDENTITY_PROBES) + len(GAMER_BANTER_PROBES),
    ]
    for b in boundaries:
        ax.axhline(b, color="black", linewidth=2)

    # Section labels on the right
    section_centers = [
        len(IDENTITY_PROBES) / 2,
        len(IDENTITY_PROBES) + len(GAMER_BANTER_PROBES) / 2,
        len(IDENTITY_PROBES) + len(GAMER_BANTER_PROBES) + len(GENUINE_TOXICITY_PROBES) / 2,
    ]
    section_labels = ["IDENTITY\n(should be ~0)", "GAMER BANTER\n(should be low)", "GENUINE TOXIC\n(should be ~1)"]
    for c, lbl in zip(section_centers, section_labels):
        ax.text(2.15, c, lbl, va="center", ha="left", fontsize=9, fontweight="bold")

    ax.set_title("Probe-by-probe P(toxic) for each model", fontsize=11)
    fig.tight_layout()
    return _save(fig, "fig_probe_heatmap.png")


def make_all() -> list[str]:
    paths = []
    print("Generating fig_confusion_matrices.png ...")
    paths.append(plot_confusion_matrices())
    print("Generating fig_fpr_comparison.png ...")
    paths.append(plot_fpr_comparison())
    print("Generating fig_top_coefficients.png ...")
    paths.append(plot_top_coefficients())
    print("Generating fig_probe_heatmap.png ...")
    paths.append(plot_probe_heatmap())
    print("\nDone:")
    for p in paths:
        print(f"  {p}")
    return paths


if __name__ == "__main__":
    make_all()
