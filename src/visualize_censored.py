"""
visualize_censored.py
---------------------
Regenerates fig_top_coefficients with slurs censored (first letter +
asterisks), suitable for inclusion in the report. The uncensored
version is preserved at fig_top_coefficients.png.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.train import load_pipeline


SLUR_FIRST_CHARS = set("nbcfktrsd")  # heuristic: words starting with these get censored if they look like slurs

# Explicit list of words to censor (first letter + asterisks).
KNOWN_SLURS = {
    "nigger", "nigga", "niggers", "faggot", "faggots", "fag", "fags",
    "retard", "retards", "retarded", "tranny", "kike", "chink", "spic",
    "cunt", "cunts", "bitch", "bitches", "whore", "whores",
    "pussy", "pussies", "dyke", "dykes",
}


def censor(word: str) -> str:
    if word.lower() in KNOWN_SLURS:
        return word[0] + "*" * (len(word) - 1)
    return word


def get_top_features(pipe, n: int = 15) -> pd.DataFrame:
    vec = pipe.named_steps["tfidf"]
    clf = pipe.named_steps["clf"]
    feature_names = np.array(vec.get_feature_names_out())
    coefs = clf.coef_[0]

    top_pos_idx = np.argsort(coefs)[-n:][::-1]
    top_neg_idx = np.argsort(coefs)[:n]

    pos = pd.DataFrame({
        "word": [censor(w) for w in feature_names[top_pos_idx]],
        "coef": coefs[top_pos_idx],
        "side": "toxic-leaning",
    })
    neg = pd.DataFrame({
        "word": [censor(w) for w in feature_names[top_neg_idx]],
        "coef": coefs[top_neg_idx],
        "side": "benign-leaning",
    })
    return pd.concat([pos, neg], ignore_index=True)


def plot() -> str:
    model_a = load_pipeline("model_a_davidson")
    model_b = load_pipeline("model_b_conda")

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    for ax, (name, model) in zip(axes, [("Model A (Davidson)", model_a), ("Model B (CONDA)", model_b)]):
        feats = get_top_features(model, n=15)
        # Plot just the toxic-leaning side for clarity
        toxic = feats[feats["side"] == "toxic-leaning"].sort_values("coef")
        ax.barh(toxic["word"], toxic["coef"], color="#C44E52")
        ax.set_title(f"{name}: top toxic-leaning features", fontsize=11)
        ax.set_xlabel("Logistic regression coefficient")
        ax.tick_params(axis="y", labelsize=9)
        ax.grid(alpha=0.25, axis="x")

    fig.suptitle(
        "Each model has learned a different definition of toxicity\n"
        "(slur tokens censored for presentation; raw form available in figures/fig_top_coefficients.png)",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    path = "figures/fig_top_coefficients_censored.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


if __name__ == "__main__":
    print(plot())
