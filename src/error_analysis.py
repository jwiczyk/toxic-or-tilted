"""
error_analysis.py
-----------------
Where does Model C make mistakes? Two outputs:

  1. Confusion matrix across the 4 CONDA classes (E, I, A, O) on the held-out
     CONDA test set.

  2. The model's most-confident wrong predictions in each direction:
       - high-confidence false positives: predicted E or I, actually O
       - high-confidence false negatives: predicted O, actually E or I
       - confusable cases: predicted A or I when actually the other (the
         A/I distinction is the dataset's hardest category by F1).

These examples drive §4.4 / §5 of the report — they show the texture of
the failures, not just the rates.
"""

from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

import matplotlib.pyplot as plt

LABELS = ["O", "A", "I", "E"]


def main() -> None:
    test = pd.read_parquet("models/model_c_test.parquet")
    model = joblib.load("models/model_c.joblib")

    X = test["x"].to_numpy()
    y_true = test["y"].to_numpy()
    y_pred = model.predict(X)
    proba = model.predict_proba(X)
    classes = list(model.classes_)
    e_idx = classes.index("E"); i_idx = classes.index("I")
    p_toxic = proba[:, e_idx] + proba[:, i_idx]

    test = test.copy()
    test["pred"] = y_pred
    test["p_toxic"] = p_toxic

    # 1. Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    disp = ConfusionMatrixDisplay(cm, display_labels=LABELS)
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title("Model C confusion matrix (CONDA test, 4-class)\n"
                 "Rows = true class, columns = predicted",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig("figures/fig_model_c_confusion.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved figures/fig_model_c_confusion.png")

    # Per-class precision/recall/F1 from the matrix
    print("\nConfusion matrix (rows=true, cols=pred):")
    cm_df = pd.DataFrame(cm, index=LABELS, columns=LABELS)
    print(cm_df.to_string())

    print("\nPer-class metrics:")
    for i, lbl in enumerate(LABELS):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) else 0
        rec = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
        print(f"  {lbl}: precision={prec:.3f}  recall={rec:.3f}  f1={f1:.3f}  n={cm[i].sum()}")

    # 2. High-confidence errors
    print("\n" + "=" * 75)
    print("HIGH-CONFIDENCE FALSE POSITIVES (predicted toxic, true label O)")
    print("=" * 75)
    fps = test[(test["y"] == "O") & (test["pred"].isin(["E", "I"]))]
    fps_top = fps.sort_values("p_toxic", ascending=False).head(10)
    for _, r in fps_top.iterrows():
        snippet = r["x"][:90].replace("\n", " ")
        print(f"  P(toxic)={r['p_toxic']:.2f}  pred={r['pred']}  "
              f"true=O  text={snippet!r}")

    print("\n" + "=" * 75)
    print("HIGH-CONFIDENCE FALSE NEGATIVES (predicted benign, true label E or I)")
    print("=" * 75)
    fns = test[(test["y"].isin(["E", "I"])) & (test["pred"].isin(["O", "A"]))]
    fns_top = fns.sort_values("p_toxic", ascending=True).head(10)
    for _, r in fns_top.iterrows():
        snippet = r["x"][:90].replace("\n", " ")
        print(f"  P(toxic)={r['p_toxic']:.2f}  pred={r['pred']}  "
              f"true={r['y']}  text={snippet!r}")

    print("\n" + "=" * 75)
    print("A vs I CONFUSION (the dataset's hardest distinction)")
    print("=" * 75)
    ai = test[((test["y"] == "A") & (test["pred"] == "I"))
              | ((test["y"] == "I") & (test["pred"] == "A"))]
    print(f"Total A↔I confusions: {len(ai)}")
    sample = ai.sample(min(8, len(ai)), random_state=42)
    for _, r in sample.iterrows():
        snippet = r["x"][:90].replace("\n", " ")
        print(f"  pred={r['pred']}  true={r['y']}  text={snippet!r}")

    os.makedirs("figures", exist_ok=True)
    fps.to_csv("figures/error_false_positives.csv", index=False)
    fns.to_csv("figures/error_false_negatives.csv", index=False)


if __name__ == "__main__":
    main()
