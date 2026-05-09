"""
evaluate.py
-----------
Computes the full 2x2 cross-domain evaluation matrix:

                 Davidson test     CONDA test
   Model A          (in-domain)    (out-of-domain)
   Model B       (out-of-domain)      (in-domain)

The interesting cells are the off-diagonals. They tell us what happens
when a moderation classifier trained on one type of online text is asked
to police a very different one — exactly the situation game studios are
in when they buy off-the-shelf moderation APIs.

Reported metrics:
  - Accuracy           (with the caveat that base rates differ between
                        domains, so accuracy is not directly comparable
                        across cells)
  - Precision, Recall, F1  (class 1 = toxic)
  - False Positive Rate    (the most policy-relevant metric here:
                            FPR = of all genuinely non-toxic messages,
                            what fraction did the model wrongly flag?)
  - ROC AUC            (threshold-independent ranking quality)

Why FPR specifically: in real-world moderation, false positives are not
just an accuracy nit — they translate into innocent players being muted,
warned, or banned. The cost is asymmetric and the rubric's ethics section
will lean on this point.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from src.data_loader import Split, make_splits
from src.train import load_pipeline


@dataclass
class EvalResult:
    model_name: str
    test_set: str
    n: int
    n_toxic: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    fpr: float
    roc_auc: float
    tn: int
    fp: int
    fn: int
    tp: int


def evaluate(model: Pipeline, model_name: str, test: Split) -> EvalResult:
    y_pred = model.predict(test.X)
    y_score = model.predict_proba(test.X)[:, 1]
    tn, fp, fn, tp = confusion_matrix(test.y, y_pred, labels=[0, 1]).ravel()
    n_neg = tn + fp
    return EvalResult(
        model_name=model_name,
        test_set=test.name,
        n=len(test.y),
        n_toxic=int(test.y.sum()),
        accuracy=accuracy_score(test.y, y_pred),
        precision=precision_score(test.y, y_pred, zero_division=0),
        recall=recall_score(test.y, y_pred, zero_division=0),
        f1=f1_score(test.y, y_pred, zero_division=0),
        fpr=fp / max(n_neg, 1),
        roc_auc=roc_auc_score(test.y, y_score),
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        tp=int(tp),
    )


def run_full_matrix() -> pd.DataFrame:
    """Run all four (model, test) combinations and return as a DataFrame."""
    splits = make_splits()
    models = {
        "Model A (Davidson)": load_pipeline("model_a_davidson"),
        "Model B (CONDA)": load_pipeline("model_b_conda"),
    }

    rows = []
    for model_name, model in models.items():
        for test_name in ("davidson_test", "conda_test"):
            res = evaluate(model, model_name, splits[test_name])
            rows.append(asdict(res))

    df = pd.DataFrame(rows)
    return df


def pretty_print(df: pd.DataFrame) -> None:
    cols = ["model_name", "test_set", "accuracy", "precision", "recall",
            "f1", "fpr", "roc_auc"]
    show = df[cols].copy()
    for c in ("accuracy", "precision", "recall", "f1", "fpr", "roc_auc"):
        show[c] = show[c].map(lambda x: f"{x:.3f}")
    print(show.to_string(index=False))


if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    df = run_full_matrix()
    pretty_print(df)
    out = "figures/eval_matrix.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved full results to {out}")
