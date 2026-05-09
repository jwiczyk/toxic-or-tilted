"""
benchmark_all.py
----------------
End-to-end head-to-head comparison of Models A, B, C on:

  1. The adversarial probe set (does normalization + integrated training help?)
  2. The CONDA test set (in-domain performance, binary view)
  3. The probe sets from probes.py (identity / banter / genuine toxicity)

Model A and B are binary classifiers; Model C is 4-class. To compare
fairly on toxicity detection, we collapse Model C's predictions:
toxic = (predicted class is E or I).
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from src.adversarial import build_adversarial_probes, normalize_and_flag
from src.data_loader import clean_text, make_splits
from src.model_c import _preprocess
from src.probes import (
    GAMER_BANTER_PROBES,
    GENUINE_TOXICITY_PROBES,
    IDENTITY_PROBES,
)
from src.train import load_pipeline


def predict_binary(model_name, model, texts) -> np.ndarray:
    """Return 0/1 toxicity predictions for any of A, B, or C."""
    if model_name == "C":
        # Model C is multi-class; map E and I to 1, others to 0.
        # Apply Model C's preprocessor (normalize_and_flag) to raw text.
        processed = [_preprocess(t, "") for t in texts]
        cls = model.predict(processed)
        return np.isin(cls, ["E", "I"]).astype(int)
    # A and B are binary; raw text in.
    return model.predict(texts)


def predict_proba_toxic(model_name, model, texts) -> np.ndarray:
    """Return P(toxic) for any of A, B, or C."""
    if model_name == "C":
        processed = [_preprocess(t, "") for t in texts]
        proba = model.predict_proba(processed)
        cls_order = list(model.classes_)
        idx = [cls_order.index(c) for c in ("E", "I") if c in cls_order]
        return proba[:, idx].sum(axis=1)
    return model.predict_proba(texts)[:, 1]


def adversarial_comparison() -> pd.DataFrame:
    probes = build_adversarial_probes()
    texts = [p.text for p in probes]
    expected = np.array([p.expected_toxic for p in probes])
    techniques = [p.technique for p in probes]

    models = {
        "A": load_pipeline("model_a_davidson"),
        "B": load_pipeline("model_b_conda"),
        "C": __import__("joblib").load("models/model_c.joblib"),
    }

    rows = []
    for mname, model in models.items():
        preds = predict_binary(mname, model, texts)
        for p, pred in zip(probes, preds):
            rows.append({
                "model": f"Model {mname}",
                "text": p.text,
                "technique": p.technique,
                "expected_toxic": p.expected_toxic,
                "pred_toxic": int(pred),
            })
    return pd.DataFrame(rows)


def in_domain_comparison() -> pd.DataFrame:
    """Binary toxicity F1 on CONDA test set, all three models."""
    splits = make_splits()
    test = splits["conda_test"]

    # Model C uses its own preprocessed test set with the SAME row indices.
    # For a fair comparison we just predict on the same raw test texts using
    # each model's preferred input format.
    models = {
        "A": load_pipeline("model_a_davidson"),
        "B": load_pipeline("model_b_conda"),
        "C": __import__("joblib").load("models/model_c.joblib"),
    }

    y_true = test.y
    rows = []
    for mname, model in models.items():
        y_pred = predict_binary(mname, model, test.X.tolist())
        rows.append({
            "model": f"Model {mname}",
            "test_set": "conda_test (binary)",
            "f1": f1_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "n_test": len(y_true),
            "n_toxic": int(y_true.sum()),
        })
    return pd.DataFrame(rows)


def probe_comparison() -> pd.DataFrame:
    """Compare on identity / banter / genuine-toxicity probes."""
    sets = [
        ("identity", IDENTITY_PROBES),
        ("banter", GAMER_BANTER_PROBES),
        ("genuine_toxic", GENUINE_TOXICITY_PROBES),
    ]
    models = {
        "A": load_pipeline("model_a_davidson"),
        "B": load_pipeline("model_b_conda"),
        "C": __import__("joblib").load("models/model_c.joblib"),
    }
    rows = []
    for set_name, probes in sets:
        texts = [p.text for p in probes]
        expected = np.array([p.expected_toxic for p in probes])
        for mname, model in models.items():
            preds = predict_binary(mname, model, texts)
            n = len(probes)
            if expected.sum() == 0:
                detection = float("nan")
                fp = preds.mean()
            else:
                detection = float((preds[expected == 1]).mean()) if (expected == 1).any() else float("nan")
                fp = float((preds[expected == 0]).mean()) if (expected == 0).any() else float("nan")
            rows.append({
                "model": f"Model {mname}",
                "probe_set": set_name,
                "n": n,
                "detection_rate_on_toxic": detection,
                "false_positive_rate_on_clean": fp,
            })
    return pd.DataFrame(rows)


def main() -> None:
    os.makedirs("figures", exist_ok=True)

    print("=" * 75)
    print("ADVERSARIAL PROBE SET — detection rate on toxic / FPR on controls")
    print("=" * 75)
    adv = adversarial_comparison()
    summary = adv.groupby("model").apply(lambda g: pd.Series({
        "detection_rate": g[g["expected_toxic"] == 1]["pred_toxic"].mean(),
        "false_positive": g[g["expected_toxic"] == 0]["pred_toxic"].mean(),
    }), include_groups=False).reset_index()
    summary["detection_rate"] = summary["detection_rate"].map(lambda x: f"{x:.1%}")
    summary["false_positive"] = summary["false_positive"].map(lambda x: f"{x:.1%}")
    print(summary.to_string(index=False))
    adv.to_csv("figures/adversarial_all_models.csv", index=False)

    by_tech = adv[adv["expected_toxic"] == 1].groupby(["technique", "model"])["pred_toxic"].mean().unstack()
    print("\nBy technique (detection rate):")
    print(by_tech.map(lambda x: f"{x:.0%}").to_string())

    print("\n" + "=" * 75)
    print("IN-DOMAIN — CONDA test set (binary)")
    print("=" * 75)
    ind = in_domain_comparison()
    show = ind.copy()
    for c in ("f1", "precision", "recall"):
        show[c] = show[c].map(lambda x: f"{x:.3f}")
    print(show.to_string(index=False))
    ind.to_csv("figures/in_domain_all_models.csv", index=False)

    print("\n" + "=" * 75)
    print("PROBE SETS — identity / banter / genuine toxic")
    print("=" * 75)
    pro = probe_comparison()
    show = pro.copy()
    for c in ("detection_rate_on_toxic", "false_positive_rate_on_clean"):
        show[c] = show[c].map(lambda x: "n/a" if pd.isna(x) else f"{x:.0%}")
    print(show.to_string(index=False))
    pro.to_csv("figures/probes_all_models.csv", index=False)


if __name__ == "__main__":
    main()
