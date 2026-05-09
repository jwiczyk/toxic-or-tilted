"""
benchmark_adversarial.py
------------------------
Measures the lift from applying our adversarial normalizer to existing
classifiers on an adversarial probe set.

For each (model, condition) pair we report:
  - detection rate on toxic probes (recall — higher better)
  - false-positive rate on benign controls (lower better)
  - by-technique breakdown (which obfuscation styles fail hardest)

Conditions tested:
  - raw            (no preprocessing)
  - normalized     (apply normalize() before predict)
  - normalize+flag (apply normalize_and_flag(), which appends a sentinel
                    token when a known slur is detected)

The "normalize+flag" condition simulates the deployment we'd actually
ship: a cheap regex-based slur detector running before a learned model,
nudging the model with an extra feature when an obfuscated slur is
detected.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from src.adversarial import (
    AdvProbe,
    build_adversarial_probes,
    normalize,
    normalize_and_flag,
)
from src.train import load_pipeline


CONDITIONS = {
    "raw": lambda x: x,
    "normalized": normalize,
    "normalize+flag": normalize_and_flag,
}


def evaluate_condition(model, probes: list[AdvProbe], transform) -> pd.DataFrame:
    rows = []
    texts = [transform(p.text) for p in probes]
    probs = model.predict_proba(texts)[:, 1]
    preds = (probs >= 0.5).astype(int)
    for p, prob, pred in zip(probes, probs, preds):
        rows.append({
            "text": p.text,
            "transformed": transform(p.text),
            "technique": p.technique,
            "expected_toxic": p.expected_toxic,
            "p_toxic": float(prob),
            "pred_toxic": int(pred),
        })
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> dict:
    pos = df[df["expected_toxic"] == 1]
    neg = df[df["expected_toxic"] == 0]
    return {
        "detection_rate_on_toxic": float(pos["pred_toxic"].mean()) if len(pos) else float("nan"),
        "false_positive_on_controls": float(neg["pred_toxic"].mean()) if len(neg) else float("nan"),
        "n_toxic": int(len(pos)),
        "n_control": int(len(neg)),
    }


def by_technique(df: pd.DataFrame) -> pd.DataFrame:
    pos = df[df["expected_toxic"] == 1].copy()
    g = pos.groupby("technique").agg(
        n=("text", "size"),
        detection_rate=("pred_toxic", "mean"),
        mean_p_toxic=("p_toxic", "mean"),
    )
    return g.reset_index()


def run() -> None:
    models = {
        "Model A (Davidson)": load_pipeline("model_a_davidson"),
        "Model B (CONDA)": load_pipeline("model_b_conda"),
    }
    probes = build_adversarial_probes()

    summary_rows = []
    by_tech_rows = []

    for mname, model in models.items():
        for cname, transform in CONDITIONS.items():
            df = evaluate_condition(model, probes, transform)
            s = summarize(df)
            s["model"] = mname
            s["condition"] = cname
            summary_rows.append(s)

            tech = by_technique(df)
            tech["model"] = mname
            tech["condition"] = cname
            by_tech_rows.append(tech)

    os.makedirs("figures", exist_ok=True)
    summary = pd.DataFrame(summary_rows)[
        ["model", "condition", "detection_rate_on_toxic",
         "false_positive_on_controls", "n_toxic", "n_control"]
    ]
    summary.to_csv("figures/adversarial_summary.csv", index=False)
    print("=" * 75)
    print("ADVERSARIAL ROBUSTNESS — SUMMARY")
    print("=" * 75)
    show = summary.copy()
    show["detection_rate_on_toxic"] = show["detection_rate_on_toxic"].map(lambda x: f"{x:.1%}")
    show["false_positive_on_controls"] = show["false_positive_on_controls"].map(lambda x: f"{x:.1%}")
    print(show.to_string(index=False))

    by_tech = pd.concat(by_tech_rows, ignore_index=True)
    by_tech.to_csv("figures/adversarial_by_technique.csv", index=False)
    print("\n" + "=" * 75)
    print("BY TECHNIQUE (detection rate)")
    print("=" * 75)
    pivot = by_tech.pivot_table(
        index="technique",
        columns=["model", "condition"],
        values="detection_rate",
        aggfunc="mean",
    )
    print(pivot.to_string(float_format=lambda x: f"{x:.0%}"))


if __name__ == "__main__":
    run()
