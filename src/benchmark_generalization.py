"""
benchmark_generalization.py
---------------------------
Honest test: how much of Model C's adversarial gain comes from the
slur-flag sentinel vs from the normalizer alone?

We construct a second adversarial probe set using INSULTS that appear in
CONDA's training data but are NOT in our SLUR_STEMS regex. These include
gaming-specific insults Model B learned to flag:

    "trash", "noob", "idiot", "stupid", "garbage", "feeder", "useless"

For each, we apply the same six obfuscation transforms (canonical,
spacing, leet, repeat, symbols, combined). Model C still sees the input
through normalize_and_flag, but the flag will NEVER trigger because none
of these are in SLUR_STEMS. So any detection improvement on this set
comes from the normalizer's character-level repair, not the slur flag.

We compare:
  - Model C (normalizer + flag, but flag won't fire on these)
  - Model B raw (no normalization at all)
  - Model B with normalizer applied externally

If the normalizer alone is doing real work, Model C and Model-B-with-normalizer
will both beat Model B raw. If the slur flag is doing all the work, neither
of those will help on this set.
"""

from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd

from src.adversarial import normalize, normalize_and_flag, AdvProbe
from src.model_c import _preprocess
from src.train import load_pipeline


# Insults that ARE in CONDA's vocabulary (Model B learned them as toxic
# features) but are NOT in our slur regex. Generalization test material.
NON_REGEX_INSULTS = [
    "trash", "noob", "idiot", "stupid", "garbage",
    "feeder", "useless", "loser",
]


def _space(s: str) -> str:
    return " ".join(s)


def _leet(s: str) -> str:
    swaps = {"a": "4", "e": "3", "i": "1", "o": "0", "s": "$"}
    return "".join(swaps.get(c, c) for c in s)


def _repeat_tail(s: str) -> str:
    return s[:-1] + s[-1] * 5 if len(s) >= 2 else s


def _symbols(s: str) -> str:
    return ".".join(s)


def _combined(s: str) -> str:
    return " ".join(_leet(c) for c in s)


def build_generalization_probes() -> list[AdvProbe]:
    probes: list[AdvProbe] = []
    for word in NON_REGEX_INSULTS:
        probes.append(AdvProbe(word, word, "canonical", 1))
        probes.append(AdvProbe(_space(word), word, "spacing", 1))
        probes.append(AdvProbe(_leet(word), word, "leet", 1))
        probes.append(AdvProbe(_repeat_tail(word), word, "repeat", 1))
        probes.append(AdvProbe(_symbols(word), word, "symbols", 1))
        probes.append(AdvProbe(_combined(word), word, "combined", 1))
    return probes


def predict_binary_c(model_c, texts):
    processed = [_preprocess(t, "") for t in texts]
    cls = model_c.predict(processed)
    return np.isin(cls, ["E", "I"]).astype(int)


def main():
    probes = build_generalization_probes()
    print(f"Built {len(probes)} generalization probes "
          f"(words in CONDA vocab but NOT in slur regex)")

    model_b = load_pipeline("model_b_conda")
    model_c = joblib.load("models/model_c.joblib")

    raw_texts = [p.text for p in probes]
    norm_texts = [normalize(t) for t in raw_texts]
    expected = np.array([p.expected_toxic for p in probes])
    techniques = [p.technique for p in probes]
    canonicals = [p.canonical for p in probes]

    # Three conditions
    pred_b_raw = model_b.predict(raw_texts)
    pred_b_norm = model_b.predict(norm_texts)
    pred_c = predict_binary_c(model_c, raw_texts)

    df = pd.DataFrame({
        "word": canonicals,
        "obfuscated": raw_texts,
        "technique": techniques,
        "B_raw": pred_b_raw,
        "B_normalized": pred_b_norm,
        "C": pred_c,
    })

    print("\n" + "=" * 75)
    print("GENERALIZATION TEST — detection rate on obfuscated CONDA insults")
    print("(slur-flag NEVER fires on these because they're not in SLUR_STEMS)")
    print("=" * 75)
    summary = pd.DataFrame({
        "Model B raw": [df["B_raw"].mean()],
        "Model B + normalizer": [df["B_normalized"].mean()],
        "Model C": [df["C"].mean()],
    })
    show = summary.copy()
    for c in show.columns:
        show[c] = show[c].map(lambda x: f"{x:.1%}")
    print(show.to_string(index=False))

    by_tech = df.groupby("technique")[["B_raw", "B_normalized", "C"]].mean()
    print("\nBy technique:")
    print(by_tech.map(lambda x: f"{x:.0%}").to_string())

    # Per-word breakdown for richer analysis
    print("\nBy word (canonical → detection rate across techniques):")
    by_word = df.groupby("word")[["B_raw", "B_normalized", "C"]].mean()
    print(by_word.map(lambda x: f"{x:.0%}").to_string())

    os.makedirs("figures", exist_ok=True)
    df.to_csv("figures/generalization_results.csv", index=False)
    summary.to_csv("figures/generalization_summary.csv", index=False)
    print("\nSaved: figures/generalization_results.csv")


if __name__ == "__main__":
    main()
