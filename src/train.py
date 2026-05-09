"""
train.py
--------
Defines the model pipeline (TF-IDF features + Logistic Regression) and
trains one model per domain.

Why this architecture:

  - TF-IDF + Logistic Regression is a strong baseline for short-text
    classification, and crucially its coefficients are directly
    interpretable: we can read off which words push predictions toward
    "toxic" and which push toward "non-toxic". For a project whose
    central thesis is "models pick up on domain-specific cues", this
    interpretability IS the story.

  - We use class_weight='balanced' to handle class imbalance without
    resampling (important: resampling would change the meaning of the
    decision threshold across domains, which would muddy our cross-domain
    comparison).

  - We use sublinear_tf and english stop words. We allow unigrams and
    bigrams to capture short multi-word expressions ("ez game", "report
    mid", "you're trash") which are the heart of the domain-specific
    signal we want to study.

Saving:
  Trained pipelines are pickled to models/ for reuse by evaluate.py
  and probes.py.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.data_loader import RANDOM_SEED, Split, make_splits

MODELS_DIR = "models"


def make_pipeline() -> Pipeline:
    """Construct a fresh, identical TF-IDF + LogReg pipeline."""
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    sublinear_tf=True,
                    stop_words="english",
                    strip_accents="unicode",
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=1.0,
                    max_iter=2000,
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def train_one(name: str, train_split: Split) -> Pipeline:
    """Fit a fresh pipeline on the given training split."""
    print(f"Training {name} on {train_split} ...")
    pipe = make_pipeline()
    pipe.fit(train_split.X, train_split.y)
    return pipe


def save_pipeline(pipe: Pipeline, name: str) -> str:
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, f"{name}.joblib")
    joblib.dump(pipe, path)
    return path


def load_pipeline(name: str) -> Pipeline:
    path = os.path.join(MODELS_DIR, f"{name}.joblib")
    return joblib.load(path)


def train_all() -> dict[str, Pipeline]:
    """Train Model A (Davidson) and Model B (CONDA) and persist both."""
    splits = make_splits()
    models: dict[str, Pipeline] = {}

    models["model_a_davidson"] = train_one(
        "Model A (general internet, Davidson)", splits["davidson_train"]
    )
    save_pipeline(models["model_a_davidson"], "model_a_davidson")

    models["model_b_conda"] = train_one(
        "Model B (in-game, CONDA)", splits["conda_train"]
    )
    save_pipeline(models["model_b_conda"], "model_b_conda")

    print(f"\nTrained and saved {len(models)} models to {MODELS_DIR}/")
    return models


if __name__ == "__main__":
    train_all()
