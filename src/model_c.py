"""
model_c.py
----------
Model C — our improved in-game toxicity classifier.

Three things distinguish Model C from Models A and B:

  1. MULTI-CLASS OUTPUT
     Predicts CONDA's 4 classes directly (E = explicit toxic, I = implicit
     toxic, A = action/order, O = other). This matters for moderation: an
     explicit threat warrants a different response than a competitive taunt
     or a tactical "report mid" call. Binary toxic/not collapses signal that
     real moderation systems need.

  2. CONVERSATION CONTEXT
     Each utterance is encoded together with the previous K (default 2)
     utterances from the same conversation, separated by a sentinel. This
     lets the model condition on whether "trash" follows "I just inted 0/12"
     (probably venting at oneself) versus following nothing (probably
     directed at a teammate). Production toxicity APIs are typically
     stateless per message and miss this.

  3. INTEGRATED ADVERSARIAL NORMALIZATION
     Inputs pass through normalize_and_flag() before vectorization. Because
     the model sees the __slur_flag__ sentinel during training, it learns
     to associate it with toxicity directly. This closes the loop on the
     adversarial robustness gap measured in benchmark_adversarial.py.

The architecture remains TF-IDF + multinomial Logistic Regression so the
model is fast, interpretable, and reproducible from a fixed seed.
"""

from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.adversarial import normalize_and_flag
from src.data_loader import RANDOM_SEED, CONDA_TRAIN_CSV, CONDA_VALID_CSV, clean_text

CONTEXT_K = 2  # number of prior utterances to attach as context
CONTEXT_SEP = " <PREV> "
MODELS_DIR = "models"
LABELS = ["O", "A", "I", "E"]  # display order: benign -> action -> implicit -> explicit


# -----------------------------------------------------------------------------
# Data construction with conversation context
# -----------------------------------------------------------------------------
def _attach_context(df: pd.DataFrame, k: int = CONTEXT_K) -> pd.DataFrame:
    """For each row, prepend the previous k utterances from the same
    conversationId, sorted by chatTime. Adds 'context_text' column."""
    df = df.copy()
    df["utterance"] = df["utterance"].fillna("").astype(str)
    df = df[df["utterance"].str.len() > 0].reset_index(drop=True)
    df = df.sort_values(["conversationId", "chatTime"]).reset_index(drop=True)
    contexts: list[str] = []
    for _, group in df.groupby("conversationId", sort=False):
        msgs = group["utterance"].tolist()
        for i in range(len(msgs)):
            prev = msgs[max(0, i - k):i]
            ctx = CONTEXT_SEP.join(prev) if prev else ""
            contexts.append(ctx)
    df["context_text"] = contexts
    return df


def _preprocess(text: str, context: str) -> str:
    """Apply cleaning + normalize_and_flag to (context, current) and join."""
    cur = normalize_and_flag(clean_text(text))
    if context:
        ctx = normalize_and_flag(clean_text(context))
        return f"{ctx} <PREV> {cur}"
    return cur


def load_conda_with_context() -> pd.DataFrame:
    parts = [pd.read_csv(p) for p in (CONDA_TRAIN_CSV, CONDA_VALID_CSV)]
    df = pd.concat(parts, ignore_index=True)
    df = _attach_context(df, k=CONTEXT_K)
    df["x"] = [
        _preprocess(t, c) for t, c in zip(df["utterance"], df["context_text"])
    ]
    df["y"] = df["intentClass"].astype(str)
    df = df[df["x"].str.len() > 0].reset_index(drop=True)
    return df[["x", "y", "playerId", "matchId", "conversationId", "chatTime"]]


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
def make_pipeline_c() -> Pipeline:
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
                    # No English stopwords here — gaming chat is too short for
                    # stopword removal to be safe; "you" and "we" matter.
                    strip_accents="unicode",
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=1.0,
                    max_iter=3000,
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def train_model_c() -> tuple[Pipeline, pd.DataFrame, pd.DataFrame]:
    df = load_conda_with_context()
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=RANDOM_SEED, stratify=df["y"]
    )
    pipe = make_pipeline_c()
    print(f"Training Model C on {len(train_df)} examples (4-class) ...")
    pipe.fit(train_df["x"].to_numpy(), train_df["y"].to_numpy())

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(pipe, os.path.join(MODELS_DIR, "model_c.joblib"))

    # Save splits with metadata so reputation system can reuse them
    train_df.to_parquet(os.path.join(MODELS_DIR, "model_c_train.parquet"))
    test_df.to_parquet(os.path.join(MODELS_DIR, "model_c_test.parquet"))
    return pipe, train_df, test_df


def evaluate_model_c(pipe: Pipeline, test_df: pd.DataFrame) -> None:
    y_true = test_df["y"].to_numpy()
    y_pred = pipe.predict(test_df["x"].to_numpy())
    print("\n=== Model C: 4-class classification report ===")
    print(classification_report(y_true, y_pred, digits=3, labels=LABELS))

    # Binary view: is_toxic = (E or I)
    y_true_bin = np.isin(y_true, ["E", "I"]).astype(int)
    y_pred_bin = np.isin(y_pred, ["E", "I"]).astype(int)
    print(f"Binary (E|I vs O|A) F1: {f1_score(y_true_bin, y_pred_bin):.3f}")
    print(f"Macro F1 across 4 classes: "
          f"{f1_score(y_true, y_pred, average='macro'):.3f}")


if __name__ == "__main__":
    pipe, train_df, test_df = train_model_c()
    evaluate_model_c(pipe, test_df)
