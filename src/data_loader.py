"""
data_loader.py
--------------
Loads the two domain datasets used in this project:

  1. Davidson (Davidson et al., 2017): ~25K tweets labeled as
     hate speech / offensive language / neither. Used as our
     "general internet toxicity" domain.

  2. CONDA (Weld et al., 2021): ~45K utterances from Dota 2
     in-game chat labeled at the utterance level as
     E (explicit toxic), I (implicit toxic), A (action), or O (other).
     Used as our "in-game chat" domain.

Both are binarized to a common (text, is_toxic) schema so that
classifiers trained on one domain can be evaluated on the other.

Binarization rules:
  Davidson: is_toxic = 1 if class in {0 (hate), 1 (offensive)} else 0
  CONDA:    is_toxic = 1 if intentClass in {'E', 'I'} else 0
            (Action 'A' is treated as non-toxic since it is a tactical
            request, not toxicity per se. This matches the binary task
            definition used in the original paper.)
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# -----------------------------------------------------------------------------
# Paths (relative to project root)
# -----------------------------------------------------------------------------
DAVIDSON_CSV = "data/hate-speech-and-offensive-language/data/labeled_data.csv"
CONDA_TRAIN_CSV = "data/CONDA/data/CONDA_train.csv"
CONDA_VALID_CSV = "data/CONDA/data/CONDA_valid.csv"

RANDOM_SEED = 42  # for reproducibility — fixed across the project


# -----------------------------------------------------------------------------
# Container type
# -----------------------------------------------------------------------------
@dataclass
class Split:
    """Holds an aligned (text, label) split with a human-readable name."""
    name: str
    X: np.ndarray  # 1-D array of strings
    y: np.ndarray  # 1-D array of {0, 1}

    def __repr__(self) -> str:
        n_pos = int(self.y.sum())
        return (
            f"Split(name={self.name!r}, n={len(self.X)}, "
            f"n_toxic={n_pos} ({n_pos / max(len(self.y), 1):.1%}))"
        )


# -----------------------------------------------------------------------------
# Light text cleaning
# -----------------------------------------------------------------------------
_SEPA_RE = re.compile(r"\[SEPA\]")
_URL_RE = re.compile(r"http\S+")
_USER_RE = re.compile(r"(?<![A-Za-z0-9])@\w+")
_RT_RE = re.compile(r"^RT[\s:]*", flags=re.IGNORECASE)
_HTML_AMP_RE = re.compile(r"&amp;")
_WS_RE = re.compile(r"\s+")


def clean_text(s: str) -> str:
    """
    Conservative text cleaning shared across both domains.

    We deliberately keep this light: aggressive normalization (lowercasing,
    stripping punctuation) would erase exactly the kind of stylistic signal
    we want our models to be able to use ("EZ", "?????", "!!!!!!"). We
    handle the obvious noise (URLs, @-mentions, HTML entities, CONDA's
    [SEPA] segmentation token) and leave the rest to the vectorizer.
    """
    if not isinstance(s, str):
        return ""
    s = _SEPA_RE.sub(" ", s)
    s = _URL_RE.sub(" ", s)
    s = _USER_RE.sub(" ", s)
    s = _RT_RE.sub("", s)
    s = _HTML_AMP_RE.sub("&", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


# -----------------------------------------------------------------------------
# Loaders
# -----------------------------------------------------------------------------
def load_davidson(path: str = DAVIDSON_CSV) -> pd.DataFrame:
    """Return Davidson data with binarized 'is_toxic' column and cleaned text."""
    df = pd.read_csv(path)
    df = df.rename(columns={"tweet": "text", "class": "davidson_class"})
    # class 0 = hate speech, 1 = offensive language, 2 = neither
    df["is_toxic"] = (df["davidson_class"] != 2).astype(int)
    df["text"] = df["text"].astype(str).map(clean_text)
    df = df[df["text"].str.len() > 0].reset_index(drop=True)
    return df[["text", "is_toxic", "davidson_class"]]


def load_conda(
    train_path: str = CONDA_TRAIN_CSV,
    valid_path: str = CONDA_VALID_CSV,
) -> pd.DataFrame:
    """
    Return CONDA train+valid combined with binarized 'is_toxic' column.

    We combine train+valid because the CONDA test set is unannotated
    (held out for the official Codalab competition). We will re-split
    locally for our own train/test.
    """
    parts = []
    for p in (train_path, valid_path):
        df = pd.read_csv(p)
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)
    df = df.rename(columns={"utterance": "text", "intentClass": "conda_class"})
    # E = explicit toxic, I = implicit toxic, A = action request, O = other
    df["is_toxic"] = df["conda_class"].isin(["E", "I"]).astype(int)
    df["text"] = df["text"].astype(str).map(clean_text)
    df = df[df["text"].str.len() > 0].reset_index(drop=True)
    return df[["text", "is_toxic", "conda_class"]]


# -----------------------------------------------------------------------------
# Splits
# -----------------------------------------------------------------------------
def make_splits(test_size: float = 0.2, seed: int = RANDOM_SEED) -> dict[str, Split]:
    """
    Build the four splits the project needs:

      davidson_train / davidson_test  — for training/evaluating Model A
      conda_train    / conda_test     — for training/evaluating Model B

    Cross-domain evaluation is done by taking any model and applying it to
    the *_test split of the OTHER domain.
    """
    splits: dict[str, Split] = {}

    dav = load_davidson()
    X_tr, X_te, y_tr, y_te = train_test_split(
        dav["text"].to_numpy(),
        dav["is_toxic"].to_numpy(),
        test_size=test_size,
        random_state=seed,
        stratify=dav["is_toxic"].to_numpy(),
    )
    splits["davidson_train"] = Split("davidson_train", X_tr, y_tr)
    splits["davidson_test"] = Split("davidson_test", X_te, y_te)

    con = load_conda()
    X_tr, X_te, y_tr, y_te = train_test_split(
        con["text"].to_numpy(),
        con["is_toxic"].to_numpy(),
        test_size=test_size,
        random_state=seed,
        stratify=con["is_toxic"].to_numpy(),
    )
    splits["conda_train"] = Split("conda_train", X_tr, y_tr)
    splits["conda_test"] = Split("conda_test", X_te, y_te)

    return splits


if __name__ == "__main__":
    # Smoke test: load everything and print summary
    splits = make_splits()
    print("Loaded splits:")
    for s in splits.values():
        print(f"  {s}")
