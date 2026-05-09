"""
reputation.py
-------------
Player-level reputation scoring built on top of Model C.

For each player, we aggregate their predicted per-message toxicity into a
single reputation score in [0, 1]. Higher = better-behaved. Recent
messages weigh more (exponential decay over message index).

Why this matters:
  Per-message moderation is reactive. A reputation system is proactive —
  it builds a model of *who* tends to be toxic so that future moderation
  decisions can use a prior. Riot's Tribunal, Activision's Defiant trust
  score, and Xbox's reputation system all do versions of this internally.
  We build a transparent, fully-reproducible analogue.

Aggregation:
  rep(player) = sigmoid( -alpha * (2 * weighted_avg(severity) - 1) )

Temporal validation:
  For each player with >= 10 messages we split their messages
  chronologically: first 70% as history, last 30% as future. We then ask
  whether the reputation score computed from history predicts the
  player's actual toxicity rate in the future window.

  This mirrors how a reputation system would actually be deployed: you
  only have past behavior, you predict future behavior, you compare to
  what actually happens.
"""

from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from src.model_c import _preprocess

ALPHA = 1.5
GAMMA = 0.05
MIN_MESSAGES = 10
HISTORY_FRACTION = 0.7


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def _weighted_severity(severities: np.ndarray, gamma: float = GAMMA) -> float:
    n = len(severities)
    if n == 0:
        return 0.0
    weights = np.exp(-gamma * np.arange(n)[::-1])
    weights /= weights.sum()
    return float(np.sum(weights * severities))


def reputation_from_severities(severities: np.ndarray, alpha: float = ALPHA) -> float:
    if len(severities) == 0:
        return 0.5
    avg = _weighted_severity(severities)
    return float(_sigmoid(-alpha * (2 * avg - 1)))


def predict_severities(model, df: pd.DataFrame) -> np.ndarray:
    # df['x'] is already preprocessed by model_c._preprocess
    proba = model.predict_proba(df["x"].to_numpy())
    cls_order = list(model.classes_)
    idx = [cls_order.index(c) for c in ("E", "I") if c in cls_order]
    return proba[:, idx].sum(axis=1)


def build_player_table(df: pd.DataFrame, model) -> pd.DataFrame:
    df = df.sort_values(["playerId", "matchId", "chatTime"]).reset_index(drop=True)
    df["severity"] = predict_severities(model, df)
    df["true_toxic"] = df["y"].isin(["E", "I"]).astype(int)

    rows = []
    for pid, group in df.groupby("playerId", sort=False):
        n = len(group)
        if n < MIN_MESSAGES:
            continue
        cut = int(np.floor(n * HISTORY_FRACTION))
        history = group.iloc[:cut]
        future = group.iloc[cut:]
        if len(history) == 0 or len(future) == 0:
            continue

        rep = reputation_from_severities(history["severity"].to_numpy())
        rows.append({
            "playerId": pid,
            "n_total": n,
            "n_history": len(history),
            "n_future": len(future),
            "reputation": rep,
            "history_toxic_rate": float(history["true_toxic"].mean()),
            "future_toxic_rate": float(future["true_toxic"].mean()),
            "future_toxic_rate_binary": int(future["true_toxic"].mean() > 0),
        })
    return pd.DataFrame(rows)


def evaluate_reputation(player_df: pd.DataFrame) -> dict:
    if len(player_df) < 5:
        return {"error": "too few players"}

    rho, p = spearmanr(player_df["reputation"], player_df["future_toxic_rate"])
    y = player_df["future_toxic_rate_binary"].to_numpy()

    if len(np.unique(y)) > 1:
        auc = roc_auc_score(y, -player_df["reputation"].to_numpy())
        baseline_auc = roc_auc_score(y, player_df["history_toxic_rate"].to_numpy())
    else:
        auc = baseline_auc = float("nan")

    bottom_q = player_df["reputation"].quantile(0.10)
    worst = player_df[player_df["reputation"] <= bottom_q]

    return {
        "n_players": int(len(player_df)),
        "spearman_rho": float(rho),
        "spearman_pvalue": float(p),
        "auc_reputation": float(auc),
        "auc_history_baseline": float(baseline_auc),
        "lift_over_baseline": float(auc - baseline_auc),
        "overall_future_toxic_rate": float(player_df["future_toxic_rate"].mean()),
        "worst_decile_future_toxic_rate": float(worst["future_toxic_rate"].mean()),
    }


def main() -> None:
    from src.model_c import load_conda_with_context

    print("Loading data and Model C ...")
    df = load_conda_with_context()
    model = joblib.load("models/model_c.joblib")

    print(f"Total messages: {len(df):,}")
    print(f"Unique players: {df['playerId'].nunique():,}")

    player_df = build_player_table(df, model)
    os.makedirs("figures", exist_ok=True)
    player_df.to_csv("figures/reputation_table.csv", index=False)

    print(f"\nPlayers with >= {MIN_MESSAGES} messages: {len(player_df):,}")
    print(f"Mean reputation:           {player_df['reputation'].mean():.3f}")
    print(f"Mean future toxic rate:    {player_df['future_toxic_rate'].mean():.3f}")

    print("\n" + "=" * 60)
    print("TEMPORAL VALIDATION")
    print("=" * 60)
    metrics = evaluate_reputation(player_df)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:35s} {v:8.3f}")
        else:
            print(f"  {k:35s} {v}")


if __name__ == "__main__":
    main()
