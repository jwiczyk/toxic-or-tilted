"""
probes.py
---------
Holds the curated probe sets used for the project's qualitative and
fairness analyses.

Three probe sets, each defined as a list of dicts so they're easy to
edit and easy to print as report tables:

  IDENTITY_PROBES
    Sentences that mention an identity (gender, sexual orientation,
    race, religion, disability) in a clearly NON-TOXIC way. A fair
    classifier should label all of these "not toxic" with high
    confidence. Models that systematically flag these higher are
    exhibiting identity-based bias — replicating the well-known
    finding from Dixon et al. (2018), "Measuring and Mitigating
    Unintended Bias in Text Classification".

  GAMER_BANTER_PROBES
    Sentences that look toxic to an outsider but are normal in-game
    chat: trash talk, "ez", tactical complaints, common MOBA slang.
    A model trained on general internet text will likely over-flag
    these.

  GENUINE_TOXICITY_PROBES
    Sentences that are unambiguously toxic regardless of domain:
    targeted slurs, threats, harassment. A useful moderation model
    should flag all of these with high confidence.

  LEAGUE_PROBES
    Hand-curated League of Legends-flavored utterances spanning the
    spectrum from clearly fine to clearly toxic, with the
    surface-level toxicity-to-outsider score deliberately variable.
    This is the qualitative demo set.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


@dataclass
class Probe:
    text: str
    category: str
    expected_toxic: int  # 1 = should be flagged toxic, 0 = should not


# -----------------------------------------------------------------------------
# Identity probes — should NOT be flagged. Modeled on Dixon et al. 2018.
# -----------------------------------------------------------------------------
IDENTITY_PROBES = [
    Probe("I am a woman who plays this game", "identity_gender", 0),
    Probe("I'm a girl gamer and I love mid lane", "identity_gender", 0),
    Probe("As a man I find this game fun", "identity_gender", 0),
    Probe("I am gay and I main support", "identity_lgbtq", 0),
    Probe("My boyfriend taught me to play", "identity_lgbtq", 0),
    Probe("I am a lesbian and proud of it", "identity_lgbtq", 0),
    Probe("I am transgender and this is my main account", "identity_lgbtq", 0),
    Probe("As a Black player I enjoy the lore", "identity_race", 0),
    Probe("I am Asian and learning the meta", "identity_race", 0),
    Probe("I am Hispanic and I just hit Diamond", "identity_race", 0),
    Probe("I'm Muslim and I play during Ramadan", "identity_religion", 0),
    Probe("I am Jewish and I love this team comp", "identity_religion", 0),
    Probe("I am Christian and I play for fun", "identity_religion", 0),
    Probe("My friend is autistic and an amazing player", "identity_disability", 0),
    Probe("I have ADHD and this game keeps me focused", "identity_disability", 0),
    Probe("I am deaf and use pings to communicate", "identity_disability", 0),
]

# -----------------------------------------------------------------------------
# Gamer banter — looks toxic out of context, normal in MOBA chat. The right
# answer here is debatable (and that debatability is part of the report's
# point), but a CONDA-trained model should treat these closer to non-toxic
# than a Davidson-trained model will.
# -----------------------------------------------------------------------------
GAMER_BANTER_PROBES = [
    Probe("ez game", "banter_flex", 0),
    Probe("ez mid", "banter_flex", 0),
    Probe("gg ez", "banter_flex", 0),
    Probe("noob team", "banter_complaint", 0),
    Probe("report mid feeding", "banter_report", 0),
    Probe("mid is trash this patch", "banter_complaint", 0),
    Probe("our jungler is griefing", "banter_complaint", 0),
    Probe("just ff at 15", "banter_complaint", 0),
    Probe("wp", "banter_neutral", 0),
    Probe("rekt", "banter_flex", 0),
    Probe("get good", "banter_flex", 0),
    Probe("L bot lane", "banter_complaint", 0),
    Probe("smurf detected", "banter_complaint", 0),
    Probe("uninstall the game", "banter_flex", 0),
]

# -----------------------------------------------------------------------------
# Unambiguous toxicity — should be flagged.
# -----------------------------------------------------------------------------
GENUINE_TOXICITY_PROBES = [
    Probe("kill yourself", "toxic_threat", 1),
    Probe("kys noob", "toxic_threat", 1),
    Probe("you should die", "toxic_threat", 1),
    Probe("I hope your family dies", "toxic_threat", 1),
    Probe("I will find you and hurt you", "toxic_threat", 1),
    Probe("you are worthless trash human", "toxic_personal", 1),
    Probe("women shouldn't be allowed to play", "toxic_identity", 1),
    Probe("go back to your country", "toxic_identity", 1),
]

# -----------------------------------------------------------------------------
# Qualitative League of Legends-flavored examples for the demo table.
# -----------------------------------------------------------------------------
LEAGUE_PROBES = [
    Probe("gl hf", "lol_neutral", 0),
    Probe("nice play", "lol_neutral", 0),
    Probe("ez lane", "lol_banter", 0),
    Probe("report yasuo 0/10", "lol_banter", 0),
    Probe("our adc is so bad", "lol_banter", 0),
    Probe("teemo top is cancer", "lol_idiom", 0),  # gaming idiom, not actual cancer
    Probe("ff15 we lose", "lol_banter", 0),
    Probe("just dodge next time", "lol_banter", 0),
    Probe("kindred gigadiff", "lol_banter", 0),
    Probe("uninstall the game noob", "lol_borderline", 0),
    Probe("kys jungler", "lol_toxic", 1),
    Probe("you ruin every game you trash human", "lol_toxic", 1),
    Probe("women like you shouldn't queue ranked", "lol_toxic", 1),
    Probe("I am a girl playing top lane", "lol_identity", 0),
    Probe("I'm gay and main support", "lol_identity", 0),
]


def score_probes(model: Pipeline, probes: list[Probe]) -> pd.DataFrame:
    """Run a model over a probe list and return per-row predictions + probs."""
    texts = [p.text for p in probes]
    probs = model.predict_proba(texts)[:, 1]
    preds = (probs >= 0.5).astype(int)
    return pd.DataFrame(
        {
            "text": texts,
            "category": [p.category for p in probes],
            "expected_toxic": [p.expected_toxic for p in probes],
            "pred_toxic": preds,
            "p_toxic": probs.round(3),
        }
    )


def fpr_by_category(df: pd.DataFrame) -> pd.DataFrame:
    """For probes that should be non-toxic, compute false-positive rate per category."""
    sub = df[df["expected_toxic"] == 0].copy()
    grp = sub.groupby("category").agg(
        n=("text", "size"),
        false_positives=("pred_toxic", "sum"),
        mean_p_toxic=("p_toxic", "mean"),
    )
    grp["fpr"] = grp["false_positives"] / grp["n"]
    return grp.reset_index()


if __name__ == "__main__":
    from src.train import load_pipeline

    model_a = load_pipeline("model_a_davidson")
    model_b = load_pipeline("model_b_conda")

    for label, model in [("Model A (Davidson)", model_a), ("Model B (CONDA)", model_b)]:
        print(f"\n========== {label} ==========")
        for set_name, probes in [
            ("IDENTITY", IDENTITY_PROBES),
            ("GAMER BANTER", GAMER_BANTER_PROBES),
            ("GENUINE TOXICITY", GENUINE_TOXICITY_PROBES),
        ]:
            df = score_probes(model, probes)
            n = len(df)
            should_be_clean = df[df["expected_toxic"] == 0]
            fp = int(should_be_clean["pred_toxic"].sum()) if len(should_be_clean) else 0
            should_be_toxic = df[df["expected_toxic"] == 1]
            tp = int(should_be_toxic["pred_toxic"].sum()) if len(should_be_toxic) else 0
            print(
                f"  {set_name}: n={n}  "
                f"FP={fp}/{len(should_be_clean)}  "
                f"TP={tp}/{len(should_be_toxic)}"
            )
