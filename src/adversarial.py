"""
adversarial.py
--------------
Pre-processing module that normalizes adversarially obfuscated text back
toward a canonical form so downstream classifiers can recognize it.

Real-world motivation: in competitive online games, players routinely
evade text-based moderation by obfuscating slurs and threats in
predictable ways:

    "retard"   ->  "r e t a r d"
    "faggot"   ->  "f@gg0t"
    "nigger"   ->  "n1gg3r"
    "kys"      ->  "kyssss"
    "fag"      ->  "f.a.g"

Off-the-shelf TF-IDF and most BERT-based moderation systems treat each
of these as a distinct token from the canonical form, so the "flag this
slur" lesson doesn't transfer. This module applies a stack of
conservative, well-known normalization transforms before classification:

  1. Lowercase
  2. De-space single-letter-spaced sequences ("r e t a r d" -> "retard")
  3. Collapse internal punctuation between letters ("k.i.l.l" -> "kill")
  4. Collapse 3+ repeated chars to 2 ("retardddd" -> "retardd")
  5. Conservative leet-speak character substitution
  6. Detect any known slur form using a fuzzy-anchor regex over the
     normalized string

Two integration points:
  normalize(text)            -> str
  normalize_and_flag(text)   -> str (appends a __slur_flag__ sentinel
                                     when a known slur is detected)

DESIGN NOTE: We deliberately keep the slur stem list short, focused, and
clearly documented rather than scraping a comprehensive lexicon. The goal
is to demonstrate methodology and produce measurable lift on a constructed
adversarial probe set, not to ship a production filter (which would also
need active maintenance to keep up with new evasion patterns).
"""

from __future__ import annotations

import re
from dataclasses import dataclass


# Leet-speak mappings. Conservative on purpose.
_LEET_MAP = str.maketrans({
    "0": "o", "1": "i", "3": "e", "4": "a",
    "@": "a", "$": "s", "!": "i",
})

# Known slur stems (drawn from our trained-model coefficients and prior lit).
SLUR_STEMS = [
    "nigger", "nigga", "faggot", "fag", "retard",
    "tranny", "kike", "chink", "spic",
    "kys", "kms",
    "cunt",
]


def _build_slur_regex(stems: list[str]) -> re.Pattern:
    # No leading \b so that de-spacing artifacts like "aretard" still match.
    # Trailing \w{0,3}\b allows for plurals/short suffixes (retards, retarded).
    parts = [rf"{s}\w{{0,3}}\b" for s in stems]
    return re.compile("|".join(parts), flags=re.IGNORECASE)


_SLUR_REGEX = _build_slur_regex(SLUR_STEMS)
# Allow multiple spaces between letters (handles "F   a   gg") and also
# treat sequences with embedded multi-character chunks as candidates by
# checking the whitespace-stripped form too.
_PUNCT_BETWEEN_LETTERS = re.compile(r"(?<=[A-Za-z])[._\-+*~'`]+(?=[A-Za-z])")
_SINGLE_LETTER_SPACED = re.compile(r"\b(?:[A-Za-z](?:[ \-]+[A-Za-z]){2,})\b")
_REPEAT_3PLUS = re.compile(r"([A-Za-z])\1{2,}")


def _despace_token(match: re.Match) -> str:
    return re.sub(r"[ \-]+", "", match.group(0))


def normalize(text: str) -> str:
    """Apply normalization transforms to convergence. The order of
    transforms matters for some inputs (e.g. "k Y s$$$$" needs despace
    BEFORE leet expands the $'s, but "R3T 4 R d" needs leet first to
    convert 4->a). Iterating to a fixed point handles both cases."""
    if not isinstance(text, str):
        return ""
    s = text.lower()
    for _ in range(3):  # 3 passes is enough for any realistic combination
        prev = s
        s = _SINGLE_LETTER_SPACED.sub(_despace_token, s)
        s = _PUNCT_BETWEEN_LETTERS.sub("", s)
        s = _REPEAT_3PLUS.sub(r"\1\1", s)
        s = s.translate(_LEET_MAP)
        if s == prev:
            break
    return s


def contains_slur(text: str) -> bool:
    norm = normalize(text)
    if _SLUR_REGEX.search(norm):
        return True
    # Fallback: also check after stripping ALL whitespace, for cases like
    # "F   a   gg   0   T" that normalize to "f   a   gg   o   t" (multi-char
    # chunks survive despacing) but match "faggot" once whitespace is gone.
    no_ws = re.sub(r"\s+", "", norm)
    return bool(_SLUR_REGEX.search(no_ws))


def normalize_and_flag(text: str) -> str:
    norm = normalize(text)
    if _SLUR_REGEX.search(norm) or _SLUR_REGEX.search(re.sub(r"\s+", "", norm)):
        return norm + " __slur_flag__"
    return norm


# -----------------------------------------------------------------------------
# Adversarial probe construction
# -----------------------------------------------------------------------------
@dataclass
class AdvProbe:
    text: str
    canonical: str
    technique: str
    expected_toxic: int


def _space(s: str) -> str:
    return " ".join(list(s))


def _leet(s: str) -> str:
    swaps = {"a": "4", "e": "3", "i": "1", "o": "0", "s": "$"}
    return "".join(swaps.get(c, c) for c in s)


def _repeat_tail(s: str) -> str:
    if len(s) < 2:
        return s
    return s[:-1] + s[-1] * 5


def _symbols(s: str) -> str:
    return ".".join(list(s))


def _combined(s: str) -> str:
    return " ".join(_leet(c) for c in s)


def build_adversarial_probes() -> list[AdvProbe]:
    canonicals = ["retard", "faggot", "nigger", "fag", "cunt", "kys"]
    probes: list[AdvProbe] = []
    for c in canonicals:
        probes.append(AdvProbe(c, c, "canonical", 1))
        probes.append(AdvProbe(_space(c), c, "spacing", 1))
        probes.append(AdvProbe(_leet(c), c, "leet", 1))
        probes.append(AdvProbe(_repeat_tail(c), c, "repeat", 1))
        probes.append(AdvProbe(_symbols(c), c, "symbols", 1))
        probes.append(AdvProbe(_combined(c), c, "combined", 1))

    # In-context obfuscations
    probes += [
        AdvProbe("you are such a r e t a r d", "retard", "spacing_in_sentence", 1),
        AdvProbe("stop being a f@gg0t", "faggot", "leet_in_sentence", 1),
        AdvProbe("just kysssss already", "kys", "repeat_in_sentence", 1),
        AdvProbe("learn to play r.e.t.a.r.d", "retard", "symbols_in_sentence", 1),
        AdvProbe("uninstall n 1 g g 3 r", "nigger", "combined_in_sentence", 1),
    ]

    # Multi-layer combinations: stacking case mixing, leet, spacing, and
    # punctuation in the same token. These are the hardest cases — even a
    # human reader has to pause to decode them.
    probes += [
        AdvProbe("R3T 4 R d", "retard", "multilayer_case_leet_space", 1),
        AdvProbe("F@G g 0 T", "faggot", "multilayer_case_leet_space", 1),
        AdvProbe("n.1.g.g.3.r", "nigger", "multilayer_leet_symbols", 1),
        AdvProbe("R.E.T.A.R.D.D.D.D", "retard", "multilayer_symbols_repeat", 1),
        AdvProbe("k Y s$$$$", "kys", "multilayer_case_leet_space_repeat", 1),
        AdvProbe("F   a   gg   0   T", "faggot", "multilayer_case_leet_widespacing", 1),
    ]

    # Negative controls — should NOT be flagged. Replaced "ezzzzz" with
    # cleaner alternatives because "ez" is itself flagged-as-toxic by
    # CONDA's annotators, so a repeated "ez" is not a true control.
    controls = [
        ("good game everyone", "control_polite"),
        ("g g w p", "control_spacing_benign"),
        ("nooooooo", "control_repeat_benign"),     # disappointment, not insult
        ("h3ll0 t34m", "control_leet_benign"),
        ("n.i.c.e p.l.a.y", "control_symbols_benign"),
        ("5v5 ranked is rough today", "control_numbers"),
        ("yesssssss", "control_repeat_excitement"),
        ("o m g", "control_spacing_short"),
    ]
    for text, tech in controls:
        probes.append(AdvProbe(text, "", tech, 0))

    return probes


if __name__ == "__main__":
    examples = [
        "you are a retard", "you are a r e t a r d", "you are a r3t4rd",
        "f@gg0t", "f.a.g.g.o.t", "kysssssss",
        "good game", "g g w p", "h3ll0 t34m", "5v5 ranked",
    ]
    for e in examples:
        norm = normalize(e)
        flagged = contains_slur(e)
        print(f"  {e!r:40s} -> {norm!r:30s}  slur={flagged}")

    print(f"\nBuilt {len(build_adversarial_probes())} adversarial probes")
