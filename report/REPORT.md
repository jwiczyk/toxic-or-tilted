# Toxic or Just Tilted? Auditing and Improving Toxicity Classifiers for In-Game MOBA Chat

**GenEd1187 — AI Computing and Thinking — Final Project**

## Abstract

Player toxicity is one of the defining problems of competitive online games, but the moderation tools studios use are mostly opaque to outside researchers. The largest public dataset of in-game chat (Riot's Tribunal corpus) was retired in 2014, and the dominant industrial moderation API (Google's Perspective) stopped accepting new users in February 2026 ahead of a December shutdown. So this paper asks a basic question: with the limited public data we have, how far can simple classifiers actually go? I trained two standard models, audited their failure modes on hand-curated probe sets, and then built an improved classifier ("Model C") that incorporates four-class output, conversational context, and a regex-based normalizer for adversarially obfuscated text. Model C raised in-domain F1 from 0.83 to 0.87 and detection on obfuscated slurs from 17% to 100%, while substantially improving on multi-layer obfuscations (mixed casing plus leet plus spacing plus repetition) that defeat both baselines entirely. The cost: false-positive rate on benign identity statements jumped from 12% to 50%. A separate ablation showed that most of the obfuscation-handling gain came from the generic character-level normalizer (+69 percentage points) rather than the targeted slur-flag, so the approach generalizes beyond hand-coded slurs. On top of Model C I built a player reputation system that significantly predicts future toxicity from past behavior (Spearman ρ = −0.31, p < 10⁻¹⁶, n = 687 players), but does not meaningfully outperform a simple "count past toxic messages" baseline. The practical takeaway is that small interpretable methods do real work here. The methodological one is harder: every choice of training corpus encodes its annotators' values, improvements on one fairness dimension can quietly worsen another, and the public-data picture in this field is degrading fast.

## 1. Introduction & Motivation

If you've spent any time playing League of Legends or Dota 2, you know what in-game toxicity feels like. It's casual, constant, and notoriously hard to moderate. The community discourse around it tends to fork into two complaints that sound contradictory but aren't: people complain the game is unbearably toxic, *and* people complain that they've been chat-restricted or reported for things they consider harmless. Both can be true at once. They reflect, respectively, a moderation system that misses real harm and a moderation system that misfires on benign speech. The same system can fail in both directions.

In-game toxicity matters for three downstream reasons that show up in both the academic literature and any community forum. First, it spreads. Players exposed to toxic teammates produce more of it themselves [Shores et al. 2014; Märtens et al. 2015]. A single venting teammate can shift the temperature of a whole match. Second, it drives players out. Riot's published research has tied the chat-restricted minority's behavior to the early-game churn of the rest of the playerbase. Third, when chat feels hostile, players just stop using it. They turn off all-chat, refuse to coordinate, and the game loses some of its social character. Any moderation system that addresses one of these without the other two ends up creating new problems.

What makes the moderation hard is that toxicity in MOBA chat is heavily context-dependent. The same word ("trash") can be heated rivalry between teams, gentle ribbing between friends, or genuine harassment of a struggling teammate. Competitive trash talk after a hard-won match is just not the same as targeted abuse, and a system that treats them identically will make a lot of people angry for a lot of different reasons. Meanwhile, slur-using players have learned to evade text-based moderation in predictable ways: spacing letters out, substituting digits for letters, repeating final characters, mixing in punctuation. These patterns are easy for a regex to catch and easy for an unprepared classifier to miss.

There's a second, broader reason this problem is hard right now: the public data is disappearing. The Tribunal corpus that defined the field a decade ago was retired in 2014. Google's Perspective API, the de facto industrial standard, stopped accepting new signups in February 2026 and is sunsetting the service entirely at the end of 2026. The privatization of in-game chat data, combined with the gradual retirement of the public moderation tooling that supported a generation of academic research, has left the field operating on a single mid-sized 2021 dataset. This isn't a side note. The shape of what's possible here is being constrained by what data exists at all.

This project asks: with the public data that survives, can small interpretable classifiers reach a useful operating point? The answer turns out to be a qualified yes, with consequences worth taking seriously.

**A note on terminology.** To keep this report inoffensive, I illustrate obfuscation transforms below using the placeholder word *doofus* rather than the actual slurs the system targets. The real slur stems live in `src/adversarial.py`; they cover nine standard categories of slur (anti-Black, anti-gay with a short-form variant, anti-disabled, anti-trans, anti-Jewish, anti-Asian, anti-Hispanic, gendered, and self-harm exhortations) for a total of twelve regex stems. The smallness of this list is deliberate and discussed in §4.5.

## 2. Background and Related Work

The academic study of in-game toxicity began with Riot's Tribunal system (2011–2014), in which players collectively judged reported chat logs from heavily-reported teammates. Kwak and Blackburn [2014] used the resulting ~590k judgment corpus to characterize linguistic markers of toxicity in League of Legends; Blackburn and Kwak [2014] predicted Tribunal verdicts at roughly 80% accuracy. The Tribunal corpus was the largest publicly-available in-game chat dataset ever released. Riot retired the system in 2014 and the data is no longer accessible.

Since then the field has worked with smaller datasets. The most prominent recent contribution is **CONDA** [Weld et al. 2021], a 44,869-utterance corpus from 1,921 Dota 2 matches, dual-annotated at the utterance level (E = explicit toxic, I = implicit toxic, A = action/order, O = other) and at the token level. The other widely-used resource is **Davidson et al.'s [2017]** Twitter hate-speech corpus (~25k tweets), which is general-purpose internet toxicity rather than gaming-specific. Both are public.

On the model side, Google's Perspective API [Wulczyn et al. 2017] became the de facto industrial standard for ML toxicity moderation, with thousands of platforms relying on it. Its known weaknesses include over-flagging African American Vernacular English [Sap et al. 2019], over-flagging benign mentions of identity terms [Dixon et al. 2018], and brittleness to adversarial obfuscation [Hosseini et al. 2017]. As of February 2026 Google stopped accepting new Perspective signups; the service is sunsetting at the end of 2026. The published open-source equivalent Detoxify [Hanu and Unitary 2020] uses essentially the same training data as Perspective and inherits its biases.

Player-level reputation modeling has been studied [Kwak et al. 2015], but the academic literature has not kept pace with the production systems studios run internally — Riot's Honor system, Activision's Defiant trust score, Microsoft's Reputation system. The methods are not public.

This project sits between these threads. I take what's available — CONDA and Davidson — and ask how far simple, transparent methods can go.

## 3. Methodology

I built three classifiers and a player-level aggregation system, then evaluated everything across four test conditions: in-domain CONDA test set, hand-curated probe sets covering identity / banter / genuine toxicity, an adversarial obfuscation probe set (now including multi-layer combinations), and a separate generalization probe set designed to disentangle two parts of Model C's design.

### 3.1 Datasets

I binarize CONDA's four-class labels to (E, I) → toxic and (O, A) → not-toxic for cross-domain evaluation against Davidson, while keeping the full four-class labels for Model C's training target. Davidson's three classes (hate, offensive, neither) collapse to (hate, offensive) → toxic, (neither) → not-toxic. Empty utterances are dropped. Cleaning removes URLs, real Twitter @-mentions (not embedded `@` chars, which would clobber leet substitutions like the dummy form `d@@fus`), HTML entities, and CONDA's `[SEPA]` segmentation tokens. Stylistic features (capitalization, repetition, punctuation) are deliberately preserved.

| Dataset | Items | % Toxic (binary) |
|---|---|---|
| Davidson | 24,783 | 83.2% |
| CONDA | 35,886 | 19.5% |

The base-rate gap matters for cross-domain transfer: Davidson was sampled to find toxicity, while CONDA reflects natural Dota 2 chat where toxic is a 20% minority.

A note on language: both datasets are essentially English. Davidson is 100% ASCII English Twitter. CONDA is 99.18% ASCII; the remaining 0.82% is mostly Dota's in-game emote codes (`\ue047` and similar), occasional accented characters in mostly-English words ("repórt"), and isolated foreign tokens. So all results below should be read as English-only. Generalization to Spanish, Portuguese, Korean, or Mandarin MOBA chat — all substantial player populations — is future work.

### 3.2 Model A and Model B (the standard-approach baselines)

Model A is TF-IDF + balanced-class Logistic Regression trained on Davidson; Model B is identically architected, trained on CONDA. Both use unigrams and bigrams, sublinear TF, English stopwords, `min_df=2` and `max_df=0.95`. I chose logistic regression specifically for interpretability: model coefficients tell us directly which words each model has learned to flag as toxic.

These two stand in for the two ends of standard practice — deploy a general-purpose toxicity model off-the-shelf (Model A) or train one on your own domain (Model B). Real production systems are usually one or the other.

### 3.3 Model C

Model C makes three changes to the baseline.

**Four-class output.** It predicts CONDA's four classes (E, I, A, O) directly instead of binarizing. This matters for moderation in practice: an explicit threat warrants a different response than a competitive taunt or a tactical "report mid" call, and binary toxic-or-not collapses information a real moderation system needs.

**Conversational context.** Each utterance is encoded together with the previous two utterances from the same conversation, separated by a `<PREV>` sentinel. The point is to give the model some shot at distinguishing "trash" said in response to "I just inted 0/12" (probably venting at oneself) from "trash" said with no prior context (probably directed at a teammate). Most production toxicity APIs are stateless per message and miss this entirely.

**Adversarial normalization.** Inputs pass through a regex-based normalizer that handles four obfuscation styles, illustrated using the placeholder word *doofus*: single-letter spacing (`d o o f u s` → `doofus`), conservative leet substitution (`d00fus` → `doofus`), internal punctuation (`d.o.o.f.u.s` → `doofus`), and 3+ character repetition collapse (`doofussss` → `doofuss`). The transforms are applied iteratively to a fixed point, which matters for multi-layer obfuscations; the order of transforms turns out to matter for inputs like `D0 O ff u.$` (mixed case, leet, irregular spacing, internal punctuation) that combine several techniques in a single token. When a known slur stem is detected after normalization, a `__slur_flag__` sentinel token is appended. Because the model sees this sentinel during training, it learns to associate it with toxicity directly.

Architecture is otherwise identical to Models A and B. Random seed 42 throughout, fully reproducible.

### 3.4 Player reputation system

For each player with at least 10 messages, I split their messages chronologically: the first 70% as "history," the last 30% as "future." Per-message severity scores from Model C are aggregated across history using exponential-decay weighted averaging (γ = 0.05 over message index, so the most recent message has weight 1 and earlier ones taper off), then mapped through a sigmoid into a [0, 1] reputation score. Then I asked the obvious question: does the score derived from history predict the player's actual toxicity rate in the held-out future window?

The temporal split mirrors how this kind of system would actually be deployed: you only ever have past behavior, you predict future behavior, you compare to what happens.

### 3.5 Probe sets

Five hand-curated probe sets, totaling 124 examples:

- **Identity** (16). Sentences with benign mentions of gender, sexual orientation, race, religion, or disability (e.g. "I am a girl gamer and I love mid lane"; "I am Jewish and I love this team comp"; "I am deaf and use pings to communicate"). Should not be flagged.
- **Banter** (14). Things that look toxic to outsiders but are normal MOBA chat ("ez mid", "report mid feeding"). Treated as not-toxic for FPR purposes.
- **Genuine toxicity** (8). Unambiguous threats and harassment. Should be flagged.
- **Adversarial** (55). Six obfuscation transforms applied to six canonical slur forms, plus five in-sentence variants, plus six new **multi-layer** combinations that stack two or more transforms in a single token (e.g. mixed case + leet + spacing in one obfuscation), plus eight benign-looking obfuscation controls. The original probe set used "ezzzzz" as a benign control, but since CONDA's annotators flag "ez" as implicit toxicity, this was an inconsistent test. The current control set replaces it with "noooooo" and "yesssssss" (genuine repetition not connected to in-game taunting).
- **Generalization** (48). Same six obfuscation transforms applied to eight CONDA-vocabulary insults that are *not* in our slur regex (`trash`, `noob`, `idiot`, `stupid`, `garbage`, `feeder`, `useless`, `loser`). The slur-flag sentinel will never fire on these, so any detection improvement isolates the normalizer's contribution from the slur-flag's contribution.

## 4. Results

### 4.1 Cross-domain transfer breaks the standard approach

| Model | Test set | Acc. | F1 | FPR | AUC |
|---|---|---|---|---|---|
| A (Davidson) | Davidson | .945 | .966 | .032 | .983 |
| A (Davidson) | CONDA | .781 | .408 | **.124** | .612 |
| B (CONDA) | CONDA | .935 | .829 | .035 | .946 |
| B (CONDA) | Davidson | .769 | .847 | .215 | .810 |

Model A's false-positive rate roughly quadruples when applied cross-domain. A model trained on general internet text wrongly flags 12.4% of innocent Dota 2 messages as toxic. ROC-AUC collapses from 0.983 to 0.612, just barely above chance. For a studio using an off-the-shelf moderation API, the implication is concrete: roughly one in eight innocent in-game messages will be misclassified. That is a lot of angry players.

### 4.2 The two models have learned different things, and they disagree about what "toxic" means

Logistic regression makes its features directly inspectable. Model A's top toxic-leaning features are dominated by slurs and explicit profanity. Model B's are dominated by gaming-specific insults: `ez` is its single strongest toxic indicator, followed by `noob`, `shit`, `fucking`, `idiot`, `trash`, `stupid`. The same word can play opposite roles in the two models. `ez` is almost neutral in Model A and the #1 toxic feature in Model B.

This isn't a bug. It's a faithful reflection of what each annotation process treated as toxic. Davidson's annotators were instructed to identify hate speech on Twitter. CONDA's annotators marked Dota 2 utterances including competitive taunts as implicit toxicity. Neither labeling is wrong. They're answers to different questions. Every choice of training data is a choice of values.

### 4.3 Both baselines fail on adversarial obfuscation

On the adversarial probe set, Models A and B both catch only 14.9% of obfuscated slurs. The breakdown by technique is consistent: spacing, symbol-stripping, leet substitution, and repetition each individually defeat both models, and combined obfuscations defeat them entirely. This matches Hosseini et al.'s [2017] findings on Perspective and other neural classifiers — the failure mode is general, not specific to any particular architecture.

The new **multi-layer** probes are even harsher. These stack multiple transforms in one token — `R3T 4 R d` mixes case, leet, and irregular spacing; `F a gg 0 T` stretches spacing across multiple characters; `D0 O ff u.$` (the *doofus* placeholder) layers four transforms. Both Models A and B catch **0%** of these. They are visually decodable for a human reader who's willing to slow down, and completely invisible to TF-IDF over raw text.

### 4.4 Model C: large robustness gains, real fairness costs

| Metric | Model A | Model B | Model C |
|---|---|---|---|
| In-domain F1 (CONDA, binary) | .408 | .829 | **.875** |
| Adversarial detection rate | 14.9% | 14.9% | **100.0%** |
| &nbsp;&nbsp;multi-layer subset | 0% | 0% | **100%** |
| Genuine-toxicity recall | 12% | 25% | **50%** |
| Identity-statement FPR (lower better) | **12%** | 31% | 50% |
| Banter FPR (taunts, by our coding) | **0%** | 43% | 36% |

The adversarial number is dramatic and worth scrutinizing rather than just celebrating. Every one of 16 obfuscation techniques is caught at 100%, including the new multi-layer combinations. But the slurs in the adversarial set were chosen specifically because the regex knows about them — so the question is: how much of this is the regex doing the work, and how much is the normalizer doing real generalizable repair?

The other thing in this table that should be uncomfortable is the identity-statement FPR. Model C wrongly flags 50% of benign identity-mentioning sentences, more than four times Model A's rate. Concrete examples: "I am gay and I main support" (predicted explicit toxic, P=0.84); "I am transgender and this is my main account" (predicted implicit toxic, P=0.86); "I am deaf and use pings to communicate" (predicted implicit toxic, P=0.76); "I am Jewish and I love this team comp" (predicted explicit toxic, P=0.64); "I am Asian and learning the meta" (predicted explicit toxic, P=0.59). The mechanism is plain: identity terms appear in CONDA almost exclusively in toxic context, because in-game chat doesn't include much casual self-identification. The model therefore learns that mentioning identity *is* a signal of toxicity. This is exactly the kind of replication-of-historical-bias failure mode that Dixon et al. [2018] characterized for general-purpose toxicity classifiers, recurring here in the specific in-game setting.

### 4.5 The generalization test: most of the lift is the normalizer, not the slur list

The slur stem list in our regex is small. It's twelve entries covering nine standard slur categories (anti-Black, anti-gay with a short-form variant, anti-disabled, anti-trans, anti-Jewish, anti-Asian, anti-Hispanic, gendered, and self-harm exhortations). This is realistically close to the entire space of unambiguous slurs in English game chat. A small auditable list is arguably a feature, not a coverage problem: a moderation team can explicitly approve every entry, and the list is small enough to maintain by hand. But it leaves the obvious question: how much of Model C's win is just the slur list paying off, and how much is real?

To isolate this, I built a separate probe set using eight insults that appear in CONDA's vocabulary but are *not* in our slur regex: `trash`, `noob`, `idiot`, `stupid`, `garbage`, `feeder`, `useless`, `loser`. I applied the same six obfuscation transforms. The slur-flag sentinel cannot fire on these, so any detection improvement here comes purely from the normalizer's character-level repair.

| Condition | Detection rate (n=48) |
|---|---|
| Model B raw (no normalization) | 16.7% |
| Model B with normalizer applied | **85.4%** |
| Model C (full pipeline) | 85.4% |

The normalizer alone produces a 5x lift on words the slur regex doesn't know about. Every technique except character repetition was caught at 100% by the normalizer (repetition only got 12%, since `trashhhhh` collapses to `trashh`, which is still an out-of-vocabulary token). The implication is encouraging: the obfuscation-handling story is mostly the generic normalizer, with the slur-flag adding a final boost on known slurs specifically. The gains generalize beyond hand-coded vocabulary.

### 4.6 Where Model C fails

The 4-class confusion matrix is dominated by O→{A, I, E} confusions: Model C predicts toxic on 1,103 out of 5,321 truly-O messages (over-flagging benign chat) and misses 222 of 942 truly-E messages (predicting O or A on what should be explicit toxic). Per-class F1 is 0.86 on O, 0.54 on A, 0.56 on I, and 0.62 on E.

The hardest distinction in the dataset is A vs I — tactical "report X" calls vs implicit toxicity directed at someone — and Model C makes 42 confusions between them on the test set. Some representative high-confidence errors:

**False positives Model C flags as toxic when CONDA labels them O:**
- `"fuck"` (predicted E with P=1.00). Profanity that the annotators read as venting, not aggression.
- `"so ez midi <prev> ez lol <prev> dont cryi"` (predicted I with P=0.99). The "ez" trigger doing exactly what the coefficient analysis predicted.
- `"noob shit idiot <prev> meepo wanna be"` (predicted E with P=0.98). Looks toxic to me, frankly. Annotation may be inconsistent.

**False negatives Model C predicts benign when CONDA labels them E:**
- `"report centaur please"` (predicted A, P=0.01)
- `"all report phenix chat abuse"` (predicted A, P=0.03)
- `"report low priority <prev> report this moron"` (predicted A, P=0.02)

The "report X" pattern is genuinely ambiguous in MOBA chat. Sometimes it's tactical (calling for an in-game report); sometimes it's harassment dressed up as a report; usually it's both. The model has no good way to tell, and neither do humans without more context.

### 4.7 Player reputation predicts future behavior, but a dumb baseline does too

Among 687 players with at least 10 messages, the reputation score computed from their first 70% of messages predicts the toxicity rate of their final 30% with Spearman ρ = −0.314 (p = 3.4 × 10⁻¹⁷). Players in the worst-reputation decile have a future toxic rate of 41% versus 10% for the best-reputation decile, a 4× spread. ROC-AUC for predicting "any future toxic message" is 0.656.

But there's a deflating result alongside that. A simple baseline — "what fraction of past messages were flagged toxic" — gets AUC = 0.664, slightly higher. The honest reading is that reputation modeling clearly works on this dataset, but the particular continuous-aggregation formulation I used doesn't earn its complexity over a much simpler one. If I were doing this again, I'd start with the simple baseline and only add complexity if the data demanded it.

## 5. Critical Reflection

The most useful results in this project are the failures, not the successes. Three of them are worth dwelling on.

**No choice of training data is value-neutral.** The Model A vs Model B coefficient comparison made this concrete in a way I wasn't expecting. The same word, `ez`, is essentially neutral in one model and the strongest toxic feature in another, because the people who labeled the two datasets disagreed about what toxicity is. Davidson's annotators on Twitter had no reason to flag competitive taunts. CONDA's annotators on Dota 2 had every reason to. Neither annotation choice was wrong. They were answers to different questions. Toxicity is contested — and the community's experience of their own moderation system as simultaneously "too strict" and "not strict enough" is therefore not contradictory. It's what happens when the system's implicit definition of toxicity doesn't match the community's distribution of definitions. There's no resolution to this at the dataset level. Whoever labels the data gets to define the values.

**Robustness in one dimension can come at the cost of fairness in another.** Model C's huge adversarial-robustness gains arrived bundled with a roughly 4× regression in identity-statement FPR. I didn't seek this trade-off; it fell out of the combination of an aggressive normalizer and a training set with too few benign identity-mentioning examples for the model to learn the difference. If Model C were deployed as a moderation system, the result would be predictable and bad: players who happen to mention their identity in chat — disproportionately marginalized players — would be systematically over-flagged. A real production system using these methods would need targeted data augmentation specifically for benign identity mentions to be ethically deployable. I am not reporting Model C as production-ready. I'm reporting it as a research artifact that demonstrates both the kind of robustness gains available and the kind of fairness costs they can quietly carry.

**Reputation systems are surveillance systems by another name.** That past behavior predicts future behavior is unsurprising. The harder question is what we do with the prediction. Reputation scores translate into shadow-moderation effects: lower-rep players have their messages quietly down-ranked, their reports against others taken less seriously, their queue priority shifted. Mechanisms like this exist in production today. They are also opaque — players don't know their score, don't know how it's computed, can't appeal it. Every parameter I set in the reputation system (the decay rate γ, the alpha scale, the threshold for "low reputation") is a policy decision with consequences for real players, and I set every one of them as a hyperparameter without justification. That's fine for a class project. It's not fine as a model for actual deployment.

**The data picture matters.** A separate concern about the data itself. CONDA, like every moderation dataset, was annotated by humans whose specific judgments we can't audit. Any model trained on CONDA inherits any biases its annotators brought. We caught a glimpse of this in the `ez` finding — CONDA's annotators classified competitive taunts as implicit toxicity, which is a defensible call but also a contestable one. Other annotation choices may be less defensible and we have no way to see them. The honest position is that the public-data poverty of this field — one mid-sized dataset of one game from 2021 being the most usable public corpus — makes accountability much harder than it should be. Every researcher in this area is working with the same potentially-biased corpus because it is the only one. The Tribunal corpus that defined the field a decade ago is gone. The Perspective API that built the toxicity-detection pipeline for the last decade is being switched off this year. Each retirement compounds the problem: the public data shrinks, the public tooling shrinks, and the academic literature about systems that millions of players interact with daily ends up correspondingly thinner. This isn't a story about negligence by any one party. It's the predictable consequence of moderation tooling being good business to keep proprietary.

I should also be direct about the limits of this specific work. CONDA is from one game (Dota 2), and the results here cannot establish that they transfer to League of Legends or other MOBAs without additional data. The adversarial probe set was constructed by me and is small (55 items including the new multi-layer combinations); the generalization probe set partially addresses this, but the full claim that "the normalizer generalizes" really wants a much larger and more diverse evaluation set. The reputation system is validated against CONDA's own toxicity labels rather than ground-truth player behavior in the live game ecosystem. All findings here should be read as proofs of concept under controlled conditions, not as evidence of deployment readiness.

## 6. Conclusion

The headline numbers — adversarial detection rising from 15% to 100%, including the multi-layer combinations the baselines miss completely; in-domain F1 from 0.83 to 0.87; identity-bias FPR rising from 12% to 50%; reputation Spearman ρ = −0.31 — should be read together rather than separately. They describe a system whose strengths and weaknesses are both consequences of the same design choices. There's no free lunch in moderation: every choice of training data, normalization aggressiveness, and aggregation rule is a values commitment, and the costs of those commitments fall on different players in different ways.

The methodological takeaway is that the state of public data in this field is poor and getting worse, the state of public production tooling is degrading on roughly the same timescale, and most academic work is operating on a single 2021 dataset because nothing else is available to operate on. Reproducible, transparent, simple methods of the kind used here aren't a fallback for resource-constrained projects. They're increasingly the only methods the broader research community can actually inspect.

## References

- Blackburn, J. and Kwak, H. (2014). STFU NOOB! Predicting Crowdsourced Decisions on Toxic Behavior in Online Games. *WWW*.
- Davidson, T., Warmsley, D., Macy, M., and Weber, I. (2017). Automated Hate Speech Detection and the Problem of Offensive Language. *ICWSM*.
- Dixon, L., Li, J., Sorensen, J., Thain, N., and Vasserman, L. (2018). Measuring and Mitigating Unintended Bias in Text Classification. *AIES*.
- Hanu, L. and Unitary team (2020). Detoxify. https://github.com/unitaryai/detoxify
- Hosseini, H., Kannan, S., Zhang, B., and Poovendran, R. (2017). Deceiving Google's Perspective API Built for Detecting Toxic Comments. *arXiv:1702.08138*.
- Kwak, H. and Blackburn, J. (2014). Linguistic Analysis of Toxic Behavior in an Online Video Game. *International Conference on Social Informatics*.
- Kwak, H., Blackburn, J., and Han, S. (2015). Exploring Cyberbullying and Other Toxic Behavior in Team Competition Online Games. *CHI*.
- Märtens, M., Shen, S., Iosup, A., and Kuipers, F. (2015). Toxicity Detection in Multiplayer Online Games. *NetGames*.
- Sap, M., Card, D., Gabriel, S., Choi, Y., and Smith, N. A. (2019). The Risk of Racial Bias in Hate Speech Detection. *ACL*.
- Shores, K. B., He, Y., Swanenburg, K. L., Kraut, R., and Riedl, J. (2014). The Identification of Deviance and its Impact on Retention in a Multiplayer Game. *CSCW*.
- Weld, H., Huang, G., Lee, J., Zhang, T., Wang, K., Guo, X., Long, S., Poon, J., and Han, S. C. (2021). CONDA: a CONtextual Dual-Annotated Dataset for In-Game Toxicity Understanding and Detection. *Findings of ACL-IJCNLP*.
- Wulczyn, E., Thain, N., and Dixon, L. (2017). Ex Machina: Personal Attacks Seen at Scale. *WWW*.

---

## Appendix A: Use of Generative AI

This project was developed in collaboration with Anthropic's Claude Opus, used as a coding and writing partner under the course's explicit allowance for generative AI tools.

**Claude generated:** the initial repository scaffolding; all source code (data loading, model training, evaluation, probe sets, the adversarial normalizer, Model C, the reputation system, all benchmarks, error analysis, and visualization scripts); all generated figures; a first draft of every section of this report; and the references list.

**The author contributed:** the project's core motivating questions, including the framing of in-game toxicity in terms of contagion, attrition, and communicative withdrawal; the insight that obfuscated slurs (spacing, leet, repetition, and especially multi-layer combinations) are widespread in League of Legends and worth modeling explicitly, which drove the entire normalizer design; the player-reputation system idea; the choice of League of Legends as the framing context with Dota 2 as the data proxy due to public-data availability; editorial direction on tone, scope, and the use of a placeholder word (*doofus*) rather than actual slurs in figures and text; the insight that the small size of the slur stem list is a feature rather than a coverage problem (used to reframe §4.5); the catch that "ezzzzz" as a control was inconsistent with CONDA's own labeling and needed to be replaced; the multi-layer obfuscation idea (`D0 O ff u.$`) that produced the new probe category in §4.3; the personal-experience observations woven into the Reflection; and a final review-and-edit pass on every section before submission.

The author understands the technical and ethical content of the work submitted here and takes full responsibility for it. Both the author and Claude believe the result is stronger than either could have produced alone, and weaker than either of us would prefer to claim individually.
