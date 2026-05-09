# Toxic or Just Tilted? — Auditing and Improving Toxicity Classifiers for In-Game MOBA Chat

GenEd1187 final project. We audit two standard approaches to in-game toxicity moderation, demonstrate three systematic failure modes (cross-domain transfer, adversarial obfuscation, identity bias), then build an improved classifier and a player-level reputation system. Full report in [`report/REPORT.md`](report/REPORT.md).

## Quick start

```bash
# 1. Clone this repo
git clone https://github.com/jwiczyk/stunning-rotary-phone.git
cd stunning-rotary-phone

# 2. Install dependencies
pip install -r requirements.txt

# 3. Pull the two datasets (CONDA + Davidson)
mkdir -p data
git clone --depth 1 https://github.com/usydnlp/CONDA.git data/CONDA
git clone --depth 1 https://github.com/t-davidson/hate-speech-and-offensive-language.git data/hate-speech-and-offensive-language

# 4. Run everything end-to-end (~3 minutes on CPU)
python -m src.train                       # trains Models A and B
python -m src.model_c                     # trains Model C
python -m src.evaluate                    # cross-domain matrix
python -m src.benchmark_adversarial       # Model A/B adversarial robustness
python -m src.benchmark_all               # full A/B/C comparison
python -m src.benchmark_generalization    # ablation: normalizer vs slur-flag
python -m src.error_analysis              # Model C confusion matrix + errors
python -m src.reputation                  # player reputation temporal validation
python -m src.visualize                   # original audit figures
python -m src.visualize_v2                # Model C and reputation figures
python -m src.visualize_censored          # censored coefficients figure for report
```

All figures land in `figures/`. All trained models in `models/`. All evaluation CSVs alongside the figures.

## Reproducibility

All experiments use a fixed random seed (42) for `train_test_split` and `LogisticRegression`. Total run time on a 2020-era laptop CPU is approximately 3 minutes.

## Repository layout

```
.
├── README.md                       this file
├── requirements.txt                Python dependencies
├── report/
│   └── REPORT.md                   full 8-page report
├── src/
│   ├── data_loader.py              loads + binarizes Davidson and CONDA
│   ├── train.py                    trains Models A (Davidson) and B (CONDA)
│   ├── evaluate.py                 2x2 cross-domain evaluation matrix
│   ├── probes.py                   identity / banter / genuine-toxicity probes
│   ├── adversarial.py              normalizer + adversarial probe set
│   ├── model_c.py                  multi-class context-aware classifier
│   ├── benchmark_adversarial.py    Models A/B w/ and w/o normalization
│   ├── benchmark_all.py            full A/B/C head-to-head
│   ├── reputation.py               player reputation + temporal validation
│   ├── visualize.py                audit-phase figures
│   └── visualize_v2.py             Model C and reputation figures
├── notebooks/
│   └── demo.ipynb                  end-to-end walkthrough
├── data/                           datasets (cloned externally; not in repo)
├── models/                         trained pipeline pickles
└── figures/                        all generated figures and CSVs
```

## Key results

| Metric | Model A | Model B | Model C |
|---|---|---|---|
| In-domain F1 (CONDA, binary) | 0.408 | 0.829 | **0.874** |
| Adversarial detection rate | 17.1% | 17.1% | **100.0%** |
| Genuine-toxicity recall | 12% | 25% | **50%** |
| Identity-statement FPR (lower better) | **12%** | 31% | 50% |

Generalization ablation (obfuscated insults *not* in slur regex):

| Condition | Detection rate |
|---|---|
| Model B raw | 16.7% |
| Model B + normalizer | **85.4%** |
| Model C (full) | 85.4% |

The normalizer alone provides most of the lift; the slur-flag adds the remaining ~15pp on slurs specifically. So the gains generalize beyond hand-coded vocabulary.

Player reputation (n = 687 players, 70/30 chronological split):

- Spearman ρ = −0.31 (p = 3.4 × 10⁻¹⁷) between reputation and held-out toxic rate
- AUC = 0.656 for predicting any future toxic message  
- Worst-decile players: 41% future toxic rate vs 10% best decile (4× spread)
- Caveat: the simpler "count past toxic messages" baseline gets AUC = 0.664 — barely worse

## Datasets

We use two public datasets:

- **CONDA** [Weld et al., 2021]: ~45k Dota 2 in-game chat utterances with utterance-level (E/I/A/O) and token-level annotation. https://github.com/usydnlp/CONDA
- **Davidson** [Davidson et al., 2017]: ~25k tweets labeled hate / offensive / neither. https://github.com/t-davidson/hate-speech-and-offensive-language

Cite their work if you build on this.

## Note on Perspective API

We initially planned to benchmark against Google's Perspective API, the de-facto industrial standard. Google stopped accepting new Perspective signups in February 2026 and is sunsetting the service entirely at the end of 2026. We discuss this transition explicitly in the report as evidence that the moderation tooling landscape is unstable and that reproducible self-contained methods like the ones in this repo are increasingly important.

## Use of generative AI

Per the course's AI-tools policy, this repository was developed in collaboration with Anthropic's Claude Opus. See the disclosure section at the end of `report/REPORT.md` for a full accounting of the division of work.

## License

MIT (for the code in this repository). The CONDA and Davidson datasets retain their original licenses; see the respective upstream repositories.
