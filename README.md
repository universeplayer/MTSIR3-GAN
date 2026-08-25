# When Does Adversarial Refinement Help? — R3GAN for Time Series Imputation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/he-yufeng/adversarial-refinement-imputation/actions/workflows/ci.yml/badge.svg)](https://github.com/he-yufeng/adversarial-refinement-imputation/actions/workflows/ci.yml)

[中文文档](README_CN.md) | English

Code and paper for:

> **When Does Adversarial Refinement Help? A Negative Result and Open Problem in Adapting R3GAN to Time Series Imputation**
> Yufeng He (The University of Hong Kong)
> *[12th SIGKDD Workshop on Mining and Learning from Time Series (**MiLeTS 2026**)](https://kdd-milets.github.io/milets2026/).*
> 📄 Paper: [`FMGAN/paper/main_milets2026_sigconf.pdf`](FMGAN/paper/main_milets2026_sigconf.pdf) · [official workshop copy](https://kdd-milets.github.io/milets2026/accepted%20papers/16main_milets2026_sigconf%201.pdf)

---

## TL;DR

Diffusion models and transformers have largely supplanted GANs for multivariate time series imputation. [R3GAN](https://github.com/brownvc/R3GAN) (NeurIPS 2024) offers a modern regularized relativistic GAN baseline, so we asked: *does adapting that system revive adversarial refinement for time-series imputation?*

We adapt R3GAN to 1D temporal data (**R3GAN-1D**) as a coarse-to-fine **refiner** with a frequency-domain discriminator, and audit all **14 saved configurations** across 3 datasets. The configurations are heterogeneous and have one saved stochastic run each, so the result is descriptive rather than a controlled causal ablation. **The pattern is a clearly scoped negative result:**

- ✅ In all 5 saved zero/mean-start configurations, the full R3GAN-1D system reduced MAE by **48.4–70.2%**.
- ❌ The results contain 9 linear-start configurations. After excluding the one documented legacy logging anomaly, the 8 eligible configurations have mean change **−0.7%** (range **−3.0% to +1.1%**): no consistent saved-run gain beyond a plausible coarse fill.
- 🔍 One legacy AirQuality run reports **−21.9%**, but its training log records zero reconstruction loss for all 200 epochs despite a nonzero configured weight. It is shown for provenance and excluded from the aggregate.
- 🧪 These runs do **not** identify which R3GAN-1D component caused the pattern. The decisive next experiment is a matched reconstruction-only control with shared masks, train-only scaling, and multiple recorded seeds.

This is a negative-result workshop paper: the value is in exposing a baseline-strength pattern, surfacing every saved run and anomaly, and defining the control needed to test whether adversarial discrimination adds value beyond the generator's reconstruction path.

## Key result — a baseline-strength pattern in the saved runs

MAE ↓ (lower is better). `Δ` = relative MAE reduction (positive = improvement).

| Dataset     | Coarse method     | Before | After | Δ          |
|-------------|-------------------|:------:|:-----:|:----------:|
| Weather     | Zero fill         | 0.728  | 0.228 | **+68.6%** |
|             | Mean fill         | 0.728  | 0.223 | **+69.4%** |
|             | Linear interp     | 0.067  | 0.067 | +1.1%      |
| Electricity | Zero fill         | 0.832  | 0.426 | **+48.7%** |
|             | Mean fill         | 0.831  | 0.429 | **+48.4%** |
|             | Linear interp     | 0.164  | 0.165 | −0.7%      |
| AirQuality  | Zero fill         | 0.765  | 0.228 | **+70.2%** |
|             | Linear interp     | 0.151  | 0.152 | −0.4%      |

Standalone vs. established methods (Weather, 25% point-missing):

| Method               | Type         | MAE ↓ |
|----------------------|--------------|:-----:|
| BRITS                | RNN          | **0.039** |
| SAITS                | Transformer  | 0.062 |
| Linear interpolation | Simple       | 0.067 |
| R3GAN-1D + linear    | GAN refine   | 0.067 |
| R3GAN-1D standalone  | GAN          | 0.228 |

> The table shows selected configurations with absolute MAE values. Across all
> 14 saved configurations, all 5 zero/mean starts improved by 48.4–70.2%; the
> 8 aggregate-eligible linear starts averaged −0.7% and ranged from −3.0% to +1.1%.
> These heterogeneous runs are not repeated seeds or a controlled coarse-method
> ablation. The separate legacy −21.9% logging anomaly is retained in the raw
> results and excluded from that aggregate.

## Reproduce the paper tables

The raw per-run outputs live in [`FMGAN/results/`](FMGAN/results/); the tables above are regenerated from them by a single script (no GPU, no training):

```bash
python3 FMGAN/analysis.py
```

This walks `FMGAN/results/results/phase1_*/results.json` + `baseline_*.json` and prints the refinement and comparison tables. It prints **every** run and marks aggregate eligibility explicitly. Across the 8 eligible linear-start configurations, changes lie in **[−3.0%, +1.1%]** with mean **−0.7%**. The separately displayed legacy AirQuality run is a **−21.9% logging anomaly** and is retained in the detailed table but omitted by the script's named aggregate-exclusion registry. The script also rejects stored `Δ%` values that do not recompute from their saved before/after MAEs.

## Repository structure

This repo is the companion to the MiLeTS 2026 paper. The study code lives under [`FMGAN/`](FMGAN/) (**F**oundation-**M**odel-coarse + **GAN**-refiner):

```
.
├── FMGAN/
│   ├── models/r3gan_1d.py     # R3GAN-1D architecture (1D adaptation + frequency-domain discriminator)
│   ├── train_refiner.py       # coarse-to-fine refinement training
│   ├── foundation_model/      # MOMENT wrapper (a foundation-model coarse imputer)
│   ├── evaluation/            # metrics + BRITS / SAITS / CSDI baselines (PyPOTS)
│   ├── data/                  # unified loaders (point / block / subsequence missingness)
│   ├── scripts/               # experiment runners
│   ├── configs/               # default config
│   ├── results/               # raw results.json (paper tables reproduce from these)
│   ├── analysis.py            # reproduce paper tables
│   └── paper/                 # MiLeTS 2026 camera-ready (LaTeX + PDF), references, figures
├── requirements.txt
└── LICENSE                    # MIT
```

> The author's earlier undergraduate final-year project (FYP) — an R3GAN adaptation to image-style MTSI, plus SSGAN / TimesNet baselines and a Dash GUI — is **not** part of this paper; it is preserved on the [`fyp-archive`](https://github.com/he-yufeng/adversarial-refinement-imputation/tree/fyp-archive) branch.

## Setup

```bash
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Reproducing the tables (`analysis.py`) needs only Python stdlib + the bundled JSON. Re-running experiments needs the full stack (PyTorch, PyPOTS, MOMENT) and a GPU — see [`FMGAN/requirements.txt`](FMGAN/requirements.txt) and [`FMGAN/scripts/`](FMGAN/scripts/).

## Datasets

Standard public benchmarks, evaluated at 25% point-missing (MCAR):

- **Weather** — 52K timesteps, 21 meteorological features
- **Electricity** — 140K timesteps, 370 client-consumption features
- **AirQuality** — 8.7K timesteps, 36 PM2.5 stations (13% originally missing)

They are downloaded on demand via the [PyPOTS](https://github.com/WenjieDu/PyPOTS) / `tsdb` ecosystem (see `FMGAN/data/`); none are bundled.

## Limitations & open problem

Honest scope (also in the paper, expanded for camera-ready):

- The most direct control — a **reconstruction-only (discriminator-removed) ablation** — is the single most valuable next experiment; our current evidence (a reconstruction-weight sweep) is indirect.
- An **in-protocol diffusion baseline** (CSDI / FGTI on all three datasets), **multi-seed error bars**, and **block / higher-rate / MNAR** settings would further strengthen the claim.
- The open problem: under matched data handling and model controls, does the
  discriminator add useful conditional-refinement signal beyond the same
  generator trained only with reconstruction? The current saved runs cannot
  answer that causal question.

## Citation

```bibtex
@inproceedings{he2026adversarial,
  title     = {When Does Adversarial Refinement Help? A Negative Result and Open
               Problem in Adapting R3GAN to Time Series Imputation},
  author    = {He, Yufeng},
  booktitle = {12th SIGKDD Workshop on Mining and Learning from Time Series (MiLeTS)},
  year      = {2026}
}
```

## Acknowledgments & references

- **R3GAN** — Huang, Gokaslan, Kuleshov, Tompkin. *The GAN is dead; long live the GAN! A Modern GAN Baseline.* NeurIPS 2024.
- **BRITS** — Cao et al. NeurIPS 2018 · **SAITS** — Du et al. 2023 · **CSDI** — Tashiro et al. NeurIPS 2021.
- Baselines run via [PyPOTS](https://github.com/WenjieDu/PyPOTS); coarse foundation-model imputer via [MOMENT](https://github.com/moment-timeseries-foundation-model/moment).

## License

Original code is released under the [MIT License](LICENSE). The R3GAN-1D implementation is an original 1D adaptation; see the references above for the upstream ideas it builds on.

## Contact

Yufeng He — [@he-yufeng](https://github.com/he-yufeng) · he-yufeng@connect.hku.hk
