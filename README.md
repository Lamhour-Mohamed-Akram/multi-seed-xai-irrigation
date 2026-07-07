# Multi-Seed XAI Validation for Reliable LSTM Irrigation Control

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)

Code, data, trained models, and results for the IEEE Access manuscript:

> **Multi-Seed XAI Validation for Reliable LSTM Irrigation Control**
> M. A. Lamhour, M. Msalek, S. Ardchir, M. Ghazouani, and M. Azouazi
> LTIM, Faculty of Sciences Ben M'Sik, Hassan II University, Casablanca, Morocco

We audit training stability and explanation reliability for LSTM greenhouse irrigation controllers using multi-seed training audits (n=8, extended to n=16), bootstrap SHAP with 95% confidence intervals (200 stratified instances, 1,000 iterations), attention-augmented architectures, and MC-dropout uncertainty estimation — plus cross-dataset validation on an independent greenhouse (Tikrit University, Iraq).

## Key Findings

1. **Hidden training brittleness**: Vanilla LSTMs failed catastrophically in 7 of 8 training runs (87.5%) despite smooth validation curves, producing strongly negative test R². Failures are invisible in single-seed reporting.
2. **Attention stabilizes training**: Bahdanau attention-LSTMs converged in 6/8 seeds (75%), with the best seed reaching R² = 0.95 (MAE = 0.0021); the model selected for XAI analysis achieved R² = 0.83 (MAE = 0.003).
3. **Statistically grounded explanations**: Bootstrap SHAP identifies soil moisture as the dominant driver (97% of total importance) with non-overlapping confidence intervals (SE = 0.00005).
4. **Explanation divergence flags risk**: SHAP–attention divergence (D > 0.124, 75th percentile) marks predictions with 21% higher error, correlating with MC-dropout uncertainty (r = 0.42).
5. **Protocol portability**: The complete pipeline transferred without modification to the Iraq dataset (37,922 samples, arid climate, binary pump target): 8/8 seed convergence for both architectures, humidity dominance (96.9%), while SHAP–attention alignment proved dataset-dependent (r = 0.19 vs. r = 0.86 on Morocco).

## Repository Structure

```
├── notebooks/
│   ├── 01_data_preprocessing.ipynb        # Data cleaning, windowing, chronological split
│   ├── 02_baseline_lstm_model.ipynb       # 8-seed baseline LSTM training + MC-dropout
│   ├── 03_shap_explainability.ipynb       # Initial SHAP analysis
│   ├── 04_attention_lstm_model.ipynb      # Attention-LSTM training
│   ├── 05_attention_visualizations.ipynb  # Attention weight diagnostics
│   ├── 06_reviewer_experiments.ipynb      # Simpler baselines, MC-dropout (attention), perturbation,
│   │                                      #   hyperparameter sensitivity, latency, window sensitivity
│   ├── 06b_remaining_experiments.ipynb    # Continuation of 06 (rebuilt model loading)
│   ├── 07_architecture_comparison.ipynb   # GRU / TCN / Transformer vs. attention-LSTM (4 seeds each)
│   ├── 07b_shap_validation.ipynb          # SHAP ranking stability across subset sizes
│   ├── 08_regenerate_all_results.ipynb    # Regenerates/verifies all manuscript numbers
│   ├── 09_8seed_attention_training.ipynb  # 8-seed attention-LSTM on full data
│   ├── 10_prepare_kaggle_upload.ipynb     # Packages files for Kaggle GPU runs
│   ├── 11_kaggle_shap_200instances.ipynb  # Bootstrap SHAP, 200 stratified instances (Kaggle GPU T4)
│   └── 12_kaggle_cross_validation_iraq.ipynb  # Full protocol on IoT Agriculture 2024 (Iraq)
├── data/
│   └── data.csv                           # Greenhouse sensor dataset (947,682 records, 18 Apr-22 May 2022)
├── results/
│   ├── models/                            # Trained model weights (.h5)
│   ├── figures/                           # Manuscript figures
│   ├── tables/                            # Result tables (per-seed metrics, comparisons, sensitivity)
│   ├── shap_200/                          # 200-instance SHAP arrays + bootstrap importance (1,000 iter)
│   └── iraq/                              # Cross-dataset validation outputs (Tikrit University, Iraq)
├── requirements.txt
└── README.md
```

## Datasets

**Primary (Morocco).** 947,682 sensor records from a tomato greenhouse in Casablanca (Mediterranean climate), logged between 18 April and 22 May 2022 (34 days). The acquisition system wrote bursts of near-simultaneous readings approximately every minute, yielding dense but temporally redundant coverage. Six sensors: air temperature, relative humidity, luminosity, CO₂, soil moisture, soil temperature. Target: normalized valve opening command (continuous, 0–1). Preprocessing (interpolation of <0.4% missing/implausible values, MinMax scaling, sliding windows of 15 consecutive records, 80/20 chronological split) is performed in `notebooks/01_data_preprocessing.ipynb`.

**Cross-validation (Iraq).** [IoT Agriculture 2024](https://www.kaggle.com/datasets/wisam1985/iot-agriculture-2024) — 37,922 readings from a smart greenhouse at Tikrit University (arid climate). Features: temperature, humidity, water level, N, P, K. Target: binary pump command. Processed entirely in `notebooks/12_kaggle_cross_validation_iraq.ipynb`.

## Headline Results

### Model performance (best seed, Morocco test set)

| Model | RMSE | MAE | R² |
| ------- | ---- | --- | --- |
| Linear Regression | 0.0041 | 0.0019 | 0.967 |
| Ridge Regression | 0.0038 | 0.0018 | 0.972 |
| SVR (RBF) | 0.0374 | 0.0087 | −1.748 |
| Random Forest | 0.0280 | 0.0196 | −0.536 |
| Gradient Boosting | 0.0249 | 0.0164 | −0.223 |
| Baseline LSTM | 0.0185 | 0.0090 | 0.327 |
| **Attention-LSTM** | **0.0094** | **0.0030** | **0.827** |

Ridge outperforms both LSTMs in aggregate accuracy — the paper's contribution is the *validation protocol* (multi-seed auditing, bootstrap SHAP, divergence monitoring), not a claim of model superiority.

### Multi-seed convergence (16 seeds, 100K subset)

| Threshold | Baseline LSTM | Attention-LSTM |
| --------- | ------------- | -------------- |
| R² > 0.0 | 6/16 (37.5%) | 12/16 (75.0%) |
| R² > 0.5 | 4/16 (25.0%) | 11/16 (68.8%) |
| R² > 0.7 | 3/16 (18.8%) | 10/16 (62.5%) |

## Installation

```bash
git clone https://github.com/Lamhour-Mohamed-Akram/multi-seed-xai-irrigation.git
cd multi-seed-xai-irrigation

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Note: `data.csv` and `.npy`/`.h5`/`.pkl` files are stored with [Git LFS](https://git-lfs.com) — install it (`git lfs install`) before cloning to fetch the full files.

## Reproducing the Experiments

Run notebooks in order. Hardware notes:

- **01–09**: run locally (CPU is sufficient; full multi-seed training takes ~8 h on an Intel i7-10750H).
- **11–12**: designed for **Kaggle GPU (Tesla T4)** — bootstrap SHAP on 200 instances involves ~20M model calls. Upload the files produced by notebook 10 as a Kaggle dataset, then run 11/12 there.
- Precomputed outputs for every stage are included in `results/`, so any notebook can be inspected against its expected outputs without re-running the full pipeline.

## Citation

```bibtex
@article{lamhour2026multiseed,
  title={Multi-Seed XAI Validation for Reliable LSTM Irrigation Control},
  author={Lamhour, Mohamed Akram and Msalek, Mohamed and Ardchir, Soufiane and Ghazouani, Mohamed and Azouazi, Mohamed},
  journal={IEEE Access},
  year={2026},
  note={Under review}
}
```

## License

MIT — see [LICENSE](LICENSE).

## Contact

Mohamed Akram Lamhour — <mohamed.lamhour-etu@etu.univh2c.ma>
LTIM (Laboratoire de Technologies de l'Information et Modélisation), Hassan II University, Casablanca, Morocco
