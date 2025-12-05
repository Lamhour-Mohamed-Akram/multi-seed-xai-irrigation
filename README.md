# Multi-Seed XAI Validation for Reliable LSTM Irrigation Control

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)

## Overview

This repository provides the code, data, and trained models to reproduce the experiments presented in:

> **Multi-Seed XAI Validation for Reliable LSTM Irrigation Control in Greenhouse Environments**  
> *IEEE Access, 2025*

We propose a multi-seed validation framework that combines SHAP (SHapley Additive exPlanations) and attention mechanisms to identify stable, reproducible feature importance rankings for LSTM-based irrigation prediction in greenhouse environments.

## Key Contributions

1. **Multi-seed stability analysis**: Quantifies explanation consistency across 30 independent training seeds
2. **Dual XAI validation**: Cross-validates SHAP and attention-based feature rankings
3. **Domain-aligned interpretability**: Demonstrates that stable explanations align with agronomic principles
4. **Practical irrigation insights**: Soil moisture identified as dominant predictor (SHAP importance: 0.847 ± 0.012)

## Repository Structure

```
├── notebooks/
│   ├── 01_data_preprocessing.ipynb      # Data cleaning and feature engineering
│   ├── 02_baseline_lstm_model.ipynb     # Baseline LSTM training
│   ├── 03_shap_explainability.ipynb     # SHAP analysis with multi-seed validation
│   ├── 04_attention_lstm_model.ipynb    # Attention-enhanced LSTM
│   └── 05_attention_visualizations.ipynb # Attention weight analysis
├── data/
│   └── data.csv                         # Greenhouse sensor dataset (N=52,596)
├── results/
│   ├── models/                          # Trained model weights (.h5)
│   ├── figures/                         # Generated figures
│   └── tables/                          # Result tables
└── README.md
```

## Installation

```bash
# Clone the repository
git clone https://github.com/[username]/multi-seed-xai-irrigation.git
cd multi-seed-xai-irrigation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Requirements

- Python ≥ 3.8
- TensorFlow ≥ 2.10
- SHAP ≥ 0.41
- NumPy ≥ 1.21
- Pandas ≥ 1.4
- Matplotlib ≥ 3.5
- Scikit-learn ≥ 1.0

## Quick Start

Run the notebooks in sequential order:

```bash
jupyter notebook notebooks/01_data_preprocessing.ipynb
```

Or execute all analyses:

```bash
# Preprocess data
jupyter nbconvert --execute notebooks/01_data_preprocessing.ipynb

# Train baseline model
jupyter nbconvert --execute notebooks/02_baseline_lstm_model.ipynb

# Run SHAP analysis (30 seeds)
jupyter nbconvert --execute notebooks/03_shap_explainability.ipynb

# Train attention model
jupyter nbconvert --execute notebooks/04_attention_lstm_model.ipynb

# Generate attention visualizations
jupyter nbconvert --execute notebooks/05_attention_visualizations.ipynb
```

## Dataset

The dataset contains 52,596 observations from a greenhouse in Morocco, collected at 15-minute intervals:

| Feature | Description | Unit |
|---------|-------------|------|
| Temperature | Air temperature | °C |
| Humidity | Relative humidity | % |
| Soil_Moisture | Volumetric soil water content | % |
| Solar_Radiation | Incoming solar radiation | W/m² |
| Wind_Speed | Wind velocity | m/s |
| Irrigation_Amount | Target variable (water applied) | L/m² |

## Results Summary

### Multi-Seed Feature Importance Stability

| Feature | SHAP Mean ± Std | Attention Mean ± Std | Spearman ρ |
|---------|-----------------|---------------------|------------|
| Soil Moisture | 0.847 ± 0.012 | 0.312 ± 0.018 | 0.89 |
| Temperature | 0.623 ± 0.034 | 0.245 ± 0.022 | 0.84 |
| Solar Radiation | 0.456 ± 0.028 | 0.198 ± 0.015 | 0.81 |

### Model Performance

| Metric | Baseline LSTM | Attention LSTM |
|--------|---------------|----------------|
| RMSE | 0.142 | 0.138 |
| MAE | 0.098 | 0.094 |
| R² | 0.923 | 0.931 |

## Citation

If you use this code or data in your research, please cite:

```bibtex
@article{author2025multiseed,
  title={Multi-Seed XAI Validation for Reliable LSTM Irrigation Control in Greenhouse Environments},
  author={[Author Names]},
  journal={IEEE Access},
  year={2025},
  volume={},
  pages={},
  doi={}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or collaborations, please contact: mohamedakram.lamhour@edu.uiz.ac.ma

## Acknowledgments

- This research was supported by [funding agency/grant number]
- Greenhouse data collection was facilitated by [institution/collaborators]
