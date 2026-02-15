# Used Car Price Prediction – End-to-End ML Pipeline

This project implements a production-ready machine learning pipeline to predict used car prices based on structured vehicle data.

The workflow integrates preprocessing, feature engineering, cross-validation, hyperparameter optimization, model persistence, and structured metric reporting.

---

## Business Context

Accurate vehicle pricing is critical for dealerships and automotive marketplaces to remain competitive while maximizing margins.

This project simulates a real-world pricing optimization scenario using historical vehicle attributes to predict market price.

---

## Model Performance

The model was evaluated on both training and testing datasets.

| Dataset | R² Score | MSE | MAE |
|----------|----------|----------|----------|
| Train | 0.9625 | 0.0293 | 0.1220 |
| Test  | 0.9268 | 0.0621 | 0.1479 |

### Interpretation

- Strong generalization performance (R² > 0.92 on test set)
- Low prediction error
- Minimal overfitting gap between train and test
- Stable regression performance after log transformation

---

## Modeling Approach

The pipeline includes:

- Feature engineering (Vehicle age calculation)
- Log transformation of price variables
- One-Hot Encoding for categorical variables
- Min-Max Scaling for numerical variables
- Feature selection using SelectKBest
- Linear Regression model
- 10-fold cross-validation using MAE
- Model serialization using pickle + gzip
- Metrics export in structured JSON format

---

## Project Structure

used-car-price-prediction-ml-pipeline/
│
 ├── data/
│   └── raw/
│       ├── train_data.csv
│       └── test_data.csv
│
 ├── models/
│   └── model.pkl.gz
│
 ├── outputs/
│   └── metrics.json
│
 ├── src/
│   └── model_development.ipynb
│
 ├── requirements.txt
 ├── setup.sh
 └── README.md

## Reproducibility

To replicate the results locally:

```bash
python3 -m venv .venv
source .venv/bin/activate
source setup.sh

---

## Reproducibility

To replicate the results locally:

```bash
python3 -m venv .venv
source .venv/bin/activate
source setup.sh
```

## Key Technical Highlights
	•	Modular sklearn pipeline architecture
	•	Proper separation of preprocessing and modeling
	•	Cross-validated hyperparameter optimization
	•	Log-transformed regression modeling
	•	Structured model persistence and evaluation reporting
