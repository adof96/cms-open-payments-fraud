# CMS Open Payments Fraud Detection

An attempt to detect possible fraud among US healthcare providers from the payments they
receive from drug and medical-device manufacturers, by cross-referencing those payments
with the federal list of providers excluded for fraud.

**Status:** this is a learning project, not a validated fraud detector. The first
supervised results were inflated by two forms of leakage. Once those were removed, the
payment-level model the Streamlit app uses scored close to random on new providers. A
reframed provider-level model does better than a specialty-only baseline, but it has not
been promoted and the app does not use it. See [Results](#results) and
[Limitations](#limitations).

## Overview and goal

The payments data has no fraud label, so one is built by matching recipients against the
exclusion list. The project then compares two approaches:

- **Supervised learning** on that derived label (XGBoost, plus a Random Forest at the
  payment level).
- **Unsupervised anomaly detection** (Isolation Forest), which never sees the label during
  training. The label is used only afterwards, to check whether the most anomalous payments
  include known excluded providers.

## Data sources

- **CMS Open Payments, Program Year 2024, General Payments**: payments from manufacturers
  to physicians and other recipients. This is the source of every feature.
- **LEIE (List of Excluded Individuals/Entities)** from HHS-OIG: providers excluded from
  federal healthcare programs. This is the source of the label.

Download URLs live in [src/config.py](src/config.py).

**Labels.** `is_excluded` is a strict match on NPI (the provider's national identifier)
against the LEIE. It marks 3,809 payments, 0.025% of the data. About 89% of LEIE records
have no NPI, so they can never match, which caps how many excluded providers the label can
find. A second, exploratory fuzzy name-match label (`is_excluded_name_match`) is built
separately. It is never used for training.

**Raw data is never committed.** Everything under `data/raw/` and `data/processed/` is
gitignored, except `data/processed/dashboard_summary.json`, which the app needs. To
regenerate the raw data, run `python -m src.data.load_data`; it downloads both sources into
`data/raw/cms_open_payments/` and `data/raw/leie/` and skips files already present.
`python -m src.data.clean_data_cache` deletes the downloaded raw files to free space. All
large files are read in chunks, because the project was built on a machine with limited RAM.

## Pipeline

| Step | Module | Output |
|---|---|---|
| Download raw data | `src/data/load_data.py` | `data/raw/...` |
| Build labels | `src/data/clean_data.py` | `data/processed/fraud_labels.csv` |
| Build payment-level features | `src/features/build_features.py` | `data/processed/fraud_features.csv` |
| Exploratory analysis | `notebooks/eda_fraud_features.ipynb` | — |
| Supervised training (XGBoost + Random Forest) | `src/models/train_supervised.py` | `models/*.pkl` (candidate names by default) |
| Unsupervised training (Isolation Forest) | `src/models/train_unsupervised.py` | `models/*.pkl` (candidate names by default) |
| Threshold tuning | `src/models/tune_threshold.py` | `docs/threshold_tuning_pr_curve.png` |
| Full-dataset inference | `src/inference/predictor.py` | `data/processed/xgboost_predictions.csv` |
| Dashboard summary | `src/inference/build_dashboard_summary.py` | `data/processed/dashboard_summary.json` |
| Streamlit app | `streamlit_app/app.py` | — |
| Provider-level features | `src/features/build_provider_features.py` | `data/processed/provider_features.csv`, `data/processed/provider_manufacturer_amounts.parquet` |
| Provider-level model | `src/models/train_provider_level.py` | `models/xgboost_provider_level.pkl` (candidate) |
| Provider-level capacity experiment | `src/models/provider_level_capacity.py` | `models/xgboost_provider_level_*.pkl` (candidates) |
| Provider-level cross-validated selection | `src/models/provider_level_cv.py` | `models/xgboost_provider_level_cv.pkl` (candidate) |

Diagnostic scripts used in the investigation:

- `src/models/ablation_target_encoding.py`: drops one encoded feature at a time and
  checks whether manufacturer risk is just a stand-in for specialty.
- `src/models/diagnose_provider_generalization.py`: memorization check (train vs. test
  scores), single-feature baselines, and score variation across provider splits.

`src/models/evaluate_model.py`, `src/models/promote_model.py` and `src/models/tune_model.py`
are still unimplemented placeholders from the original project scaffold.

## Results

The results are listed in the order they were found, because the early numbers turned out
to be misleading.

### 1. Payment-level model (the one the app uses)

XGBoost trained and tested with a **row-level** split, where each row is one payment:

- PR-AUC **0.696**.
- At threshold 0.724 (chosen in `tune_threshold.py`): recall **0.803**, precision **0.312**
  on the row-level test set.

### 2. Two leaks

- **Target encoding leak.** Manufacturer and specialty were target-encoded on the same
  training rows the encoding was then applied to, so each row's own label partly shaped its
  feature. Encoding out-of-fold instead lowered PR-AUC to **0.617**, still on the row-level
  split.
- **Provider overlap.** `is_excluded` is a fact about a provider, not a payment. The 3,809
  excluded payments come from only **381 providers**, and with a row-level split about 90%
  of excluded test payments belonged to a provider that also had payments in training. The
  model could score well by recognizing providers it had already seen. `payment_frequency`
  (a per-provider payment count) worked as a provider fingerprint.

### 3. Provider-level split: the payment-level model collapses

When the split is made by provider, so test providers are never seen in training:

- PR-AUC **0.0165**, ROC-AUC **0.529**: nearly random.
- The same model scores PR-AUC **0.904** on its own training data, so it was memorizing
  providers rather than learning patterns that carry over to new ones.

### 4. Reframing at the provider level

One row per provider, with aggregated features (payment counts and amounts, specialty,
manufacturer concentration and risk, payment-type mix). There are **984,069** providers,
**381** of them excluded. The models were trained on a sample of **24,973** providers, with
**76** excluded providers in each test set.

### 5. Capacity control

The default XGBoost still memorized at the provider level. A heavily constrained one
(depth 2, `min_child_weight` 100), averaged over 5 provider splits:

| | Mean ROC-AUC | Mean PR-AUC |
|---|---|---|
| Constrained XGBoost | 0.750 | 0.066 |
| Specialty alone | 0.686 | 0.034 |

It beat specialty alone on all 5 splits, with no memorization gap between training and test.

### 6. Cross-validated model selection

Choosing the configuration by cross-validation on training providers only, without looking
at the test providers, gave a **depth-1** XGBoost: 5-split mean ROC-AUC **0.755**, PR-AUC
**0.0574**. The grid could not tell its configurations apart, and a logistic regression
landed in the same range. Further gains now need better features or more labeled
providers, not more tuning.

### Unsupervised: Isolation Forest

On a 2.5% random sample containing 94 excluded payments, the top 1% most anomalous payments
captured **10.6%** of them (10 of 94). The sample is small, so this number is noisy.

### Current status

No provider-level model has been promoted. The app still uses the payment-level model from
step 1.

## Limitations

- **Label recall is capped.** About 89% of LEIE records have no NPI, so the strict match
  cannot find those excluded providers.
- **Specialty is noisy per provider.** 21.7% of providers have more than one specialty
  across their payment records; the provider-level table uses each provider's most common
  one.
- **Few positives.** Only 381 excluded providers exist, and each test set has 76, so
  results vary noticeably from one split to another.
- **The app uses the payment-level model,** the one that scores near random on new
  providers.
- **The row-level numbers overstate performance.** Every payment-level figure in the app
  (precision, recall, PR-AUC, flagged counts) comes from row-level evaluation or from
  scoring data the model was trained on.
- **Placeholder defaults in the app.** Tab 2 ("Try It Yourself") fixes `payment_month` to
  6, `payment_day_of_week` to 2 and `num_payments_included` to 1. Only the
  `payment_frequency` median is learned from data.
- **The app's model cannot be retrained exactly from the current code.**
  `train_supervised.py` now uses the provider-level split by default, so it no longer
  reproduces the row-level model behind the app. TODO: record which commit produced the
  current production payment-level model.
- **Model files are not committed** (`models/*.pkl` is gitignored). On a fresh clone, Tab 1
  of the app works from the committed `dashboard_summary.json`, but Tab 2 needs
  `models/xgboost_is_excluded.pkl` and `models/supervised_preprocessing.pkl`, and given the
  previous point, retraining them produces a different model.

## How to run

### Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
pip install pyarrow   # needed by build_provider_features.py, not yet in requirements.txt
```

The required Python version is in `requires-python` in `pyproject.toml`. Running the
notebook also needs Jupyter, which is not in `requirements.txt`.

### Pipeline order

Run from the repository root:

```bash
python -m src.data.load_data                    # download raw data (cached)
python -m src.data.clean_data                   # labels
python -m src.features.build_features           # payment-level features
# notebooks/eda_fraud_features.ipynb            # optional: exploratory analysis
python -m src.models.train_supervised           # writes *_candidate.pkl (see Safety gates)
python -m src.models.train_unsupervised         # writes *_candidate.pkl (see Safety gates)
python -m src.models.tune_threshold             # needs the production-named supervised models
python -m src.inference.predictor               # needs the production-named supervised models
python -m src.inference.build_dashboard_summary
streamlit run streamlit_app/app.py
```

`tune_threshold.py` checks that the production models still reproduce their recorded
metrics, and stops if they don't. `predictor.py` uses the fixed threshold 0.724.

### Provider-level experiments

```bash
python -m src.features.build_provider_features
python -m src.models.train_provider_level
python -m src.models.provider_level_capacity
python -m src.models.provider_level_cv
```

These scripts only ever write candidate model files; no other script or the app reads
them.

### Tests

```bash
pytest tests/models tests/features/test_build_provider_features.py
```

These pass. The other test files (under `tests/data/`, `tests/inference/`,
`tests/streamlit_app/`, `tests/visualization/`, and `tests/features/test_build_features.py`)
are unimplemented placeholders from the original scaffold and currently fail.

## Safety gates

Model files with production names (no suffix) are read by other parts of the pipeline, so
overwriting them takes two explicit steps. `train_supervised.py` and `train_unsupervised.py`
follow the same rules:

| Command | Writes |
|---|---|
| `python -m src.models.train_supervised` | `*_candidate.pkl` |
| `python -m src.models.train_supervised _name` | `*_name.pkl` |
| `python -m src.models.train_supervised ""` | `*_candidate.pkl`, with a notice |
| `python -m src.models.train_supervised "" --promote` | production names, with a printed warning |
| `--promote` without `""`, or with a named suffix | error before any training |

From Python, `train_and_evaluate(artifact_suffix="")` without `promote=True` raises before
loading any data.

Promoting a new supervised model also means regenerating everything that depends on it:
the recorded metrics in `tune_threshold.py`, the threshold in `predictor.py`,
`xgboost_predictions.csv`, and `dashboard_summary.json`.
