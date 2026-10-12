# EV Battery SOH and RUL Prognostics

An auditable machine-learning framework for estimating lithium-ion battery State of Health (SOH) and Remaining Useful Life (RUL) from NASA Ames Prognostics Center of Excellence (PCoE) laboratory aging data. The system parses original MATLAB files, derives causal cycle-level features and degradation targets, evaluates regressors via strict leave-one-battery-out (LOBO) cross-validation, and provides an interactive Streamlit dashboard for exploration, audit inspection, and real-time inference.

NASA describes this experiment as a laboratory testbed with differing operational discharge conditions and an End-of-Life (EOL) criterion of 30% rated-capacity fade (2.0 Ah down to 1.4 Ah). These laboratory data do not establish EV fleet or pack-level performance. See the [official NASA repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) and its [linked archive](https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip).

**Dataset citation:** B. Saha and K. Goebel (2007), "Battery Data Set", NASA Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA.

---

## Project Layout

```text
SOH_RUL_predictor/
├── app/
│   ├── streamlit_app.py          # Interactive dashboard entry point
│   └── components/               # Cached data loader, upload validation, Plotly visualizers
├── data/
│   ├── raw/                      # Place official NASA .mat files here (Git-ignored)
│   └── processed/                # Preprocessed cycle feature and target tables (Git-ignored)
├── models/                       # Final demonstration model fits and schema metadata (Git-ignored)
├── reports/
│   ├── figures/                  # Exploratory and degradation plots (Git-ignored)
│   └── results/                  # LOBO metrics, OOF predictions, and model comparisons (Git-ignored)
├── scripts/                      # CLI entry points (inspect_dataset.py, prepare_data.py, run_experiments.py)
├── src/ev_battery/               # Core library: parser, features, targets, models, evaluation
├── tests/                        # 30-test automated verification suite
├── requirements.txt              # Core runtime dependencies
└── requirements-optional.txt     # Optional modeling packages (XGBoost, PyTorch, SHAP)
```

Raw `.mat` files, processed tables, model weights, figures, and result tables are excluded by `.gitignore`. You can verify path exclusions with:

```bash
git check-ignore -v data/raw/your-file.mat
git ls-files 'data/raw/**/*.mat'
```

---

## 🚀 Getting Started (Step-by-Step Guide)

Tested with Python 3.13.5, NumPy 2.2.6, SciPy 1.16.0, pandas 2.2.3, scikit-learn 1.6.1, Streamlit 1.45.1, and Plotly 6.1.2. Optional libraries include XGBoost 3.0.2, PyTorch 2.7.1, and SHAP 0.48.0.

Follow these exact steps from the project root after cloning or pulling the repository:

### Step 1: Create and Activate Virtual Environment

```bash
# macOS / Linux:
python3 -m venv .venv
source .venv/bin/activate

# Windows (Command Prompt / PowerShell):
# python -m venv .venv
# .venv\Scripts\activate

# Upgrade pip and install core dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .

# Optional (recommended): enables XGBoost, PyTorch (LSTM), and SHAP
python -m pip install -r requirements-optional.txt
```

### Step 2: Download and Extract NASA Battery Dataset

Because raw experimental files (`.mat`) are excluded from Git, download the official NASA PCoE battery dataset and place the extracted files into `data/raw/`:

**Option A — Terminal download (macOS/Linux):**
```bash
# Download official archive directly from NASA PHM repository
curl --fail --location --retry 3 \
  'https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip' \
  --output data/raw/nasa_battery.zip

# Unzip archive into data/raw/
unzip -q data/raw/nasa_battery.zip -d data/raw/

# Extract nested battery archives (e.g., 1. BatteryAgingARC_25-44.zip, etc.)
find data/raw -name "*.zip" ! -name "nasa_battery.zip" -exec unzip -qo {} -d data/raw/ \;
```

**Option B — Manual download:**
1. Visit the [official NASA PCoE repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/).
2. Under **5. Batteries**, download the [Battery Data Set archive](https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip).
3. Extract the downloaded zip and any nested zip archives directly into `data/raw/` so that `.mat` files (e.g., `B0005.mat`, `B0006.mat`, etc.) reside in `data/raw/`.

### Step 3: Inspect Dataset Integrity

Scan all `.mat` files in `data/raw/` to verify structure schemas and discover valid battery records:
```bash
python scripts/inspect_dataset.py --output reports/inspection.json
```

### Step 4: Extract Features & Prepare Datasets

Process raw discharge/charge records into clean tabular cycle features, handle operational anomalies, and generate SOH and RUL ground-truth targets:
```bash
python scripts/prepare_data.py
```
*Outputs generated:* `data/processed/cycle_features.csv`, `soh_dataset.csv`, `rul_dataset.csv`, `operation_inventory.csv`, `impedance_records.csv`, and EDA figures in `reports/figures/`.

*(Optional flags: `--raw-dir`, `--output-dir`, `--reports-dir`, `--figures-dir`)*

### Step 5: Run Automated Verification Suite

Run all 30 unit tests to verify parsing logic, causal feature invariance, target computation, and LOBO splitting integrity:
```bash
python -m unittest discover -s tests -v
```
*(All 30 unit tests should pass with `OK`)*

### Step 6: Train & Evaluate Models (LOBO Cross-Validation)

Run the full Leave-One-Battery-Out (LOBO) experimental protocol across all holdout batteries:
```bash
python scripts/run_experiments.py
```
*Outputs generated:* `reports/results/soh_metrics.csv`, `rul_metrics.csv`, `fold_metrics.csv`, `oof_predictions.csv`, `common_support_metrics.csv`, `feature_importance.csv`, `model_comparison.json`, and final deployable models in `models/`.

*(Optional flags: `--data-dir`, `--results-dir`, `--models-dir`, `--seed`)*

### Step 7: Launch Interactive Streamlit Dashboard

Start the local web application:
```bash
streamlit run app/streamlit_app.py
```
The dashboard will open automatically in your browser at `http://localhost:8501`.

---

## Targets, Features, and Validation Protocol

- **State of Health (SOH):** Defined as `100 × measured discharge capacity (Ah) / reference capacity (Ah)`, reported in percentage points (% SOH). The default reference capacity is 2.0 Ah (optional `first_valid` reference uses each battery's first recorded capacity).
- **Remaining Useful Life (RUL):** Defined as `EOL discharge-cycle index − current discharge-cycle index`, in cycles. The default EOL threshold is the first recorded capacity ≤ 70% rated capacity (1.4 Ah). Batteries that never reach the EOL threshold are strictly right-censored rather than given artificial targets. Cells that start below threshold are flagged.
- **Causal Feature Engineering:** Every row represents a single discharge cycle. Predictors include cycle index, ambient temperature, charge/discharge voltage and current aggregates, and thermal dynamics. RUL predictors add observed discharge capacity and historical capacity lag features. Current discharge capacity is strictly excluded from SOH predictors to avoid trivial leakage.
- **Strict Leave-One-Battery-Out (LOBO) Evaluation:** In each outer fold, all observations belonging to an entire battery are held out. Inner hyperparameter tuning executes group-wise splits exclusively among remaining training batteries. No validation battery data ever leaks into scaler fitting, imputation, or hyperparameter selection.
- **Evaluated Models:** Mean Baseline, Ridge Regression, Random Forest Regressor, Support Vector Regression (SVR, RBF kernel), XGBoost Regressor, and a causal Sequence LSTM (5-step window).

---

## Generated Artifacts

Preprocessing (`python scripts/prepare_data.py`) outputs:
- `data/processed/cycle_features.csv`, `soh_dataset.csv`, `rul_dataset.csv`
- `data/processed/operation_inventory.csv`, `impedance_records.csv`
- `reports/results/data_quality_report.json`
- `reports/figures/capacity_degradation.png`, `soh_trajectory.png`, `feature_relationships.png`

Training (`python scripts/run_experiments.py`) outputs:
- `reports/results/soh_metrics.csv`, `rul_metrics.csv`, `fold_metrics.csv`
- `reports/results/oof_predictions.csv`, `common_support_metrics.csv`, `feature_importance.csv`
- `reports/results/model_comparison.json` (comprehensive benchmarks and embedded metric audit)
- Saved demonstration models (`*.joblib`, `*.pt`) and feature schema metadata (`*.json`) in `models/`

---

## Benchmark Results

Evaluated across 34 batteries (2,794 discharge cycles for SOH, 930 for RUL) under LOBO cross-validation:

| Target | Best Model (Macro MAE) | Benchmark Rankings (Macro MAE) | Baseline MAE |
| :--- | :--- | :--- | :--- |
| **SOH** (% points) | **SVR**: 4.27% | Random Forest: 5.32% • LSTM: 5.65% • XGBoost: 5.66% • Ridge: 7.20% | 19.45% |
| **RUL** (cycles) | **SVR**: 11.52 cycles | XGBoost: 13.72 • Random Forest: 13.77 • LSTM: 15.94 • Ridge: 25.23 | 31.30 cycles |

*Note: SVR with an RBF kernel demonstrates superior generalization on small cohort cell data, preventing overfitting to battery-specific impedance variations.*

---

## Interactive Streamlit Dashboard

Launch the local web application:
```bash
streamlit run app/streamlit_app.py
```

The application provides six functional modules:
1. **Project Overview**: Problem context, dataset architecture, processing pipeline status, and artifact readiness checks.
2. **Battery Data Explorer**: Interactive degradation curves, voltage/current profile drilldowns, and multi-cell comparisons.
3. **Model Performance**: LOBO benchmark metrics (MAE, RMSE, R²), actual vs. predicted parity plots, and residual error distributions.
4. **Prediction Explorer**: Real-time inference on uploaded cycle telemetry with schema verification, unit confirmation, and median imputation fallback.
5. **Feature Importance**: Gini-impurity and SHAP feature importance rankings identifying primary degradation indicators.
6. **Methodology & Limitations**: In-depth explanations of LOBO grouping, right-censoring handling, EOL criteria sensitivity, and operational constraints.

---

## Limitations

1. **Cohort Sample Size:** 34 cells are available for SOH, but only 10 cells reach the 1.4 Ah EOL threshold for RUL prediction after censoring.
2. **Laboratory vs. Fleet Operational Differences:** NASA PCoE data represents accelerated laboratory cycling under constant current/power conditions rather than variable real-world EV duty cycles.
3. **Capacity Jump Sensitivity:** Transient capacity recovery following relaxation periods can confound strict threshold-crossing rules.
