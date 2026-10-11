# EV Battery SOH and RUL Prognostics

An auditable machine-learning mini-project for estimating lithium-ion battery
State of Health (SOH) and Remaining Useful Life (RUL) from NASA Ames PCoE
laboratory aging data. It parses the original MATLAB files, derives causal
cycle-level features and targets, evaluates models with battery-level holdouts,
and presents measured data and saved evaluation results in Streamlit.

NASA describes this experiment as a laboratory testbed with differing discharge
conditions and an EOL criterion of 30% rated-capacity fade (2 Ah to 1.4 Ah).
These data do not establish EV fleet or pack performance. See the [official
NASA repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)
and its [linked archive](https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip).
Dataset citation: B. Saha and K. Goebel (2007), “Battery Data Set”, NASA
Prognostics Data Repository, NASA Ames Research Center.

## Project layout

```text
BL_RUL_predictor/
├── app/
│   ├── streamlit_app.py          # Dashboard entry point
│   └── components/               # Cached data, upload validation, Plotly charts
├── data/
│   ├── raw/                      # Place official .mat files here (Git-ignored)
│   └── processed/                # Generated feature tables (Git-ignored)
├── models/                       # Final demonstration fits and schemas (ignored)
├── reports/
│   ├── figures/                  # Exploratory plots (ignored)
│   └── results/                  # Metrics, OOF predictions, audits (ignored)
├── scripts/                      # Inspection, preprocessing, training and audits
├── src/ev_battery/               # Loader, features, targets, models, evaluation
├── tests/                        # unittest suite
├── requirements.txt              # Core dependencies
└── requirements-optional.txt     # XGBoost, PyTorch and SHAP
```

Raw `.mat` files and generated data, model weights, figures and result tables
are excluded by `.gitignore`. The project is inside a parent Git working tree;
the project `.gitignore` excludes `data/raw/**` and `*.mat`. Confirm with:

```bash
git check-ignore -v BL_RUL_predictor/data/raw/your-file.mat
git ls-files 'BL_RUL_predictor/data/raw/**/*.mat'
```

## Environment and reproducible commands

Tested here with Python 3.13.5, NumPy 2.2.6, SciPy 1.16.0, pandas 2.2.3,
scikit-learn 1.6.1, Streamlit 1.45.1 and Plotly 6.1.2. Optional components
XGBoost 3.0.2, PyTorch 2.7.1 and SHAP 0.48.0 were installed for the run below.

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pip install -r requirements-optional.txt  # Optional experiments

# Copy the official archive's .mat files (preserve file names) into data/raw/.
python scripts/inspect_dataset.py --output reports/inspection.json
python scripts/prepare_data.py
python -m unittest discover -s tests -v
python scripts/run_experiments.py
python scripts/audit_experiments.py
python scripts/check_numerics.py
streamlit run app/streamlit_app.py
```

The parser supports MATLAB files readable by SciPy; MATLAB v7.3/HDF5 files are
reported as unsupported. No network download is assumed. Optional dependencies
are skipped with a recorded reason if unavailable. `--raw-dir`, `--output-dir`,
`--reports-dir`, and `--figures-dir` are available on preprocessing; the
experiment script accepts `--data-dir`, `--results-dir`, `--models-dir`, and
`--seed`.

## Targets, features and evaluation

SOH is `100 × measured discharge capacity (Ah) / reference capacity (Ah)` and
is reported in percentage points. The default reference is 2.0 Ah; the optional
`first_valid` reference uses each battery's first valid recorded capacity. RUL
is `EOL discharge-cycle index − current discharge-cycle index`, in cycles. The
default EOL is the first capacity at or below 70% of the reference (1.4 Ah for
the rated-capacity setting). `--persistence K` requires K consecutive qualifying
records; the default is one. A battery with no observed crossing is censored,
not assigned an invented EOL. A first recorded value already below threshold is
flagged because its earlier crossing is unknown.

Each row represents a recorded discharge operation, in source record order.
SOH predictors are cycle index, ambient temperature and charge/discharge
voltage, current and temperature summaries. RUL adds current observed discharge
capacity and causal lag/trailing capacity history. Capacity and derived SOH are
excluded from SOH predictors. Battery ID is retained only for grouping and
diagnostics. No future cycles, EOL index or target enters predictors. Missing
predictors stay missing until imputation is learned inside each training fold.
Impedance remains separate because a defensible cycle alignment is not
established.

Models are a training-mean baseline, Ridge, Random Forest, SVR, XGBoost and a
small LSTM. Scaling/imputation are in fold-local pipelines. Primary validation
is leave-one-battery-out (LOBO): all rows from one battery are held out together;
hyperparameter searches use group-wise splits only among the remaining training
batteries. Metrics include MAE, RMSE and R² (undefined R² remains blank), with
both macro battery-equal and pooled row-weighted summaries. SOH units are
percentage points; RUL units are cycles. LSTM uses contiguous five-cycle causal
windows within each battery, so it has lower prediction coverage.

## Generated artifacts

Preprocessing generates:

- `data/processed/cycle_features.csv`, `soh_dataset.csv`, `rul_dataset.csv`
- `data/processed/operation_inventory.csv`, `impedance_records.csv`
- `reports/results/data_quality_report.json`
- `reports/figures/capacity_degradation.png`, `soh_trajectory.png`,
  `feature_relationships.png`

Training generates `soh_metrics.csv`, `rul_metrics.csv`, `fold_metrics.csv`,
`oof_predictions.csv`, `common_support_metrics.csv`, `feature_importance.csv`,
`model_comparison.json`, `experiment_audit.json`, and final inference models
with feature-schema metadata under `models/`. Every OOF prediction records its
battery, cycle, actual and predicted target, model and fold. Final all-data fits
are for demonstration only; their in-sample scores are not reported as
cross-validation performance.

## Quality assurance run and observed results

Executed on the actual files in `data/raw/` on 2026-10-11 with the versions above:

- 38 MATLAB files discovered; byte-identical duplicate files were reconciled by
  content hash. The resulting processed set contains 34 battery IDs, 2,794
  discharge rows and 930 rows in the RUL table.
- All 30 unit tests passed.
- All six dashboard sections rendered in Streamlit's `AppTest` with no
  exceptions. Upload validation rejected missing columns and nonnumeric feature
  values. Live server endpoint and browser interaction should also be checked
  in the local runtime.
- Full LOBO experiment completed: 263 evaluated folds and 19,268 OOF
  predictions. The audit passed: actual values match prepared labels, expected
  prediction coverage is exact, outer test batteries are absent from tuning,
  metrics recompute from saved predictions, and all 12 final model artifacts
  reload and produce finite predictions.
- Both XGBoost and LSTM were evaluated. LSTM has reduced SOH coverage (33 groups,
  2,546 rows versus 34 groups, 2,750 for tabular models); RUL LSTM covers 462
  versus 502 rows.

Observed macro battery-equal MAE from this run:

| Target | Best macro MAE | Next models (macro MAE) |
| --- | ---: | --- |
| SOH | SVR: 4.27 percentage points | Random Forest 5.32; LSTM 5.65; XGBoost 5.66; Ridge 7.20 |
| RUL | SVR: 11.52 cycles | XGBoost 13.72; Random Forest 13.77; LSTM 15.94; Ridge 25.23 |

The mean baseline scored 19.45 percentage points MAE for SOH and 31.30 cycles
MAE for RUL. These are small-cohort results, not evidence of stable fleet
generalization. RUL uses only 10 eligible batteries and is especially sensitive
to EOL labels: with the default single-crossing rule, changes in discharge
current/temperature and isolated capacity dips may be interpreted as EOL.
Review the per-battery traces and consider reference-condition measurements
before treating RUL results as operational forecasts.

Numerical QA passed: Ridge OOF predictions matched independent SVD refits to
within 5.5e-11 (SOH) and 3.8e-12 (RUL), and replaying the checked LSTM fold
matched to 1.5e-14. The environment nevertheless emitted thousands of NumPy
matrix-operation runtime warnings during Ridge fitting; this remains a numeric
runtime caveat, even though the saved predictions were finite and independently
reproduced.

## Limitations

There are 34 independent battery groups for SOH but only 10 observed-EOL RUL
groups after censoring and initial-threshold-only exclusions. LOBO fold results
share most training batteries, so fold variation is descriptive, not a
confidence interval. Tests are accelerated laboratory cell tests with varying
conditions, not EV fleet or pack field data. EOL threshold labels from
operational discharge capacity can be confounded by test conditions. The
dashboard reports saved results only and performs no experiment on page load.
