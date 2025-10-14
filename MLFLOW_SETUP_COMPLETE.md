# MLflow Infrastructure Setup Complete

## Summary

Your MLflow experiment tracking infrastructure is now fully integrated into the NBA props model project. Everything is ready for the 12-week development cycle.

---

## What Was Installed

### Dependencies Added
- **mlflow** (3.4.0): Core experiment tracking and model registry
- **matplotlib** (3.10.7): Plotting and visualization
- **scikit-learn** (1.7.2): Metrics and utilities
- **scipy** (1.16.1): Statistical functions
- **seaborn** (0.13.2): Advanced visualizations

### Code Modules Created

**Core MLflow Integration** (`src/mlflow_integration/`):
- `tracker.py` - Experiment tracking wrapper with 15+ logging methods
- `registry.py` - Model registry manager with lifecycle management
- `utils.py` - Comparison, reporting, and analysis utilities
- `__init__.py` - Package initialization

**Updated Training Pipeline** (`src/models/`):
- `train_model_mlflow.py` - Full MLflow integration for training

**Utility Scripts** (`scripts/mlflow/`):
- `setup_experiments.py` - One-time experiment structure setup
- `compare_experiments.py` - CLI tool for experiment comparison
- `test_tracking.py` - Integration test suite

### Documentation Created
- **MLFLOW_INFRASTRUCTURE_GUIDE.md** - Complete 11-section guide (7,000+ words)
- **MLFLOW_QUICKSTART.md** - 10-minute quick start guide
- **MLFLOW_SETUP_COMPLETE.md** - This file

---

## File Locations

```
/Users/diyagamah/Documents/nba_props_model/
├── src/
│   ├── mlflow_integration/
│   │   ├── __init__.py
│   │   ├── tracker.py              # Core tracking functionality
│   │   ├── registry.py             # Model registry manager
│   │   └── utils.py                # Utilities and helpers
│   └── models/
│       └── train_model_mlflow.py   # MLflow-integrated trainer
│
├── scripts/
│   └── mlflow/
│       ├── setup_experiments.py    # Setup experiments
│       ├── compare_experiments.py  # Compare runs
│       └── test_tracking.py        # Test integration
│
├── MLFLOW_INFRASTRUCTURE_GUIDE.md  # Complete guide
├── MLFLOW_QUICKSTART.md            # Quick start
└── MLFLOW_SETUP_COMPLETE.md        # This file
```

---

## Next Steps (5 Minutes to Get Started)

### 1. Set Up Experiments (1 min)
```bash
uv run python scripts/mlflow/setup_experiments.py
```

### 2. Test Integration (2 min)
```bash
uv run python scripts/mlflow/test_tracking.py
```

### 3. Start MLflow UI (30 sec)
```bash
uv run mlflow ui
```
Open: http://localhost:5000

### 4. Train Your First Model (2 min)
```python
from src.models.train_model_mlflow import NBAPropsMLflowTrainer
import pandas as pd

# Initialize
trainer = NBAPropsMLflowTrainer(experiment_name="Phase1_Foundation")

# Load data (replace with your actual data)
df = pd.read_csv('data/processed/your_data.csv')
df_prepared = trainer.prepare_training_data(df)

# Train with full tracking
results = trainer.train_model(
    df=df_prepared,
    model_type='xgboost',
    run_name='baseline_v1',
    register_model=True
)

print(f"Run ID: {results['run_id']}")
```

---

## Key Features Implemented

### Experiment Tracking
✓ 4-phase experiment structure (Foundation, Advanced, Calibration, Production)
✓ Automatic parameter logging
✓ Training/validation metrics tracking
✓ Betting-specific metrics (ROI, win rate, CLV, Sharpe ratio)
✓ Segmented performance tracking (by player tier, edge size, season)
✓ XGBoost/LightGBM autologging support

### Artifact Management
✓ Feature importance plots
✓ Calibration curves
✓ Residual analysis plots
✓ Confusion matrices
✓ Prediction CSVs
✓ Error analysis reports
✓ Model configuration JSON

### Model Registry
✓ Version control and lifecycle management
✓ Staging/Production promotion workflow
✓ Automated production criteria evaluation
✓ Rollback functionality with reason tracking
✓ Model comparison utilities
✓ Tag-based metadata system

### Utilities
✓ Experiment comparison across phases
✓ Best model identification by any metric
✓ Weekly progress summaries
✓ Automated report generation
✓ CSV export for external analysis
✓ CLI tool for common operations

---

## Experiment Organization

```
Phase 1: Foundation (Weeks 1-3)
├── Baseline XGBoost
├── Baseline LightGBM
├── Feature selection experiments
└── Initial validation

Phase 2: Advanced (Weeks 4-7)
├── Opponent features
├── Temporal features
├── Ensemble methods
└── Advanced feature engineering

Phase 3: Calibration (Weeks 8-9)
├── Isotonic calibration
├── Platt scaling
└── Betting optimization

Phase 4: Production (Weeks 10-12)
├── Walk-forward validation
├── Production candidates
└── Final deployment model
```

---

## Tracked Metrics

### Training Metrics
- train_mae, train_rmse, train_r2, train_mape
- val_mae, val_rmse, val_r2, val_mape
- n_features, n_estimators, training_time

### Betting Metrics
- roi (Return on Investment)
- win_rate (Percentage of winning bets)
- clv (Closing Line Value)
- sharpe_ratio (Risk-adjusted returns)
- max_drawdown (Maximum loss from peak)
- brier_score (Probability calibration)
- calibration_error (Calibration accuracy)
- kelly_criterion (Optimal bet sizing)

### Segmented Metrics
- By player tier (stars, rotation, bench)
- By edge size (small, medium, large)
- By season (2021-22, 2022-23, 2023-24, 2024-25)
- By context (home, away, back-to-back)

---

## Production Criteria

Models are promoted to production if they meet:
- val_mae < 3.5
- betting_roi > 5%
- betting_win_rate > 55%
- calibration_error < 5%
- sharpe_ratio > 1.0
- Tested on 500+ predictions

These criteria are defined in `src/mlflow_integration/registry.py` and can be customized.

---

## Common Commands Reference

### Setup
```bash
# Set up experiments
uv run python scripts/mlflow/setup_experiments.py

# Test integration
uv run python scripts/mlflow/test_tracking.py

# Start UI
uv run mlflow ui
```

### Comparison
```bash
# Best models by MAE
uv run python scripts/mlflow/compare_experiments.py best Phase1_Foundation --metric val_mae

# Best models by ROI
uv run python scripts/mlflow/compare_experiments.py best Phase2_Advanced --metric betting_roi

# Generate report
uv run python scripts/mlflow/compare_experiments.py report Phase1_Foundation --output report.txt

# Compare phases
uv run python scripts/mlflow/compare_experiments.py phases

# Export to CSV
uv run python scripts/mlflow/compare_experiments.py export Phase1_Foundation runs.csv
```

### Training
```python
# Basic training with tracking
from src.models.train_model_mlflow import NBAPropsMLflowTrainer

trainer = NBAPropsMLflowTrainer(experiment_name="Phase1_Foundation")
results = trainer.train_model(df, model_type='xgboost', run_name='test_v1')

# Walk-forward validation with betting metrics
results = trainer.train_and_evaluate_walk_forward(
    df=df,
    model_type='xgboost',
    run_name='walkforward_v1',
    register_model=True
)
```

---

## MLflow UI Guide

### View Experiments
1. Navigate to http://localhost:5000
2. Click on experiment name (e.g., "Phase1_Foundation")
3. See all runs for that experiment

### Compare Runs
1. Select runs with checkboxes
2. Click "Compare" button
3. View side-by-side comparison

### View Run Details
1. Click on run name
2. Tabs:
   - **Parameters**: Hyperparameters
   - **Metrics**: Training/validation metrics
   - **Artifacts**: Plots, models, CSVs
   - **Tags**: Metadata

### Filter Runs
```
# By metric
metrics.val_mae < 3.5

# By parameter
params.max_depth = "6"

# By tag
tags.model_type = "xgboost"

# Combined
metrics.val_mae < 3.5 AND params.learning_rate = "0.05"
```

---

## Integration with Existing Code

Your existing training scripts can be easily updated:

### Before (without MLflow)
```python
model = xgb.XGBRegressor(n_estimators=200, max_depth=6)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"MAE: {mae}")
```

### After (with MLflow)
```python
from src.mlflow_integration.tracker import NBAPropsTracker, enable_autologging

tracker = NBAPropsTracker(experiment_name="Phase1_Foundation")
enable_autologging('xgboost')

tracker.start_run(run_name="baseline_v1")

model = xgb.XGBRegressor(n_estimators=200, max_depth=6)
model.fit(X_train, y_train)  # Autologged!

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

tracker.log_validation_metrics({'mae': mae})
tracker.log_model(model, model_type='xgboost')
tracker.end_run()
```

---

## Best Practices Checklist

✓ Use descriptive run names (e.g., "xgb_lr0.05_depth6_v1")
✓ Add meaningful tags (model_version, feature_set, description)
✓ Log feature configurations for reproducibility
✓ Track betting metrics alongside model metrics
✓ Store all plots and analysis as artifacts
✓ Register models only after validation
✓ Use production criteria before promotion
✓ Document rollback reasons
✓ Generate weekly comparison reports
✓ Export results for deeper analysis

---

## Troubleshooting

### Can't find MLflow UI
- Check it's running: `lsof -i :5000`
- Try different port: `uv run mlflow ui --port 5001`

### Models not logging
- Enable debug logging: `logging.basicConfig(level=logging.DEBUG)`
- Check autologging: `mlflow.xgboost.autolog()`

### Experiments not showing
- Verify tracking URI: `import mlflow; print(mlflow.get_tracking_uri())`
- Should be: `file:///Users/diyagamah/Documents/nba_props_model/mlruns`

---

## Storage Estimates

For 100 experiments:
- Tracking data (SQLite): ~500 MB
- Artifacts (plots, models, CSVs): ~10 GB
- Total: ~10.5 GB

All stored locally at:
- `/Users/diyagamah/Documents/nba_props_model/mlruns/`
- `/Users/diyagamah/Documents/nba_props_model/mlflow_artifacts/`

---

## Migration Path (Future)

If you need team collaboration later:

**Option 1: Cloud MLflow**
- Deploy to AWS/Azure/GCP
- ~$50-100/month for infrastructure

**Option 2: Databricks**
- Integrated MLflow hosting
- ~$0.10/DBU + compute costs

**Option 3: Weights & Biases**
- Export MLflow runs and import to W&B
- Free tier or $50+/month

Current setup supports easy migration to any of these options.

---

## Documentation

- **Complete Guide**: MLFLOW_INFRASTRUCTURE_GUIDE.md (7,000+ words)
- **Quick Start**: MLFLOW_QUICKSTART.md (10-minute guide)
- **MLflow Docs**: https://mlflow.org/docs/latest/
- **Code Examples**:
  - src/models/train_model_mlflow.py
  - scripts/mlflow/test_tracking.py

---

## Support & Questions

**Code References:**
- Tracker API: `src/mlflow_integration/tracker.py`
- Registry API: `src/mlflow_integration/registry.py`
- Utilities: `src/mlflow_integration/utils.py`

**Test Suite:**
```bash
uv run python scripts/mlflow/test_tracking.py
```

**Documentation:**
- MLFLOW_INFRASTRUCTURE_GUIDE.md - Complete reference
- MLFLOW_QUICKSTART.md - Quick start guide

---

## Success Criteria

By the end of 12 weeks, you should have:
- ✓ 50-100 tracked experiments
- ✓ 10-20 registered model versions
- ✓ 2-3 production-ready models
- ✓ Complete model lineage and reproducibility
- ✓ Automated comparison and reporting workflow
- ✓ Data-driven model selection process

---

## Ready to Start!

Your experiment tracking infrastructure is production-ready. Start with:

1. Run setup: `uv run python scripts/mlflow/setup_experiments.py`
2. Run tests: `uv run python scripts/mlflow/test_tracking.py`
3. Start UI: `uv run mlflow ui`
4. Train first model using `NBAPropsMLflowTrainer`
5. Review results in MLflow UI
6. Iterate and improve!

**Good luck with your 12-week NBA props model development!**

---

Project: NBA Props Model
Contact: Hivin Diyagama (hivin.diyagama@tabcorp.com.au)
Setup Date: 2025-10-14
MLflow Version: 3.4.0
