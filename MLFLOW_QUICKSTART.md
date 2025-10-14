# MLflow Quick Start Guide
## Get Started in 10 Minutes

---

## Step 1: Install MLflow (2 minutes)

```bash
cd /Users/diyagamah/Documents/nba_props_model

# Add MLflow and dependencies
uv add mlflow matplotlib scikit-learn scipy seaborn

# Verify installation
uv run python -c "import mlflow; print(f'MLflow version: {mlflow.__version__}')"
```

---

## Step 2: Set Up Experiments (1 minute)

```bash
# Run setup script
uv run python scripts/mlflow/setup_experiments.py
```

This creates 4 experiments:
- Phase1_Foundation (Weeks 1-3)
- Phase2_Advanced (Weeks 4-7)
- Phase3_Calibration (Weeks 8-9)
- Phase4_Production (Weeks 10-12)

---

## Step 3: Test the Integration (2 minutes)

```bash
# Run test suite
uv run python scripts/mlflow/test_tracking.py
```

This will:
- Create synthetic test data
- Train 3 test models
- Log parameters, metrics, and artifacts
- Test model registry
- Verify everything works

---

## Step 4: Start MLflow UI (1 minute)

```bash
# Start the MLflow UI (keeps running in background)
uv run mlflow ui --backend-store-uri file:///Users/diyagamah/Documents/nba_props_model/mlruns

# Or use shorter version (defaults to ./mlruns)
uv run mlflow ui
```

Open browser: http://localhost:5000

You should see:
- 4 experiments listed
- Test runs from Step 3
- Metrics, parameters, and artifacts

---

## Step 5: Train Your First Real Model (5 minutes)

### Option A: Using the MLflow-integrated trainer

```python
from src.models.train_model_mlflow import NBAPropsMLflowTrainer
import pandas as pd

# Initialize trainer
trainer = NBAPropsMLflowTrainer(
    experiment_name="Phase1_Foundation"
)

# Load your data
df = pd.read_csv('data/processed/player_stats_2023_24.csv')

# Prepare data
df_prepared = trainer.prepare_training_data(df)

# Train model with MLflow tracking
results = trainer.train_model(
    df=df_prepared,
    model_type='xgboost',
    run_name='baseline_xgb_v1',
    hyperparams={
        'n_estimators': 200,
        'max_depth': 6,
        'learning_rate': 0.05,
    },
    register_model=True,
    tags={
        'model_version': 'v1.0.0',
        'description': 'First baseline model',
        'feature_set': 'basic_rolling'
    }
)

print(f"Model trained! Run ID: {results['run_id']}")
print(f"Validation MAE: {results['final_metrics']['val_mae']:.4f}")
```

### Option B: Manual integration (for existing scripts)

```python
from src.mlflow_integration.tracker import NBAPropsTracker, enable_autologging
from xgboost import XGBRegressor
import pandas as pd
import numpy as np

# Your existing code...
X_train, X_test, y_train, y_test = ...  # Load your data

# Initialize MLflow tracker
tracker = NBAPropsTracker(experiment_name="Phase1_Foundation")

# Enable autologging
enable_autologging('xgboost')

# Start run
tracker.start_run(
    run_name="baseline_xgb_v1",
    tags={"model_type": "xgboost", "version": "v1.0"}
)

# Log parameters
params = {'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.05}
tracker.log_params(params)

# Train model (autologging captures everything)
model = XGBRegressor(**params)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Log validation metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
val_metrics = {
    'mae': mean_absolute_error(y_test, y_pred),
    'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
    'r2': r2_score(y_test, y_pred)
}
tracker.log_validation_metrics(val_metrics)

# Log artifacts
tracker.log_feature_importance(feature_names, model.feature_importances_)
tracker.log_residuals_plot(y_test, y_pred)

# Log model
tracker.log_model(model, model_type='xgboost', registered_model_name='NBAPropsModel')

# End run
tracker.end_run()
```

---

## Common Commands

### Compare Experiments

```bash
# Show top 5 models by MAE
uv run python scripts/mlflow/compare_experiments.py best Phase1_Foundation --metric val_mae --top-n 5

# Show top models by ROI
uv run python scripts/mlflow/compare_experiments.py best Phase2_Advanced --metric betting_roi --top-n 3

# Generate experiment report
uv run python scripts/mlflow/compare_experiments.py report Phase1_Foundation --output reports/phase1_report.txt

# Compare all phases
uv run python scripts/mlflow/compare_experiments.py phases

# Export experiment to CSV
uv run python scripts/mlflow/compare_experiments.py export Phase1_Foundation experiments.csv
```

### Model Registry

```python
from src.mlflow_integration.registry import ModelRegistry, DEFAULT_PRODUCTION_CRITERIA

registry = ModelRegistry()

# Register model from run
model_version = registry.register_model(
    run_id="abc123...",
    model_name="NBAPropsModel",
    tags={"validation_mae": "3.42", "win_rate": "0.58"}
)

# Promote to staging
registry.promote_model(
    model_name="NBAPropsModel",
    version=1,
    stage="Staging"
)

# Evaluate for production
meets_criteria = registry.evaluate_for_production(
    model_name="NBAPropsModel",
    version=1,
    criteria=DEFAULT_PRODUCTION_CRITERIA
)

if meets_criteria:
    # Promote to production
    registry.promote_model(
        model_name="NBAPropsModel",
        version=1,
        stage="Production"
    )

# Rollback if needed
registry.rollback_model(
    model_name="NBAPropsModel",
    reason="Model performance degraded on new data"
)
```

---

## Typical Weekly Workflow

### Monday: Plan week's experiments

```bash
# Review last week's results
uv run python scripts/mlflow/compare_experiments.py phases

# Check best models from last week
uv run python scripts/mlflow/compare_experiments.py best Phase1_Foundation --top-n 10
```

### Tuesday-Thursday: Run experiments

```python
# Run multiple experiments with different configurations
for lr in [0.01, 0.05, 0.1]:
    for depth in [4, 6, 8]:
        results = trainer.train_model(
            df=df_prepared,
            model_type='xgboost',
            run_name=f'xgb_lr{lr}_depth{depth}',
            hyperparams={
                'learning_rate': lr,
                'max_depth': depth,
                'n_estimators': 200,
            },
            tags={
                'experiment_type': 'hyperparameter_search',
                'week': '1'
            }
        )
```

### Friday: Review and document

```bash
# Generate weekly report
uv run python scripts/mlflow/compare_experiments.py report Phase1_Foundation --output reports/week1_report.txt

# Export results for analysis
uv run python scripts/mlflow/compare_experiments.py export Phase1_Foundation data/results/week1_experiments.csv

# Identify best model
uv run python scripts/mlflow/compare_experiments.py best Phase1_Foundation --metric val_mae
```

---

## MLflow UI Tips

### Navigate to specific run:
1. Click experiment name (e.g., "Phase1_Foundation")
2. Click run name to see details
3. Tabs:
   - **Parameters**: All hyperparameters
   - **Metrics**: Training/validation metrics
   - **Artifacts**: Plots, CSVs, models
   - **Tags**: Metadata about the run

### Compare runs:
1. Select multiple runs (checkboxes)
2. Click "Compare" button
3. See side-by-side comparison of metrics/params

### Search/filter runs:
```
# Filter by metric
metrics.val_mae < 3.5

# Filter by parameter
params.max_depth = "6"

# Filter by tag
tags.model_type = "xgboost"

# Combine filters
metrics.val_mae < 3.5 AND params.learning_rate = "0.05"
```

---

## Troubleshooting

### MLflow UI not starting
```bash
# Check if port 5000 is in use
lsof -i :5000

# Use different port
uv run mlflow ui --port 5001
```

### Can't find experiments
```bash
# Verify tracking URI
uv run python -c "import mlflow; print(mlflow.get_tracking_uri())"

# Should print: file:///Users/diyagamah/Documents/nba_props_model/mlruns

# If wrong, set environment variable
export MLFLOW_TRACKING_URI=file:///Users/diyagamah/Documents/nba_props_model/mlruns
```

### Model not logging
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check if autologging is enabled
import mlflow.xgboost
mlflow.xgboost.autolog()  # Enable explicitly
```

---

## Next Steps

1. **Week 1**: Train baseline XGBoost and LightGBM models
2. **Week 2**: Experiment with feature selection
3. **Week 3**: Compare all Phase 1 models and select best
4. **Week 4**: Move to Phase2_Advanced experiment

---

## Additional Resources

- **Full Guide**: MLFLOW_INFRASTRUCTURE_GUIDE.md
- **MLflow Docs**: https://mlflow.org/docs/latest/index.html
- **Tracker API**: src/mlflow_integration/tracker.py
- **Registry API**: src/mlflow_integration/registry.py
- **Utils**: src/mlflow_integration/utils.py

---

## Support

Questions? Check the main guide or review the code examples in:
- `src/models/train_model_mlflow.py`
- `scripts/mlflow/test_tracking.py`

---

Last Updated: 2025-10-14
