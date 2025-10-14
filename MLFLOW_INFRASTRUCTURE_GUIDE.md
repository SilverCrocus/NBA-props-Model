# MLflow Experiment Tracking Infrastructure
## NBA Props Model - 12 Week Development

---

## Executive Summary

This guide establishes MLflow as the experiment tracking and model management infrastructure for the NBA props model development. MLflow was selected for its simplicity, local-first approach, cost-effectiveness, and excellent integration with XGBoost/LightGBM.

**Key Benefits:**
- Zero cloud costs (local SQLite + file storage)
- Native XGBoost/LightGBM autologging
- Built-in model registry with versioning
- Seamless integration with existing pipeline code
- Reproducibility through environment capture
- Easy comparison of 50-100 experiments

---

## 1. Tool Selection: MLflow

### Why MLflow over Weights & Biases?

| Criteria | MLflow | W&B | Winner |
|----------|--------|-----|--------|
| Cost | Free (local) | Free tier limited | MLflow |
| Setup Time | 5 minutes | Account + API key | MLflow |
| Offline Work | Yes | Limited | MLflow |
| XGBoost Integration | Native autologging | Manual | MLflow |
| Model Registry | Built-in | Yes | Tie |
| Learning Curve | Gentle | Steeper | MLflow |
| Data Privacy | Fully local | Cloud-based | MLflow |

**Recommendation:** Use MLflow for this project. It's simpler, faster to set up, and keeps everything local. Can migrate to W&B later if needed for team collaboration.

---

## 2. Project Organization

### Experiment Hierarchy

```
mlruns/                          # MLflow tracking directory
├── 0/                           # Default experiment
├── 1/                           # Phase 1: Foundation (Weeks 1-3)
│   ├── baseline_xgboost/
│   ├── baseline_lightgbm/
│   └── feature_selection_v1/
├── 2/                           # Phase 2: Advanced (Weeks 4-7)
│   ├── opponent_features/
│   ├── temporal_features/
│   └── ensemble_v1/
├── 3/                           # Phase 3: Calibration (Weeks 8-9)
│   ├── isotonic_calibration/
│   ├── platt_scaling/
│   └── calibrated_ensemble/
└── 4/                           # Phase 4: Production (Weeks 10-12)
    ├── walk_forward_validation/
    ├── production_candidate_v1/
    └── production_candidate_v2/

models/                          # MLflow Model Registry
├── NBAPropsModel/
│   ├── Production/              # Live model
│   ├── Staging/                 # Testing phase
│   └── Archived/                # Deprecated models
```

### Naming Conventions

**Experiment Names:**
```
{phase}_{model_type}_{feature_set}_{description}

Examples:
- foundation_xgboost_baseline_v1
- advanced_lightgbm_opponent_features_v2
- calibration_ensemble_isotonic_final
- production_xgboost_walkforward_2024_25
```

**Run Names:**
```
{date}_{model}_{features}_{key_param}

Examples:
- 20251014_xgb_core_depth6
- 20251015_lgbm_full_lr0.05
- 20251016_ensemble_calibrated_v1
```

**Model Versions:**
```
v{major}.{minor}.{patch}

Examples:
- v1.0.0: Initial baseline
- v1.1.0: Added opponent features
- v1.1.1: Bug fix in feature engineering
- v2.0.0: New calibration approach
```

---

## 3. Metrics to Track

### Training Metrics (Logged Every Epoch/Iteration)

```python
mlflow.log_metrics({
    # Core regression metrics
    "train_mae": mae_train,
    "train_rmse": rmse_train,
    "train_r2": r2_train,
    "train_mape": mape_train,

    # Validation metrics
    "val_mae": mae_val,
    "val_rmse": rmse_val,
    "val_r2": r2_val,
    "val_mape": mape_val,

    # Model complexity
    "n_features": n_features,
    "n_estimators": n_estimators,
})
```

### Betting Performance Metrics (Walk-Forward Validation)

```python
mlflow.log_metrics({
    # Overall performance
    "wf_mae_2023_24": mae_2023,
    "wf_mae_2024_25": mae_2024,
    "wf_win_rate": win_rate,
    "wf_roi": roi,
    "wf_clv": closing_line_value,

    # Calibration metrics
    "brier_score": brier,
    "calibration_error": cal_error,
    "sharpness": sharpness,

    # Risk metrics
    "max_drawdown": max_dd,
    "sharpe_ratio": sharpe,
    "kelly_criterion": kelly,
    "bet_count": n_bets,
})
```

### Segmented Performance (Tags)

```python
# By player tier
mlflow.log_metric("mae_tier1_stars", mae_stars)
mlflow.log_metric("mae_tier2_rotation", mae_rotation)
mlflow.log_metric("mae_tier3_bench", mae_bench)

# By edge size
mlflow.log_metric("roi_edge_small", roi_small)      # 2-4% edge
mlflow.log_metric("roi_edge_medium", roi_medium)    # 4-6% edge
mlflow.log_metric("roi_edge_large", roi_large)      # 6%+ edge

# By season
mlflow.log_metric("mae_2021_22", mae_21_22)
mlflow.log_metric("mae_2022_23", mae_22_23)
mlflow.log_metric("mae_2023_24", mae_23_24)
mlflow.log_metric("mae_2024_25", mae_24_25)

# By context
mlflow.log_metric("mae_home", mae_home)
mlflow.log_metric("mae_away", mae_away)
mlflow.log_metric("mae_b2b", mae_b2b)
```

---

## 4. Artifacts to Store

### Model Artifacts

```python
# 1. Model binary (automatic with autologging)
mlflow.xgboost.log_model(model, "model")

# 2. Feature importance plot
fig, ax = plt.subplots(figsize=(10, 12))
plot_feature_importance(model, ax=ax)
mlflow.log_figure(fig, "feature_importance.png")

# 3. Calibration curve
fig = plot_calibration_curve(y_true, y_pred_proba)
mlflow.log_figure(fig, "calibration_curve.png")

# 4. Prediction CSVs
predictions_df.to_csv("predictions_2024_25.csv", index=False)
mlflow.log_artifact("predictions_2024_25.csv")

# 5. Confusion matrix (for over/under classification)
cm_fig = plot_confusion_matrix(y_true, y_pred)
mlflow.log_figure(cm_fig, "confusion_matrix.png")

# 6. Error analysis report
error_analysis_df.to_csv("error_analysis.csv", index=False)
mlflow.log_artifact("error_analysis.csv")

# 7. Residual plot
residual_fig = plot_residuals(y_true, y_pred)
mlflow.log_figure(residual_fig, "residuals.png")
```

### Configuration Artifacts

```python
# Feature configuration
mlflow.log_dict({
    "feature_set_version": "v2.1",
    "n_features": 47,
    "feature_tiers": {
        "core": ["USG_L15_EWMA", "PSA_L15_EWMA", ...],
        "contextual": ["Minutes_L5_Mean", "opp_def_rating", ...],
        "temporal": ["volatility_5game", "trend_points", ...]
    }
}, "features.json")

# Model hyperparameters (automatic with autologging)
mlflow.log_params({
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
})

# Training configuration
mlflow.log_dict({
    "train_seasons": ["2021-22", "2022-23", "2023-24"],
    "validation_type": "walk_forward",
    "test_season": "2024-25",
    "min_games_played": 10,
    "min_minutes_per_game": 15,
}, "training_config.json")
```

---

## 5. Model Registry Workflow

### Model Lifecycle States

1. **None** → New model just trained
2. **Staging** → Model under evaluation
3. **Production** → Live model for predictions
4. **Archived** → Deprecated/replaced model

### Promotion Criteria

**Staging → Production:**
```python
# A model qualifies for production if:
criteria = {
    "val_mae": < 3.5,                  # Better than baseline
    "wf_roi": > 0.05,                   # 5%+ ROI on walk-forward
    "calibration_error": < 0.05,        # Well-calibrated
    "win_rate": > 0.55,                 # 55%+ win rate
    "min_sample_size": 500,             # Tested on 500+ predictions
    "sharpe_ratio": > 1.0,              # Risk-adjusted returns
}
```

### Model Registration Example

```python
# Register model after successful training
model_uri = f"runs:/{run_id}/model"
model_details = mlflow.register_model(
    model_uri=model_uri,
    name="NBAPropsModel"
)

# Add model version tags
client = mlflow.tracking.MlflowClient()
client.set_model_version_tag(
    name="NBAPropsModel",
    version=model_details.version,
    key="validation_mae",
    value="3.42"
)

# Promote to staging for testing
client.transition_model_version_stage(
    name="NBAPropsModel",
    version=model_details.version,
    stage="Staging"
)

# After testing, promote to production
if meets_production_criteria(model_details):
    client.transition_model_version_stage(
        name="NBAPropsModel",
        version=model_details.version,
        stage="Production"
    )
```

### Rollback Strategy

```python
def rollback_model(reason: str):
    """Rollback to previous production model"""
    client = mlflow.tracking.MlflowClient()

    # Get current production model
    current_prod = client.get_latest_versions(
        "NBAPropsModel",
        stages=["Production"]
    )[0]

    # Archive current
    client.transition_model_version_stage(
        name="NBAPropsModel",
        version=current_prod.version,
        stage="Archived"
    )

    # Add rollback reason
    client.set_model_version_tag(
        name="NBAPropsModel",
        version=current_prod.version,
        key="rollback_reason",
        value=reason
    )

    # Get previous version
    previous = client.get_model_version(
        "NBAPropsModel",
        version=int(current_prod.version) - 1
    )

    # Promote previous to production
    client.transition_model_version_stage(
        name="NBAPropsModel",
        version=previous.version,
        stage="Production"
    )

    logger.info(f"Rolled back from v{current_prod.version} to v{previous.version}")
```

---

## 6. Experiment Comparison Strategy

### Compare Within Phase

```python
# Get all runs from Phase 2 (Advanced Features)
experiment = mlflow.get_experiment_by_name("Phase2_Advanced")
runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="metrics.val_mae < 4.0",
    order_by=["metrics.val_mae ASC"],
    max_results=10
)

# Compare key metrics
comparison_df = runs[[
    'run_id',
    'metrics.val_mae',
    'metrics.wf_roi',
    'metrics.win_rate',
    'params.n_estimators',
    'params.max_depth',
]]
```

### Compare Across Phases

```python
def compare_best_models_by_phase():
    """Compare best model from each phase"""

    phases = {
        "Foundation": "Phase1_Foundation",
        "Advanced": "Phase2_Advanced",
        "Calibration": "Phase3_Calibration",
        "Production": "Phase4_Production",
    }

    results = []
    for phase_name, exp_name in phases.items():
        exp = mlflow.get_experiment_by_name(exp_name)
        best_run = mlflow.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["metrics.val_mae ASC"],
            max_results=1
        ).iloc[0]

        results.append({
            "phase": phase_name,
            "run_id": best_run['run_id'],
            "val_mae": best_run['metrics.val_mae'],
            "wf_roi": best_run['metrics.wf_roi'],
            "model_type": best_run['tags.model_type'],
        })

    return pd.DataFrame(results)
```

### Weekly Progress Dashboard

```python
def generate_weekly_report(week_number: int):
    """Generate weekly progress report"""

    # Get runs from this week
    week_start = datetime.now() - timedelta(days=7)
    runs_this_week = mlflow.search_runs(
        filter_string=f"attributes.start_time >= {week_start.timestamp()}",
        order_by=["metrics.val_mae ASC"]
    )

    report = {
        "week": week_number,
        "experiments_run": len(runs_this_week),
        "best_mae": runs_this_week['metrics.val_mae'].min(),
        "best_roi": runs_this_week['metrics.wf_roi'].max(),
        "models_registered": count_registered_models_this_week(),
        "avg_training_time": runs_this_week['metrics.training_time'].mean(),
    }

    return report
```

---

## 7. Setup Instructions

### Step 1: Install MLflow

```bash
# Add MLflow to project dependencies
uv add mlflow matplotlib scikit-learn

# Verify installation
uv run python -c "import mlflow; print(mlflow.__version__)"
```

### Step 2: Initialize MLflow Tracking

```bash
# Create MLflow directory structure
mkdir -p /Users/diyagamah/Documents/nba_props_model/mlruns
mkdir -p /Users/diyagamah/Documents/nba_props_model/mlflow_artifacts

# Set environment variables (add to .env file)
cat >> .env << EOF
MLFLOW_TRACKING_URI=file:///Users/diyagamah/Documents/nba_props_model/mlruns
MLFLOW_ARTIFACT_LOCATION=file:///Users/diyagamah/Documents/nba_props_model/mlflow_artifacts
EOF
```

### Step 3: Create Experiment Structure

```python
# Run this once to set up experiments
import mlflow

mlflow.set_tracking_uri("file:///Users/diyagamah/Documents/nba_props_model/mlruns")

# Create phase experiments
phases = [
    ("Phase1_Foundation", "Weeks 1-3: Baseline models and feature selection"),
    ("Phase2_Advanced", "Weeks 4-7: Advanced features and ensemble methods"),
    ("Phase3_Calibration", "Weeks 8-9: Model calibration and probability adjustment"),
    ("Phase4_Production", "Weeks 10-12: Walk-forward validation and production deployment"),
]

for name, description in phases:
    try:
        mlflow.create_experiment(
            name=name,
            artifact_location=f"mlflow_artifacts/{name}",
            tags={"project": "nba_props", "phase": name}
        )
        print(f"Created experiment: {name}")
    except Exception as e:
        print(f"Experiment {name} already exists")
```

### Step 4: Start MLflow UI

```bash
# Launch MLflow UI (runs on localhost:5000)
uv run mlflow ui --backend-store-uri file:///Users/diyagamah/Documents/nba_props_model/mlruns

# Or with custom port
uv run mlflow ui --port 5001
```

### Step 5: Test Integration

```bash
# Run the test script (created below)
uv run python src/mlflow_integration/test_tracking.py
```

---

## 8. Code Implementation

See the following files for complete implementation:

1. **src/mlflow_integration/tracker.py** - Core tracking wrapper
2. **src/mlflow_integration/registry.py** - Model registry manager
3. **src/mlflow_integration/utils.py** - Helper functions
4. **src/models/train_model_mlflow.py** - Updated training script with MLflow
5. **scripts/compare_experiments.py** - Experiment comparison utilities

---

## 9. Best Practices

### DO:
- Log experiments immediately (don't batch)
- Use descriptive run names and tags
- Store all artifacts (plots, CSVs, configs)
- Register models only after validation
- Add version tags for traceability
- Use nested runs for ensemble models
- Enable autologging for XGBoost/LightGBM
- Document rollback reasons

### DON'T:
- Log large datasets as artifacts (use references)
- Skip parameter logging (hurts reproducibility)
- Delete experiments (archive instead)
- Push models to production without staging
- Forget to log feature versions
- Skip calibration metrics for betting models
- Use default experiment (always create named experiments)

---

## 10. Cost Analysis

**MLflow (Local):**
- Storage: ~10GB for 100 experiments (free on local disk)
- Compute: Uses existing hardware (no cloud costs)
- Total: $0/month

**Weights & Biabe (Cloud):**
- Free tier: 100GB storage, limited projects
- Paid tier: $50+/month for team features
- Total: $0-50/month

**Winner:** MLflow for this project (can migrate later if needed)

---

## 11. Migration Path (Future)

If the team grows or you need collaboration:

```python
# Option 1: Migrate to MLflow on AWS/Azure
mlflow.set_tracking_uri("databricks://...")

# Option 2: Migrate to Weights & Biases
# Export MLflow runs and import to W&B

# Option 3: Keep MLflow, add remote storage
mlflow.set_tracking_uri("postgresql://...")
```

---

## Next Steps

1. Run setup instructions (Steps 1-5)
2. Review implementation files (Section 8)
3. Run first baseline experiment
4. Review results in MLflow UI
5. Establish weekly comparison workflow
6. Set up model registry promotion criteria

---

## Questions?

Contact: Hivin Diyagama (hivin.diyagama@tabcorp.com.au)
Project: NBA Props Model - 12 Week Development
Last Updated: 2025-10-14
