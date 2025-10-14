# MLflow Implementation Checklist
## NBA Props Model - 12 Week Development

---

## Pre-Implementation Checklist

- [x] MLflow installed (v3.4.0)
- [x] Dependencies added (matplotlib, scikit-learn, scipy, seaborn)
- [x] Code modules created (tracker, registry, utils)
- [x] Training script updated with MLflow integration
- [x] Utility scripts created (setup, compare, test)
- [x] Documentation written (3 guides + 1 architecture diagram)

---

## Day 1: Initial Setup (30 minutes)

### Step 1: Set Up Experiments (5 min)
```bash
cd /Users/diyagamah/Documents/nba_props_model
uv run python scripts/mlflow/setup_experiments.py
```

**Expected Output:**
```
Created experiment: Phase1_Foundation
Created experiment: Phase2_Advanced
Created experiment: Phase3_Calibration
Created experiment: Phase4_Production
```

- [ ] All 4 experiments created successfully
- [ ] No errors during setup

---

### Step 2: Test Integration (10 min)
```bash
uv run python scripts/mlflow/test_tracking.py
```

**Expected Output:**
```
TEST 1: Basic Tracking
✓ Test passed! Run ID: abc123...

TEST 2: Autologging
✓ Test passed! Run ID: def456...

TEST 3: Model Registry
✓ Model registered successfully!

TEST 4: Experiment Comparison
✓ Found best models: ...

ALL TESTS COMPLETED!
```

- [ ] All 4 tests passed
- [ ] Test runs visible in MLflow
- [ ] No errors in test output

---

### Step 3: Start MLflow UI (2 min)
```bash
uv run mlflow ui
```

**Verification:**
- [ ] UI starts without errors
- [ ] Can access http://localhost:5000
- [ ] See 4 experiments listed
- [ ] See test runs in Phase1_Foundation
- [ ] Can view run details (parameters, metrics, artifacts)

---

### Step 4: Train First Real Model (15 min)

**Option A: Use your existing data**
```python
from src.models.train_model_mlflow import NBAPropsMLflowTrainer
import pandas as pd

# Load your data
df = pd.read_csv('data/processed/player_stats_2023_24.csv')

# Initialize trainer
trainer = NBAPropsMLflowTrainer(experiment_name="Phase1_Foundation")

# Prepare data
df_prepared = trainer.prepare_training_data(df)

# Train baseline model
results = trainer.train_model(
    df=df_prepared,
    model_type='xgboost',
    run_name='baseline_xgb_week1',
    hyperparams={
        'n_estimators': 200,
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
    },
    register_model=True,
    tags={
        'week': '1',
        'model_version': 'v1.0.0',
        'description': 'First baseline model',
        'feature_set': 'basic_rolling'
    }
)

print(f"✓ Training complete!")
print(f"Run ID: {results['run_id']}")
print(f"Validation MAE: {results['final_metrics']['val_mae']:.4f}")
```

**Verification:**
- [ ] Model trains without errors
- [ ] Run appears in MLflow UI
- [ ] Parameters logged correctly
- [ ] Metrics logged (train_mae, val_mae, etc.)
- [ ] Artifacts saved (feature_importance.png, etc.)
- [ ] Model registered in Model Registry

---

## Week 1: Foundation Phase

### Monday: Setup & Baseline Models

**Tasks:**
- [ ] Train XGBoost baseline
- [ ] Train LightGBM baseline
- [ ] Compare both models
- [ ] Document baseline performance

**Commands:**
```bash
# Train models using the script above with different model_type

# Compare results
uv run python scripts/mlflow/compare_experiments.py best Phase1_Foundation --metric val_mae --top-n 5
```

**Success Criteria:**
- [ ] Both models trained successfully
- [ ] MAE < 5.0 (sanity check)
- [ ] All metrics logged
- [ ] Feature importance plots generated

---

### Tuesday-Thursday: Hyperparameter Experiments

**Tasks:**
- [ ] Grid search on learning_rate [0.01, 0.05, 0.1]
- [ ] Grid search on max_depth [4, 6, 8]
- [ ] Grid search on n_estimators [100, 200, 300]
- [ ] Run ~10-15 total experiments

**Example Loop:**
```python
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
            tags={'week': '1', 'experiment_type': 'hyperparam_search'}
        )
```

**Verification:**
- [ ] All experiments complete
- [ ] Experiments visible in MLflow
- [ ] Can filter by tags
- [ ] Can compare experiments side-by-side

---

### Friday: Review & Document

**Tasks:**
```bash
# Generate report
uv run python scripts/mlflow/compare_experiments.py report Phase1_Foundation --output reports/week1_report.txt

# Export results
uv run python scripts/mlflow/compare_experiments.py export Phase1_Foundation data/results/week1_experiments.csv

# Identify best model
uv run python scripts/mlflow/compare_experiments.py best Phase1_Foundation --metric val_mae
```

**Deliverables:**
- [ ] Week 1 report generated
- [ ] Best model identified
- [ ] Experiments exported to CSV
- [ ] Next week's plan documented

---

## Week 2-3: Feature Engineering & Selection

### Tasks:
- [ ] Add opponent features
- [ ] Add temporal features (EWMA, rolling windows)
- [ ] Test different feature combinations
- [ ] Run feature selection experiments
- [ ] Compare with baseline

**Experiment Naming:**
```
week2_opponent_features_v1
week2_temporal_ewma_v1
week3_combined_features_v1
week3_feature_selection_v1
```

**Success Metrics:**
- [ ] MAE improves by >5% vs baseline
- [ ] Feature importance analysis complete
- [ ] Selected features documented

---

## Week 4-7: Advanced Features (Phase 2)

**Switch to Phase2_Advanced experiment:**
```python
trainer = NBAPropsMLflowTrainer(experiment_name="Phase2_Advanced")
```

### Tasks:
- [ ] Implement advanced opponent modeling
- [ ] Add matchup-specific features
- [ ] Test ensemble methods
- [ ] Walk-forward validation experiments

**Success Metrics:**
- [ ] MAE < 3.5
- [ ] Feature set optimized
- [ ] Ensemble shows improvement

---

## Week 8-9: Calibration (Phase 3)

**Switch to Phase3_Calibration experiment:**
```python
trainer = NBAPropsMLflowTrainer(experiment_name="Phase3_Calibration")
```

### Tasks:
- [ ] Implement isotonic calibration
- [ ] Test Platt scaling
- [ ] Optimize for betting metrics (ROI, win rate)
- [ ] Calibration curves analysis

**Success Metrics:**
- [ ] Calibration error < 5%
- [ ] Win rate > 55%
- [ ] ROI > 5%
- [ ] Brier score improved

---

## Week 10-12: Production (Phase 4)

**Switch to Phase4_Production experiment:**
```python
trainer = NBAPropsMLflowTrainer(experiment_name="Phase4_Production")
```

### Tasks:
- [ ] Walk-forward validation on 2024-25 season
- [ ] Production candidate selection
- [ ] Model promotion to staging
- [ ] Testing and validation
- [ ] Production deployment

**Example Walk-Forward Training:**
```python
results = trainer.train_and_evaluate_walk_forward(
    df=df_full,
    model_type='xgboost',
    run_name='production_candidate_v1',
    register_model=True,
    tags={'candidate': 'true', 'version': 'v2.0.0'}
)
```

**Model Promotion:**
```python
from src.mlflow_integration.registry import ModelRegistry, DEFAULT_PRODUCTION_CRITERIA

registry = ModelRegistry()

# Evaluate model
meets_criteria = registry.evaluate_for_production(
    model_name="NBAPropsModel",
    version=5,
    criteria=DEFAULT_PRODUCTION_CRITERIA
)

if meets_criteria:
    # Promote to staging
    registry.promote_model(
        model_name="NBAPropsModel",
        version=5,
        stage="Staging"
    )

    # After testing, promote to production
    registry.promote_model(
        model_name="NBAPropsModel",
        version=5,
        stage="Production"
    )
```

**Success Metrics:**
- [ ] Walk-forward MAE < 3.5
- [ ] ROI > 5% on 2024-25 validation
- [ ] Win rate > 55%
- [ ] Sharpe ratio > 1.0
- [ ] Model promoted to production

---

## End of 12 Weeks: Final Deliverables

### Completed Checklist:

**Experiments:**
- [ ] 50-100 experiments tracked
- [ ] 4 phases completed
- [ ] All metrics logged
- [ ] All artifacts saved

**Models:**
- [ ] 10-20 model versions registered
- [ ] 2-3 production-ready models
- [ ] Model lineage documented
- [ ] Rollback strategy tested

**Documentation:**
- [ ] Weekly reports generated (12 reports)
- [ ] Best models documented
- [ ] Feature importance analysis
- [ ] Performance by segment (tier, edge, season)

**Infrastructure:**
- [ ] MLflow tracking working smoothly
- [ ] Model registry operational
- [ ] Comparison utilities used weekly
- [ ] Reproducibility verified

**Final Comparison:**
```bash
# Compare all phases
uv run python scripts/mlflow/compare_experiments.py phases

# Generate final report
uv run python scripts/mlflow/compare_experiments.py report Phase4_Production --output FINAL_PROJECT_REPORT.txt
```

---

## Common Issues & Solutions

### Issue: MLflow UI won't start
**Solution:**
```bash
# Check if port 5000 is in use
lsof -i :5000

# Use different port
uv run mlflow ui --port 5001
```
- [ ] Resolved

### Issue: Can't find experiments
**Solution:**
```bash
# Verify tracking URI
export MLFLOW_TRACKING_URI=file:///Users/diyagamah/Documents/nba_props_model/mlruns
```
- [ ] Resolved

### Issue: Model not logging
**Solution:**
```python
# Enable autologging explicitly
from src.mlflow_integration.tracker import enable_autologging
enable_autologging('xgboost')
```
- [ ] Resolved

### Issue: Artifacts not saving
**Solution:**
```python
# Check artifact location
import mlflow
print(mlflow.get_artifact_uri())
```
- [ ] Resolved

---

## Weekly Review Template

### Week X Review

**Experiments Run:** ___ / 10 target
**Best MAE:** ___
**Best ROI:** ___
**Models Registered:** ___

**Wins:**
-
-
-

**Challenges:**
-
-
-

**Next Week Focus:**
-
-
-

**Action Items:**
- [ ]
- [ ]
- [ ]

---

## Final Sign-Off

### Project Complete When:

- [ ] All 50+ experiments tracked in MLflow
- [ ] Production model deployed (MAE < 3.5, ROI > 5%)
- [ ] Model registry has 10+ versions
- [ ] All 4 phases completed
- [ ] Final report generated
- [ ] Infrastructure documented
- [ ] Reproducibility verified
- [ ] Rollback tested successfully

---

## Resources

**Documentation:**
- MLFLOW_INFRASTRUCTURE_GUIDE.md (Complete reference)
- MLFLOW_QUICKSTART.md (Quick start guide)
- MLFLOW_SETUP_COMPLETE.md (Setup summary)
- MLFLOW_ARCHITECTURE.txt (Architecture diagram)

**Code:**
- src/mlflow_integration/ (Core modules)
- src/models/train_model_mlflow.py (Training script)
- scripts/mlflow/ (Utility scripts)

**MLflow Docs:**
- https://mlflow.org/docs/latest/

---

Last Updated: 2025-10-14
Project: NBA Props Model - 12 Week Development
Contact: Hivin Diyagama (hivin.diyagama@tabcorp.com.au)
