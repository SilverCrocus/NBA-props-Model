# MLflow Quick Start Guide

**5-Minute Guide to Improved MLflow Usage**

---

## Immediate Actions (Do This First)

### 1. Create StandardNBAPropsTracker (10 minutes)

Copy this file to `src/mlflow_integration/standard_tracker.py`:

```python
"""Standard MLflow tracking patterns for NBA Props Model"""

from typing import Dict, List
import pandas as pd
import hashlib
from .tracker import NBAPropsTracker

class StandardNBAPropsTracker(NBAPropsTracker):
    """Extended tracker with standardized logging patterns"""

    def log_dataset_info(self, df: pd.DataFrame, dataset_name: str, source_path: str):
        """Log complete dataset information for reproducibility"""
        df_hash = hashlib.md5(df.to_json(orient='records').encode()).hexdigest()[:12]

        self.log_params({
            f"{dataset_name}_hash": df_hash,
            f"{dataset_name}_n_samples": len(df),
            f"{dataset_name}_source": source_path,
        })

        dataset_info = {
            "dataset_name": dataset_name,
            "hash": df_hash,
            "n_samples": len(df),
            "n_features": len(df.columns),
            "date_range": {
                "min": str(df['GAME_DATE'].min()) if 'GAME_DATE' in df.columns else None,
                "max": str(df['GAME_DATE'].max()) if 'GAME_DATE' in df.columns else None,
            },
        }

        import mlflow
        mlflow.log_dict(dataset_info, f"{dataset_name}_info.json")

    def log_walk_forward_fold(
        self,
        fold_num: int,
        fold_date: str,
        predictions: pd.DataFrame,
        y_true_col: str = "PRA",
        y_pred_col: str = "predicted_PRA"
    ):
        """Log metrics for a single walk-forward fold"""
        from sklearn.metrics import mean_absolute_error, r2_score

        mae = mean_absolute_error(predictions[y_true_col], predictions[y_pred_col])
        r2 = r2_score(predictions[y_true_col], predictions[y_pred_col])

        self.log_metrics({
            f"fold_mae": mae,
            f"fold_r2": r2,
            f"fold_n_predictions": len(predictions),
        }, step=fold_num)

        import mlflow
        mlflow.set_tag(f"fold_{fold_num}_date", fold_date)

    def log_feature_lineage(
        self,
        feature_version: str,
        feature_categories: Dict[str, List[str]],
        new_features: List[str] = None,
        baseline_version: str = None
    ):
        """Log complete feature lineage for reproducibility"""
        self.log_params({
            "feature_version": feature_version,
            "n_features_total": sum(len(v) for v in feature_categories.values()),
            "baseline_version": baseline_version or "none",
        })

        lineage = {
            "version": feature_version,
            "baseline_version": baseline_version,
            "feature_categories": feature_categories,
            "new_features": new_features or [],
            "total_features": sum(len(v) for v in feature_categories.values()),
        }

        import mlflow
        mlflow.log_dict(lineage, "feature_lineage.json")
```

### 2. Update Your Training Script (5 minutes)

**Before:**
```python
from src.mlflow_integration.tracker import NBAPropsTracker

tracker = NBAPropsTracker(experiment_name="Phase1_Foundation")
tracker.start_run(run_name="my_experiment")

# ... training code ...

tracker.log_validation_metrics({"mae": mae})
tracker.end_run()
```

**After:**
```python
from src.mlflow_integration.standard_tracker import StandardNBAPropsTracker

tracker = StandardNBAPropsTracker(experiment_name="Phase1_Foundation")
tracker.start_run(
    run_name=f"20251015_p1_opponent_eff_walkforward_v1",  # Standardized name
    tags={
        "phase": "1",
        "feature_version": "v2.1_opponent_eff",
        "baseline_mae": "6.10",
        "production_ready": "false",
    }
)

# Log dataset
tracker.log_dataset_info(
    df=train_df,
    dataset_name="training",
    source_path=str(data_path)
)

# ... training code ...

# Log walk-forward folds
for i, pred_date in enumerate(val_dates):
    date_preds = predictions[predictions['GAME_DATE'] == pred_date]
    tracker.log_walk_forward_fold(
        fold_num=i,
        fold_date=str(pred_date),
        predictions=date_preds
    )

# Log feature lineage
tracker.log_feature_lineage(
    feature_version="v2.1_opponent_eff",
    feature_categories={
        "tier1_core": ["USG%", "PSA", "AST%"],
        "tier2_contextual": ["opp_DRtg", "days_rest"],
        "tier3_temporal": ["PRA_lag1", "PRA_L5_mean"],
    },
    new_features=["opp_DRtg", "opp_pace"],
    baseline_version="v2.0_day3"
)

# Log validation metrics
tracker.log_validation_metrics({"mae": mae, "r2": r2})

# Log predictions and model
tracker.log_predictions(predictions_df, "predictions.csv")
tracker.log_model(model, model_type="xgboost")

tracker.end_run()
```

### 3. Create Run Comparison Script (5 minutes)

Save as `scripts/mlflow/compare_runs.py`:

```python
"""Quick run comparison utility"""

import sys
import pandas as pd
from mlflow.tracking import MlflowClient
from tabulate import tabulate

def compare(run_ids: list):
    """Compare runs side-by-side"""
    client = MlflowClient()
    results = []

    for run_id in run_ids:
        run = client.get_run(run_id)
        results.append({
            "run_id": run_id[:8],
            "run_name": run.data.tags.get("mlflow.runName", "N/A")[:30],
            "mae": run.data.metrics.get("val_mae", "N/A"),
            "r2": run.data.metrics.get("val_r2", "N/A"),
            "features": run.data.params.get("n_features", "N/A"),
        })

    df = pd.DataFrame(results)
    print("\n" + tabulate(df, headers='keys', tablefmt='psql', showindex=False))

if __name__ == "__main__":
    compare(sys.argv[1:])
```

**Usage:**
```bash
uv run scripts/mlflow/compare_runs.py 82148e2b 7f3055df b429e7ec
```

---

## Key Patterns to Use

### Pattern 1: Dataset Logging (Always)

```python
# At start of training
tracker.log_dataset_info(
    df=train_df,
    dataset_name="training",
    source_path="data/processed/train.parquet"
)
```

**Why:** Ensures reproducibility. Know exact data used.

### Pattern 2: Walk-Forward Fold Tracking (Always)

```python
# Inside walk-forward loop
for i, pred_date in enumerate(val_dates):
    date_preds = [p for p in predictions if p['GAME_DATE'] == pred_date]

    tracker.log_walk_forward_fold(
        fold_num=i,
        fold_date=str(pred_date),
        predictions=pd.DataFrame(date_preds)
    )
```

**Why:** Debug problematic dates, visualize MAE over time.

### Pattern 3: Feature Lineage (Always)

```python
# After feature engineering
tracker.log_feature_lineage(
    feature_version="v2.1_opponent_eff",
    feature_categories={
        "tier1_core": core_features,
        "tier2_contextual": contextual_features,
        "tier3_temporal": temporal_features,
    },
    new_features=["opp_DRtg", "opp_pace", "TS%"],
    baseline_version="v2.0_day3"
)
```

**Why:** Know exactly which features are in each model.

### Pattern 4: Predictions as Artifact (Always)

```python
# Before end_run()
tracker.log_predictions(predictions_df, "validation_predictions.csv")
```

**Why:** Analyze errors post-hoc without rerunning.

### Pattern 5: Model Registration (If Good Results)

```python
# Only if model meets threshold
if mae < 6.0:
    tracker.log_model(
        model,
        model_type="xgboost",
        registered_model_name="NBAPropsModel"  # ← This registers it
    )
```

**Why:** Track production candidates in registry.

---

## Naming Conventions

### Experiment Names (Hierarchical)

```
Phase1_Features/day4_opponent_efficiency
Phase1_Architecture/two_stage
Phase2_PositionDefense/position_features
```

### Run Names (Standardized Format)

```
{date}_{phase}_{feature_set}_{validation}_{version}

Examples:
20251015_p1_opponent_eff_walkforward_v1
20251016_p1_twostage_walkforward_v2
20251020_p2_posdef_walkforward_v1
```

### Tags (Always Include)

```python
tags = {
    "phase": "1",
    "feature_version": "v2.1_opponent_eff",
    "baseline_mae": "6.10",
    "production_ready": "false",
    "model_type": "xgboost",
    "validation_type": "walk_forward",
}
```

---

## Common Mistakes to Avoid

### ❌ Don't: Use raw MLflow

```python
# BAD
import mlflow
mlflow.log_metric("mae", 6.10)
```

### ✅ Do: Use StandardNBAPropsTracker

```python
# GOOD
tracker.log_validation_metrics({"mae": 6.10})
```

---

### ❌ Don't: Only log final metrics

```python
# BAD
tracker.log_validation_metrics({"mae": overall_mae})
```

### ✅ Do: Log fold-level metrics

```python
# GOOD
for i, pred_date in enumerate(val_dates):
    tracker.log_walk_forward_fold(...)
tracker.log_validation_metrics({"mae": overall_mae})
```

---

### ❌ Don't: Forget dataset info

```python
# BAD
tracker.start_run()
# ... train ...
tracker.end_run()
```

### ✅ Do: Log dataset at start

```python
# GOOD
tracker.start_run()
tracker.log_dataset_info(df, "training", data_path)
# ... train ...
tracker.end_run()
```

---

### ❌ Don't: Use generic run names

```python
# BAD
tracker.start_run(run_name="experiment_1")
```

### ✅ Do: Use standardized names

```python
# GOOD
tracker.start_run(
    run_name="20251015_p1_opponent_eff_walkforward_v1"
)
```

---

## Quick Commands

### View experiments
```bash
uv run mlflow ui
# Then open http://localhost:5000
```

### Compare runs
```bash
uv run scripts/mlflow/compare_runs.py <run_id1> <run_id2>
```

### List experiments
```bash
uv run python -c "import mlflow; client = mlflow.tracking.MlflowClient(); [print(e.name) for e in client.search_experiments()]"
```

### Get best run
```bash
uv run python -c "
import mlflow
client = mlflow.tracking.MlflowClient()
runs = client.search_runs(
    experiment_ids=['303946714496858297'],
    order_by=['metrics.val_mae ASC'],
    max_results=1
)
print(f'Best run: {runs[0].info.run_id} (MAE: {runs[0].data.metrics[\"val_mae\"]:.2f})')
"
```

---

## Checklist for Every Training Run

Before you start:
- [ ] Use `StandardNBAPropsTracker`
- [ ] Standardized experiment name (e.g., `Phase1_Features/...`)
- [ ] Standardized run name (e.g., `20251015_p1_...`)
- [ ] Add tags (phase, feature_version, baseline_mae)

During training:
- [ ] Log dataset info at start
- [ ] Log hyperparameters
- [ ] Log training config

During validation:
- [ ] Log fold-level metrics (walk-forward)
- [ ] Log overall validation metrics
- [ ] Log feature lineage

After training:
- [ ] Log predictions as artifact
- [ ] Log model (with registration if good)
- [ ] Log feature importance
- [ ] End run with status

After run:
- [ ] Compare to baseline with `compare_runs.py`
- [ ] Document results in experiment notes
- [ ] Update roadmap if significant improvement

---

## Next Steps

1. **Create `StandardNBAPropsTracker`** (copy code above)
2. **Update one training script** as example
3. **Create comparison utility** (copy code above)
4. **Run training** and verify improvements
5. **Standardize remaining scripts**

---

**Questions?** See full analysis in `MLFLOW_ANALYSIS_AND_RECOMMENDATIONS.md`
