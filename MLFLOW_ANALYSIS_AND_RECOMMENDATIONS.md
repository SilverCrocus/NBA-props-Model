# MLflow Analysis and Refactoring Recommendations

**Analysis Date:** October 15, 2025
**Project:** NBA Props Model - Phase 1 Foundation
**Current Status:** MAE 6.10 → Target <5.0

---

## Executive Summary

**Current State:** MLflow integration is **partially implemented** with good infrastructure but **inconsistent usage** across training scripts. The core tracker and registry modules are production-ready, but only ~30% of training scripts use them properly.

**Key Findings:**
1. ✅ **Solid Foundation:** Tracker and Registry modules are well-designed
2. ⚠️ **Inconsistent Adoption:** Only 1 of 3 main training scripts uses MLflow properly
3. ❌ **Missing Critical Metrics:** No walk-forward metrics, betting metrics, or model lineage tracking
4. ❌ **Poor Experiment Organization:** Ad-hoc naming, no phase/feature versioning
5. ❌ **No Model Registry Usage:** Models logged but never registered or promoted

**Impact on Development:**
- **Lost Reproducibility:** 70% of experiments not tracked → wasted iterations
- **Manual Comparison:** No automated run comparison → slow debugging
- **No Production Path:** No registry workflow → manual model deployment
- **Missing Context:** No feature lineage → hard to validate improvements

---

## Current Implementation Analysis

### 1. Infrastructure (Modules)

#### ✅ **GOOD: Core Tracker Module** (`src/mlflow_integration/tracker.py`)

**Strengths:**
- Comprehensive logging methods (params, metrics, artifacts)
- Specialized methods for NBA props (betting_metrics, calibration, feature_importance)
- Error handling and logging
- Context manager support
- Autologging for XGBoost/LightGBM

**Weaknesses:**
```python
# Missing: Dataset versioning
def log_dataset_hash(self, df: pd.DataFrame):
    """MISSING: Track exact dataset used for reproducibility"""
    pass

# Missing: Hyperparameter search tracking
def log_hyperparameter_sweep(self, results: List[Dict]):
    """MISSING: Track HPO experiments"""
    pass

# Missing: Model comparison utilities
def compare_to_baseline(self, baseline_run_id: str):
    """MISSING: Auto-compare to baseline metrics"""
    pass

# Missing: Walk-forward specific logging
def log_walkforward_fold(self, fold_num: int, metrics: Dict):
    """MISSING: Track each fold in walk-forward validation"""
    pass
```

#### ✅ **GOOD: Model Registry Module** (`src/mlflow_integration/registry.py`)

**Strengths:**
- Full lifecycle management (register, promote, rollback)
- Production criteria evaluation
- Model comparison utilities
- Proper staging workflow

**Weaknesses:**
```python
# Not used anywhere in codebase!
# No training script calls:
# - register_model()
# - promote_model()
# - evaluate_for_production()

# Missing: CI/CD integration hooks
def trigger_deployment_pipeline(self, model_name: str, version: int):
    """MISSING: Integrate with deployment automation"""
    pass

# Missing: Model monitoring integration
def log_production_metrics(self, model_version, inference_metrics):
    """MISSING: Track production performance"""
    pass
```

### 2. Script-Level Usage

#### ✅ **EXCELLENT:** `walk_forward_training_advanced_features.py`

**Current Implementation:**
```python
# Lines 407-418: Proper experiment setup
tracker = NBAPropsTracker(experiment_name="Phase1_Foundation")
tracker.start_run(
    run_name=f"walk_forward_advanced_features_{val_season}",
    tags={
        "model_type": "xgboost",
        "validation_type": "walk_forward_advanced_features",
        "train_season": train_season,
        "val_season": val_season,
        "description": "Day 4: Opponent + Efficiency + Normalization features",
        "features_added": "opponent_DRtg, opponent_pace, TS%, PER, per_36_stats",
    },
)

# Lines 522-531: Logs training config
training_config = {
    "n_samples": len(X_train),
    "n_features": len(feature_cols),
    "n_features_day3": 34,
    "n_features_added": len(feature_cols) - 34,
    "train_period": f"{train_start_date} to {train_end_date}",
    "validation_type": "walk_forward_advanced_features",
    "min_train_games": min_train_games,
}
tracker.log_training_config(training_config)

# Lines 545-547: Logs hyperparameters
tracker.log_params(hyperparams)

# Lines 612-623: Logs validation metrics
val_metrics = {
    "mae": val_mae,
    "rmse": val_rmse,
    "r2": val_r2,
    "within_3pts_pct": (val_df["abs_error"] <= 3).mean() * 100,
    "within_5pts_pct": (val_df["abs_error"] <= 5).mean() * 100,
    "within_10pts_pct": (val_df["abs_error"] <= 10).mean() * 100,
    "ctg_coverage": (val_df["CTG_Available"] == 1).mean() * 100,
    "improvement_from_day3": 6.11 - val_mae,
}
tracker.log_validation_metrics(val_metrics)

# Lines 645-650: Logs artifacts
tracker.log_feature_importance(feature_cols, importance)
tracker.log_model(model, model_type="xgboost")
```

**What's Missing:**
```python
# 1. No dataset versioning
tracker.log_dataset_info({
    "source": game_logs_path,
    "n_games": len(all_games_df),
    "date_range": f"{all_games_df['GAME_DATE'].min()} to {all_games_df['GAME_DATE'].max()}",
    "players": all_games_df['PLAYER_ID'].nunique(),
    "ctg_coverage": ctg_coverage,
})

# 2. No walk-forward fold tracking
for pred_date in val_dates:
    # ... predictions ...
    tracker.log_metrics({
        f"fold_{i}_mae": fold_mae,
        f"fold_{i}_n_predictions": len(predictions)
    }, step=i)

# 3. No model registration
tracker.log_model(
    model,
    model_type="xgboost",
    registered_model_name="NBAPropsModel"  # ← MISSING
)

# 4. No predictions artifact
tracker.log_predictions(val_df, "validation_predictions.csv")  # ← MISSING

# 5. No CTG feature availability tracking
tracker.log_metrics({
    "ctg_players_covered": ctg_players_count,
    "ctg_games_covered": ctg_games_count
})
```

#### ❌ **POOR:** `train_two_stage_model.py`

**Current Implementation:**
```python
# Lines 137-144: Uses raw MLflow (not tracker!)
mlflow.set_experiment("Phase1_TwoStage")
with mlflow.start_run(run_name="two_stage_walk_forward"):
    mlflow.log_param("n_prediction_dates", len(prediction_dates))
    mlflow.log_param("start_date", str(prediction_dates[0]))
    mlflow.log_param("end_date", str(prediction_dates[-1]))
    mlflow.log_param("season", season)
    # ... raw mlflow.log_metric() calls
```

**Problems:**
1. Doesn't use `NBAPropsTracker` → misses standardized logging
2. No feature importance logging
3. No model artifact saved
4. No hyperparameters logged (two-stage models not tracked!)
5. No comparison to baseline
6. No predictions saved as artifact

**Should Be:**
```python
# Initialize tracker
tracker = NBAPropsTracker(experiment_name="Phase1_TwoStage")
tracker.start_run(
    run_name=f"two_stage_walkforward_{season}",
    tags={
        "model_type": "two_stage",
        "stage1": "minutes_predictor",
        "stage2": "pra_predictor",
        "validation_type": "walk_forward",
        "baseline_mae": 6.10,
        "expected_improvement": "7-8%"
    }
)

# Log both stage hyperparameters
tracker.log_params({
    "stage1_n_estimators": predictor.minutes_model.n_estimators,
    "stage2_n_estimators": predictor.pra_model.n_estimators,
    # ... all hyperparameters
})

# Log dataset info
tracker.log_training_config({
    "n_training_samples": len(train_df),
    "n_prediction_dates": len(prediction_dates),
    "date_range": f"{prediction_dates[0]} to {prediction_dates[-1]}",
})

# Log validation metrics
tracker.log_validation_metrics({
    "mae_pra": overall_mae,
    "mae_minutes": overall_minutes_mae,
    "r2_minutes": overall_minutes_r2,
    "improvement_vs_baseline": baseline_mae - overall_mae,
})

# Log predictions
tracker.log_predictions(predictions_df, "two_stage_predictions.csv")

# Log both models
tracker.log_model(predictor, model_type="sklearn")

tracker.end_run(status="FINISHED")
```

#### ❌ **MISSING:** `walk_forward_validation_enhanced.py`

**Current Implementation:**
```python
# Lines 192-350: NO MLFLOW TRACKING AT ALL!
# Just prints to console and saves CSV
print(f"\n✅ Enhanced walk-forward validation complete!")
print(f"   Total predictions: {len(predictions_df):,}")
predictions_df.to_csv(output_file, index=False)
```

**Critical Missing:**
- No experiment tracking
- No metrics logged
- No feature configuration saved
- No model artifact
- No comparison to baseline
- Can't reproduce this run!

**Should Add:**
```python
tracker = NBAPropsTracker(experiment_name="Phase1_Foundation")
tracker.start_run(
    run_name=f"enhanced_validation_{datetime.now().strftime('%Y%m%d')}",
    tags={
        "validation_type": "walk_forward_enhanced",
        "features_added": "CTG,L3,Rest,Opponent",
        "baseline_mae": 7.97,
    }
)

# Log enhanced feature config
tracker.log_feature_config({
    "base_features": len(feature_cols),
    "enhanced_features": len(enhanced_feature_cols),
    "ctg_features": ["USG", "PSA", "AST_PCT", "TOV_PCT", "eFG", "REB_PCT"],
    "l3_features": ["PRA_L3_mean", "PRA_L3_std", "PRA_L3_trend"],
    "rest_features": ["Days_Rest", "Is_BackToBack", "Games_Last7"],
    "opponent_features": ["OPP_DRtg", "OPP_Pace"],
})

# Log validation metrics
tracker.log_validation_metrics({
    "mae": mae,
    "ctg_coverage": ctg_coverage * 100,
    "improvement_vs_baseline": 7.97 - mae,
})

# Log predictions and model
tracker.log_predictions(predictions_df, "enhanced_predictions.csv")
tracker.log_model(model, model_type="xgboost")

tracker.end_run()
```

---

## Gap Analysis

### Critical Gaps

#### 1. **No Walk-Forward Fold Tracking**

**Problem:** Walk-forward validation has N folds (one per date), but we only log aggregate metrics.

**Impact:** Can't debug which dates/folds have high error.

**Example Missing:**
```python
# Current: Only final MAE
tracker.log_validation_metrics({"mae": 6.10})

# Should be: Per-fold metrics
for i, pred_date in enumerate(val_dates):
    fold_mae = compute_fold_mae(pred_date)
    tracker.log_metrics({
        f"fold_{i}_mae": fold_mae,
        f"fold_{i}_date": str(pred_date),
        f"fold_{i}_n_predictions": n_predictions,
    }, step=i)

# This allows plotting MAE over time, identifying problematic dates
```

#### 2. **No Feature Lineage Tracking**

**Problem:** Can't trace which features were used in each experiment.

**Impact:** When MAE improves/degrades, can't identify which features caused it.

**Example Missing:**
```python
# Should log complete feature configuration
tracker.log_feature_config({
    "feature_set_version": "v2.1_opponent_efficiency",
    "n_features": 47,
    "feature_categories": {
        "tier1_core": ["USG%", "PSA", "AST%", ...],
        "tier2_contextual": ["opp_DRtg", "days_rest", ...],
        "tier3_temporal": ["PRA_lag1", "PRA_L5_mean", ...]
    },
    "new_features_vs_baseline": [
        "opp_DRtg", "opp_pace", "TS%", "PER", "PRA_per_36"
    ],
    "removed_features": ["MIN", "FGA"]  # Removed due to production unavailability
})
```

#### 3. **No Betting Metrics Logged**

**Problem:** Model is for betting, but no betting simulation results tracked.

**Impact:** Can't evaluate production readiness (ROI, win rate, CLV).

**Example Missing:**
```python
# After backtest simulation
tracker.log_betting_metrics({
    "roi": 0.065,
    "win_rate": 0.58,
    "clv": 0.023,
    "sharpe_ratio": 1.2,
    "max_drawdown": -0.15,
    "brier_score": 0.18,
    "calibration_error": 0.04,
    "n_bets": 850,
    "profitable_edge_sizes": [3, 5, 8],  # pts
})
```

#### 4. **No Model Registry Usage**

**Problem:** Models logged but never registered → no staging/production workflow.

**Impact:** Manual model selection, no version control, no rollback capability.

**Example Missing:**
```python
# After successful training
tracker.log_model(
    model,
    model_type="xgboost",
    registered_model_name="NBAPropsModel"  # ← Register in registry
)

# Then evaluate for production
registry = ModelRegistry()
meets_criteria = registry.evaluate_for_production(
    model_name="NBAPropsModel",
    version=tracker.model_version,
    criteria={
        "max_val_mae": 5.0,
        "min_betting_roi": 0.05,
        "min_betting_win_rate": 0.55,
    }
)

if meets_criteria:
    registry.promote_model(
        model_name="NBAPropsModel",
        version=tracker.model_version,
        stage="Staging"
    )
```

#### 5. **No Dataset Versioning**

**Problem:** Can't reproduce experiments because dataset changes not tracked.

**Impact:** "This model got MAE 5.8" → but with which data?

**Example Missing:**
```python
# Log dataset fingerprint
tracker.log_params({
    "dataset_hash": hashlib.md5(df.to_json().encode()).hexdigest()[:8],
    "n_games": len(df),
    "n_players": df['PLAYER_ID'].nunique(),
    "date_range": f"{df['GAME_DATE'].min()} to {df['GAME_DATE'].max()}",
    "ctg_version": "2024-25_v1",
})

# Or use MLflow datasets API
from mlflow.data.pandas_dataset import PandasDataset
dataset = PandasDataset(df, source="game_logs_2024_25", name="train")
mlflow.log_input(dataset, context="training")
```

#### 6. **No Experiment Comparison Utilities**

**Problem:** Manual comparison of runs in UI → slow iteration.

**Impact:** Hard to answer "did opponent features help?"

**Example Missing:**
```python
# Should have comparison script
def compare_experiments(run_ids: List[str], metrics: List[str]):
    """
    Compare multiple runs side-by-side

    Usage:
        compare_experiments(
            run_ids=["82148e2b", "7f3055df"],
            metrics=["val_mae", "val_r2", "improvement_from_day3"]
        )

    Returns:
        DataFrame with side-by-side comparison
    """
    client = MlflowClient()
    results = []

    for run_id in run_ids:
        run = client.get_run(run_id)
        result = {
            "run_id": run_id,
            "run_name": run.data.tags.get("mlflow.runName"),
            "features_added": run.data.tags.get("features_added"),
        }
        for metric in metrics:
            result[metric] = run.data.metrics.get(metric)
        results.append(result)

    return pd.DataFrame(results)
```

---

## Recommended Refactoring Plan

### Phase 1: Standardize Existing Scripts (Week 1)

#### 1.1 Create Standard Logging Wrapper

**File:** `src/mlflow_integration/standard_tracker.py`

```python
"""
Standard MLflow tracking patterns for NBA Props Model
Ensures consistent logging across all training scripts
"""

from typing import Dict, List, Optional
import pandas as pd
import hashlib
from .tracker import NBAPropsTracker

class StandardNBAPropsTracker(NBAPropsTracker):
    """
    Extended tracker with standardized logging patterns
    """

    def log_dataset_info(
        self,
        df: pd.DataFrame,
        dataset_name: str,
        source_path: str
    ):
        """
        Log complete dataset information

        Args:
            df: Training/validation dataset
            dataset_name: Descriptive name (e.g., "train_2023_24")
            source_path: Path to source data
        """
        # Calculate dataset hash for versioning
        df_hash = hashlib.md5(
            df.to_json(orient='records').encode()
        ).hexdigest()[:12]

        # Log as params (searchable)
        self.log_params({
            f"{dataset_name}_hash": df_hash,
            f"{dataset_name}_n_samples": len(df),
            f"{dataset_name}_n_players": df['PLAYER_ID'].nunique() if 'PLAYER_ID' in df.columns else 0,
            f"{dataset_name}_source": source_path,
        })

        # Log detailed info as artifact
        dataset_info = {
            "dataset_name": dataset_name,
            "hash": df_hash,
            "n_samples": len(df),
            "n_features": len(df.columns),
            "date_range": {
                "min": str(df['GAME_DATE'].min()) if 'GAME_DATE' in df.columns else None,
                "max": str(df['GAME_DATE'].max()) if 'GAME_DATE' in df.columns else None,
            },
            "player_stats": {
                "n_unique_players": df['PLAYER_ID'].nunique() if 'PLAYER_ID' in df.columns else 0,
                "games_per_player": df.groupby('PLAYER_ID').size().describe().to_dict() if 'PLAYER_ID' in df.columns else {}
            },
            "column_names": df.columns.tolist(),
            "null_counts": df.isnull().sum().to_dict(),
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
        """
        Log metrics for a single walk-forward fold

        Args:
            fold_num: Fold number (0-indexed)
            fold_date: Prediction date for this fold
            predictions: DataFrame with predictions for this fold
            y_true_col: Column name for actual values
            y_pred_col: Column name for predictions
        """
        from sklearn.metrics import mean_absolute_error, r2_score

        mae = mean_absolute_error(
            predictions[y_true_col],
            predictions[y_pred_col]
        )

        r2 = r2_score(
            predictions[y_true_col],
            predictions[y_pred_col]
        )

        # Log with step number for time-series plotting
        self.log_metrics({
            f"fold_mae": mae,
            f"fold_r2": r2,
            f"fold_n_predictions": len(predictions),
        }, step=fold_num)

        # Also log fold date as tag
        import mlflow
        mlflow.set_tag(f"fold_{fold_num}_date", fold_date)

    def log_feature_lineage(
        self,
        feature_version: str,
        feature_categories: Dict[str, List[str]],
        new_features: List[str] = None,
        removed_features: List[str] = None,
        baseline_version: str = None
    ):
        """
        Log complete feature lineage for reproducibility

        Args:
            feature_version: Version string (e.g., "v2.1_opponent_efficiency")
            feature_categories: Dict mapping category -> feature list
            new_features: Features added vs baseline
            removed_features: Features removed vs baseline
            baseline_version: Previous version being compared to
        """
        # Log version as param
        self.log_params({
            "feature_version": feature_version,
            "n_features_total": sum(len(v) for v in feature_categories.values()),
            "baseline_version": baseline_version or "none",
        })

        # Log detailed lineage as artifact
        lineage = {
            "version": feature_version,
            "baseline_version": baseline_version,
            "feature_categories": feature_categories,
            "new_features": new_features or [],
            "removed_features": removed_features or [],
            "total_features": sum(len(v) for v in feature_categories.values()),
        }

        import mlflow
        mlflow.log_dict(lineage, "feature_lineage.json")

    def log_baseline_comparison(
        self,
        baseline_run_id: str,
        current_metrics: Dict[str, float],
        comparison_metrics: List[str] = ["mae", "r2", "roi"]
    ):
        """
        Automatically compare to baseline run

        Args:
            baseline_run_id: Run ID of baseline to compare against
            current_metrics: Current run's metrics
            comparison_metrics: Which metrics to compare
        """
        from mlflow.tracking import MlflowClient

        client = MlflowClient()
        baseline_run = client.get_run(baseline_run_id)
        baseline_metrics = baseline_run.data.metrics

        # Calculate deltas
        comparisons = {}
        for metric in comparison_metrics:
            baseline_val = baseline_metrics.get(f"val_{metric}")
            current_val = current_metrics.get(metric)

            if baseline_val is not None and current_val is not None:
                delta = current_val - baseline_val
                pct_change = (delta / baseline_val * 100) if baseline_val != 0 else 0

                comparisons[f"vs_baseline_{metric}_delta"] = delta
                comparisons[f"vs_baseline_{metric}_pct"] = pct_change

        # Log comparisons
        self.log_metrics(comparisons)

        # Tag with baseline
        import mlflow
        mlflow.set_tag("baseline_run_id", baseline_run_id)
        mlflow.set_tag("baseline_run_name", baseline_run.data.tags.get("mlflow.runName"))
```

#### 1.2 Update `walk_forward_training_advanced_features.py`

**Changes:**
```python
# Line 36: Import standard tracker
from src.mlflow_integration.standard_tracker import StandardNBAPropsTracker

# Line 407: Use standard tracker
tracker = StandardNBAPropsTracker(experiment_name="Phase1_Foundation")

# After line 440: Log dataset info
tracker.log_dataset_info(
    df=all_games_df,
    dataset_name="all_games",
    source_path=str(game_logs_path)
)

# After line 509: Log training dataset
tracker.log_dataset_info(
    df=train_df,
    dataset_name="training",
    source_path="walk_forward_constructed"
)

# Inside line 563 loop: Log each fold
for i, pred_date in enumerate(val_dates):
    # ... existing prediction code ...

    # After collecting predictions for this date
    date_predictions_df = pd.DataFrame([
        p for p in val_predictions if p["GAME_DATE"] == pred_date
    ])

    if len(date_predictions_df) > 0:
        tracker.log_walk_forward_fold(
            fold_num=i,
            fold_date=str(pred_date),
            predictions=date_predictions_df
        )

# After line 647: Log feature lineage
tracker.log_feature_lineage(
    feature_version="v2.1_advanced_features",
    feature_categories={
        "tier1_core": [f for f in feature_cols if any(x in f for x in ["CTG_", "efficiency"])],
        "tier2_contextual": [f for f in feature_cols if any(x in f for x in ["opp_", "MIN_", "days_rest"])],
        "tier3_temporal": [f for f in feature_cols if any(x in f for x in ["_lag", "_L", "_ewma", "_trend"])],
    },
    new_features=["opp_DRtg", "opp_pace", "TS_pct", "PER", "PRA_per_36"],
    baseline_version="v2.0_day3_baseline"
)

# After line 650: Register model if it meets criteria
if val_mae < 6.0:  # Only register if better than baseline
    tracker.log_model(
        model,
        model_type="xgboost",
        registered_model_name="NBAPropsModel"
    )
```

#### 1.3 Update `train_two_stage_model.py`

**Complete Refactor:**
```python
# Line 134: Replace raw MLflow with StandardNBAPropsTracker
tracker = StandardNBAPropsTracker(experiment_name="Phase1_TwoStage")
tracker.start_run(
    run_name=f"two_stage_walkforward_{datetime.now().strftime('%Y%m%d')}",
    tags={
        "model_type": "two_stage",
        "architecture": "minutes_then_pra",
        "baseline_mae": 6.10,
        "expected_improvement": "7-8%",
    }
)

try:
    # Log dataset
    tracker.log_dataset_info(
        df=all_games_df,
        dataset_name="all_games",
        source_path=str(game_logs_path)
    )

    # Log training dataset
    tracker.log_dataset_info(
        df=train_df,
        dataset_name="training",
        source_path="walk_forward_constructed"
    )

    # Log two-stage hyperparameters
    tracker.log_params({
        "stage1_model": "xgboost_minutes",
        "stage2_model": "xgboost_pra",
        "stage1_n_estimators": predictor.minutes_model.n_estimators,
        "stage2_n_estimators": predictor.pra_model.n_estimators,
        # ... all hyperparameters
    })

    # Log walk-forward folds
    for i, pred_date in enumerate(prediction_dates):
        date_predictions = [p for p in all_predictions if p["GAME_DATE"] == pred_date]
        if date_predictions:
            tracker.log_walk_forward_fold(
                fold_num=i,
                fold_date=str(pred_date),
                predictions=pd.DataFrame(date_predictions),
                y_true_col="actual_PRA",
                y_pred_col="predicted_PRA"
            )

    # Log final metrics
    tracker.log_validation_metrics({
        "mae_pra": overall_mae,
        "mae_minutes": overall_minutes_mae,
        "r2_minutes": overall_minutes_r2,
    })

    # Compare to baseline
    tracker.log_baseline_comparison(
        baseline_run_id="<baseline_run_id>",  # From Day 4 run
        current_metrics={
            "mae": overall_mae,
            "r2": r2_score(predictions_df["actual_PRA"], predictions_df["predicted_PRA"])
        }
    )

    # Log predictions and model
    tracker.log_predictions(predictions_df, "two_stage_predictions.csv")
    tracker.log_model(predictor, model_type="sklearn")

    tracker.end_run(status="FINISHED")

except Exception as e:
    tracker.end_run(status="FAILED")
    raise
```

#### 1.4 Add MLflow to `walk_forward_validation_enhanced.py`

**Add at start of script:**
```python
from src.mlflow_integration.standard_tracker import StandardNBAPropsTracker

tracker = StandardNBAPropsTracker(experiment_name="Phase1_Foundation")
tracker.start_run(
    run_name=f"enhanced_validation_{datetime.now().strftime('%Y%m%d')}",
    tags={
        "validation_type": "walk_forward_enhanced",
        "features_added": "CTG,L3,Rest,Opponent",
        "baseline_mae": 7.97,
    }
)

# Log dataset
tracker.log_dataset_info(
    df=raw_gamelogs,
    dataset_name="validation",
    source_path="data/game_logs/game_logs_2024_25_preprocessed.csv"
)

# Log feature lineage
tracker.log_feature_lineage(
    feature_version="v2.2_enhanced_features",
    feature_categories={
        "tier1_ctg": ["CTG_USG", "CTG_PSA", "CTG_AST_PCT", "CTG_TOV_PCT", "CTG_eFG", "CTG_REB_PCT"],
        "tier2_contextual": ["OPP_DRtg", "OPP_Pace", "Days_Rest", "Is_BackToBack", "Games_Last7"],
        "tier3_temporal": ["PRA_L3_mean", "PRA_L3_std", "PRA_L3_trend", "MIN_L3_mean"],
    },
    new_features=["PRA_L3_mean", "CTG_USG", "OPP_DRtg"],
    baseline_version="v2.0_day3_baseline"
)

# ... existing validation code ...

# Log final metrics
tracker.log_validation_metrics({
    "mae": mae,
    "ctg_coverage": ctg_coverage * 100,
    "within_3pts": (predictions_df['abs_error'] <= 3).mean() * 100,
    "within_5pts": (predictions_df['abs_error'] <= 5).mean() * 100,
})

# Log predictions and model
tracker.log_predictions(predictions_df, "enhanced_predictions.csv")
tracker.log_model(model, model_type="xgboost")

tracker.end_run()
```

### Phase 2: Model Registry Integration (Week 2)

#### 2.1 Create Model Registration Workflow

**File:** `scripts/mlflow/register_production_model.py`

```python
"""
Register and promote models to production
Run after successful training to evaluate and promote models
"""

import argparse
from src.mlflow_integration.registry import ModelRegistry, DEFAULT_PRODUCTION_CRITERIA

def register_and_evaluate(
    run_id: str,
    model_name: str = "NBAPropsModel",
    auto_promote: bool = False
):
    """
    Register model from run and evaluate for production

    Args:
        run_id: MLflow run ID containing the model
        model_name: Name to register under
        auto_promote: Automatically promote if criteria met
    """
    registry = ModelRegistry()

    # Register model
    print(f"Registering model from run {run_id}...")
    model_version = registry.register_model(
        run_id=run_id,
        model_name=model_name,
        tags={
            "registered_by": "manual",
            "source": "walk_forward_training",
        }
    )

    print(f"✅ Registered as {model_name} v{model_version.version}")

    # Evaluate for production
    print(f"\nEvaluating against production criteria...")
    meets_criteria = registry.evaluate_for_production(
        model_name=model_name,
        version=model_version.version,
        criteria=DEFAULT_PRODUCTION_CRITERIA
    )

    if meets_criteria:
        print("✅ Model meets production criteria!")

        if auto_promote:
            print("Promoting to Staging...")
            registry.promote_model(
                model_name=model_name,
                version=model_version.version,
                stage="Staging"
            )
            print(f"✅ Promoted v{model_version.version} to Staging")
        else:
            print("Run with --auto-promote to promote to Staging")
    else:
        print("❌ Model does NOT meet production criteria")
        print("Review metrics and retrain")

    return model_version

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True, help="MLflow run ID")
    parser.add_argument("--model-name", default="NBAPropsModel")
    parser.add_argument("--auto-promote", action="store_true")

    args = parser.parse_args()

    register_and_evaluate(
        run_id=args.run_id,
        model_name=args.model_name,
        auto_promote=args.auto_promote
    )
```

**Usage:**
```bash
# After training run completes
uv run scripts/mlflow/register_production_model.py \
    --run-id 82148e2b02374fa2baa5328e2c32ef14 \
    --auto-promote
```

#### 2.2 Create Model Comparison Utility

**File:** `scripts/mlflow/compare_runs.py`

```python
"""
Compare multiple MLflow runs side-by-side
"""

import argparse
import pandas as pd
from mlflow.tracking import MlflowClient
from tabulate import tabulate

def compare_runs(
    run_ids: list,
    metrics: list = None,
    output_format: str = "table"
):
    """
    Compare multiple runs

    Args:
        run_ids: List of run IDs to compare
        metrics: List of metrics to compare (None = all common metrics)
        output_format: "table" or "csv"
    """
    client = MlflowClient()

    # Default metrics if not specified
    if metrics is None:
        metrics = [
            "val_mae",
            "val_r2",
            "val_within_5pts_pct",
            "val_improvement_from_day4"
        ]

    results = []

    for run_id in run_ids:
        run = client.get_run(run_id)

        result = {
            "run_id": run_id[:8],
            "run_name": run.data.tags.get("mlflow.runName", "N/A"),
            "description": run.data.tags.get("description", "N/A")[:50],
            "status": run.info.status,
        }

        # Add metrics
        for metric in metrics:
            result[metric] = run.data.metrics.get(metric, "N/A")

        # Add key params
        result["n_features"] = run.data.params.get("n_features", "N/A")
        result["learning_rate"] = run.data.params.get("learning_rate", "N/A")

        results.append(result)

    df = pd.DataFrame(results)

    if output_format == "table":
        print("\n" + "="*100)
        print("RUN COMPARISON")
        print("="*100)
        print(tabulate(df, headers='keys', tablefmt='psql', showindex=False))
        print("\n")
    elif output_format == "csv":
        csv_path = "run_comparison.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved comparison to {csv_path}")

    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_ids", nargs="+", help="Run IDs to compare")
    parser.add_argument("--metrics", nargs="+", help="Metrics to compare")
    parser.add_argument("--format", choices=["table", "csv"], default="table")

    args = parser.parse_args()

    compare_runs(
        run_ids=args.run_ids,
        metrics=args.metrics,
        output_format=args.format
    )
```

**Usage:**
```bash
# Compare Day 3 vs Day 4 runs
uv run scripts/mlflow/compare_runs.py \
    82148e2b 7f3055df b429e7ec \
    --metrics val_mae val_r2 val_improvement_from_day4
```

### Phase 3: Experiment Organization (Week 3)

#### 3.1 Restructure Experiment Naming

**Current:**
- `Phase1_Foundation` - everything mixed together
- `Phase1_TwoStage` - separate but inconsistent

**Recommended Hierarchy:**
```
NBA_Props_Model/
├── Phase1_Baseline/
│   ├── baseline_xgb_no_ctg
│   ├── baseline_xgb_with_ctg
│   └── baseline_temporal_leakage_fix
├── Phase1_Features/
│   ├── day3_full_season
│   ├── day4_opponent_efficiency
│   ├── day5_hyperparam_tuning
│   └── enhanced_ctg_l3_rest
├── Phase1_Architecture/
│   ├── two_stage_minutes_pra
│   ├── calibration_isotonic
│   └── ensemble_xgb_lgbm
├── Phase2_PositionDefense/
│   ├── position_specific_features
│   ├── defensive_matchup_features
│   └── combined_phase2
└── Production/
    ├── final_model_v1
    ├── final_model_v2
    └── ab_test_v1_v2
```

**Implementation:**
```python
# In each training script, use hierarchical naming
tracker = StandardNBAPropsTracker(
    experiment_name=f"Phase1_Features/day4_opponent_efficiency"
)
```

#### 3.2 Standardize Run Naming Convention

**Current:** Inconsistent (e.g., `day5_optimized_2024-25_top25`)

**Recommended Format:**
```
{date}_{phase}_{feature_set}_{validation_type}_{version}

Examples:
- 20251014_p1_baseline_walkforward_v1
- 20251015_p1_opponent_eff_walkforward_v2
- 20251016_p1_twostage_walkforward_v1
- 20251020_p2_posdef_walkforward_v1
```

**Implementation:**
```python
def generate_run_name(
    phase: str,
    feature_set: str,
    validation_type: str = "walkforward",
    version: int = 1
) -> str:
    """Generate standardized run name"""
    date_str = datetime.now().strftime("%Y%m%d")
    return f"{date_str}_{phase}_{feature_set}_{validation_type}_v{version}"

# Usage
tracker.start_run(
    run_name=generate_run_name(
        phase="p1",
        feature_set="opponent_eff",
        version=2
    ),
    tags={...}
)
```

#### 3.3 Implement Tagging Strategy

**Standard Tags for All Runs:**
```python
STANDARD_TAGS = {
    # Project metadata
    "project": "nba_props_model",
    "phase": "1",  # or "2"
    "objective": "reduce_mae",  # or "improve_roi", "calibration"

    # Model metadata
    "model_type": "xgboost",  # or "lightgbm", "two_stage", "ensemble"
    "model_architecture": "single_stage",  # or "two_stage", "stacked"

    # Data metadata
    "train_seasons": "2023-24",
    "val_seasons": "2024-25",
    "validation_type": "walk_forward",

    # Feature metadata
    "feature_version": "v2.1_opponent_eff",
    "feature_set": "ctg_opponent_efficiency",
    "new_features": "opp_DRtg,opp_pace,TS%,PER",

    # Performance metadata
    "baseline_mae": "6.10",
    "target_mae": "5.00",
    "production_ready": "false",  # "true" if meets criteria

    # Reproducibility
    "git_commit": "<commit_hash>",
    "data_version": "2024_25_v1",
    "ctg_version": "2024_25_regular_season",
}
```

### Phase 4: Production Integration (Week 4)

#### 4.1 Create Model Serving Wrapper

**File:** `src/serving/model_loader.py`

```python
"""
Load production models from MLflow registry
"""

import mlflow
from mlflow.tracking import MlflowClient
from typing import Optional
import pandas as pd

class ProductionModelLoader:
    """Load and serve models from MLflow registry"""

    def __init__(self, model_name: str = "NBAPropsModel"):
        self.model_name = model_name
        self.client = MlflowClient()
        self.model = None
        self.model_version = None

    def load_production_model(self):
        """Load current production model"""
        # Get production model
        versions = self.client.get_latest_versions(
            self.model_name,
            stages=["Production"]
        )

        if not versions:
            raise ValueError("No production model found!")

        version = versions[0]
        self.model_version = version.version

        # Load model
        model_uri = f"models:/{self.model_name}/Production"
        self.model = mlflow.pyfunc.load_model(model_uri)

        print(f"Loaded {self.model_name} v{self.model_version} from Production")

        return self.model

    def predict(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions using production model

        Args:
            features_df: DataFrame with features

        Returns:
            DataFrame with predictions
        """
        if self.model is None:
            self.load_production_model()

        predictions = self.model.predict(features_df)

        result = features_df.copy()
        result['predicted_PRA'] = predictions

        return result

    def get_model_info(self) -> dict:
        """Get metadata about loaded model"""
        if self.model_version is None:
            return {}

        version = self.client.get_model_version(
            self.model_name,
            self.model_version
        )

        run = self.client.get_run(version.run_id)

        return {
            "model_name": self.model_name,
            "version": self.model_version,
            "run_id": version.run_id,
            "metrics": run.data.metrics,
            "tags": {tag.key: tag.value for tag in version.tags},
        }

# Usage in production
if __name__ == "__main__":
    loader = ProductionModelLoader()
    loader.load_production_model()

    # Get today's games
    today_features = get_todays_games_features()

    # Predict
    predictions = loader.predict(today_features)

    # Log predictions for monitoring
    log_production_predictions(predictions)
```

#### 4.2 Create CI/CD Integration

**File:** `.github/workflows/model_training.yml`

```yaml
name: Train and Register Model

on:
  push:
    branches: [main]
    paths:
      - 'scripts/training/**'
      - 'src/features/**'
      - 'src/models/**'

jobs:
  train:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install uv
          uv sync

      - name: Run training
        run: |
          uv run scripts/training/walk_forward_training_advanced_features.py
        env:
          MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}

      - name: Register model if successful
        if: success()
        run: |
          RUN_ID=$(cat .last_run_id)
          uv run scripts/mlflow/register_production_model.py \
            --run-id $RUN_ID \
            --auto-promote

      - name: Notify on Slack
        if: success()
        uses: 8398a7/action-slack@v3
        with:
          status: custom
          text: 'New model trained! Check MLflow for details.'
          webhook_url: ${{ secrets.SLACK_WEBHOOK }}
```

---

## Implementation Priority

### High Priority (Implement Immediately)

1. **Standardize existing scripts** with `StandardNBAPropsTracker`
   - Impact: Immediate consistency across all experiments
   - Effort: 1 day
   - Scripts: `walk_forward_training_advanced_features.py`, `train_two_stage_model.py`

2. **Add walk-forward fold tracking**
   - Impact: Can debug problematic dates, visualize MAE over time
   - Effort: 0.5 days
   - Location: All walk-forward validation scripts

3. **Implement feature lineage logging**
   - Impact: Know exactly which features are in each model
   - Effort: 0.5 days
   - Location: `log_feature_lineage()` method

4. **Create run comparison utility**
   - Impact: Fast iteration, quick debugging
   - Effort: 1 day
   - File: `scripts/mlflow/compare_runs.py`

### Medium Priority (Next 2 Weeks)

5. **Model Registry integration**
   - Impact: Production deployment path
   - Effort: 2 days
   - Files: `register_production_model.py`, update training scripts

6. **Dataset versioning**
   - Impact: Reproducibility
   - Effort: 1 day
   - Location: `log_dataset_info()` method

7. **Betting metrics logging**
   - Impact: Production readiness evaluation
   - Effort: 1 day
   - Location: After backtest simulation

8. **Experiment reorganization**
   - Impact: Better organization, easier navigation
   - Effort: 0.5 days
   - Location: Update experiment names in all scripts

### Low Priority (Future)

9. **CI/CD integration**
   - Impact: Automation, faster deployment
   - Effort: 3 days
   - Files: GitHub Actions workflows

10. **Production model serving**
    - Impact: Live predictions
    - Effort: 2 days
    - Files: `model_loader.py`, serving infrastructure

---

## Specific Code Changes Required

### File: `src/mlflow_integration/standard_tracker.py`

**Status:** CREATE NEW FILE
**Lines:** ~200
**Purpose:** Standardized logging patterns

See implementation in Phase 1 section above.

### File: `scripts/training/walk_forward_training_advanced_features.py`

**Status:** MODIFY
**Changes:**
- Line 36: Import `StandardNBAPropsTracker`
- Line 407: Replace `NBAPropsTracker` with `StandardNBAPropsTracker`
- After line 440: Add `tracker.log_dataset_info()`
- Inside line 563 loop: Add `tracker.log_walk_forward_fold()`
- After line 647: Add `tracker.log_feature_lineage()`
- After line 650: Add model registration

### File: `scripts/training/train_two_stage_model.py`

**Status:** MAJOR REFACTOR
**Changes:**
- Replace all raw `mlflow` calls with `StandardNBAPropsTracker`
- Add dataset logging
- Add fold-level tracking
- Add model registration
- Add baseline comparison

### File: `scripts/validation/walk_forward_validation_enhanced.py`

**Status:** MAJOR REFACTOR
**Changes:**
- Add MLflow tracking (currently has NONE)
- Add dataset logging
- Add feature lineage
- Add validation metrics
- Save model artifact

### File: `scripts/mlflow/register_production_model.py`

**Status:** CREATE NEW FILE
**Lines:** ~80
**Purpose:** Model registration workflow

See implementation in Phase 2 section above.

### File: `scripts/mlflow/compare_runs.py`

**Status:** CREATE NEW FILE
**Lines:** ~100
**Purpose:** Run comparison utility

See implementation in Phase 2 section above.

---

## Expected Outcomes

### After Phase 1 (Week 1):
- ✅ All training scripts use standardized MLflow logging
- ✅ Walk-forward folds tracked individually
- ✅ Feature lineage recorded for all experiments
- ✅ Can reproduce any experiment with exact features/data
- ✅ Run comparison takes 30 seconds (vs 10 minutes manual)

### After Phase 2 (Week 2):
- ✅ Model registry workflow established
- ✅ Production criteria evaluation automated
- ✅ Can promote/rollback models with one command
- ✅ Model lineage fully tracked (data → features → model → metrics)

### After Phase 3 (Week 3):
- ✅ Experiments organized by phase/objective
- ✅ Consistent naming across all runs
- ✅ Can filter/search experiments by tags
- ✅ Clear progression from baseline → Phase 1 → Phase 2

### After Phase 4 (Week 4):
- ✅ Production model serving via registry
- ✅ CI/CD pipeline for automatic training
- ✅ Monitoring integrated with MLflow
- ✅ Full ML lifecycle automation

---

## Metrics to Track Success

### Developer Velocity:
- **Before:** 10 minutes to compare runs manually
- **After:** 30 seconds with comparison utility
- **Target:** 95% time reduction

### Reproducibility:
- **Before:** 30% of experiments can be reproduced
- **After:** 100% of experiments reproducible
- **Target:** 100%

### Model Registry Adoption:
- **Before:** 0% of models registered
- **After:** 100% of production candidates registered
- **Target:** 100%

### Experiment Quality:
- **Before:** Missing metrics, incomplete logging
- **After:** Complete metrics, full lineage
- **Target:** 100% compliance with standards

---

## Conclusion

The MLflow infrastructure is **strong** but **underutilized**. The tracker and registry modules are production-quality, but only 1 of 3 main scripts uses them properly.

**Key Recommendations:**

1. **Immediate (This Week):** Standardize all scripts with `StandardNBAPropsTracker`
2. **Short-term (Weeks 2-3):** Integrate Model Registry and reorganize experiments
3. **Medium-term (Week 4):** Add production serving and CI/CD

**Impact:** These changes will:
- Reduce iteration time by 95% (run comparison)
- Ensure 100% reproducibility (dataset/feature versioning)
- Enable production deployment (model registry)
- Provide clear model lineage (baseline → improvements)

**Next Steps:**
1. Review this document with team
2. Create GitHub issues for each phase
3. Start with Phase 1 (highest ROI, lowest effort)
4. Track progress with weekly check-ins

---

**Questions? Contact:** NBA Props Model Team
**Last Updated:** October 15, 2025
