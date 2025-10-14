"""
Baseline Experiment - Phase 1 Foundation
Creates first MLflow tracked experiment with existing training data
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.mlflow_integration.tracker import NBAPropsTracker, enable_autologging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(data_path: str = "data/processed/game_level_training_data.parquet"):
    """Load existing training data"""
    logger.info(f"Loading data from {data_path}")
    df = pd.read_parquet(data_path)
    logger.info(f"Loaded {len(df)} rows with {len(df.columns)} columns")
    return df


def prepare_features(df: pd.DataFrame):
    """Prepare features and target from existing dataset"""

    # Identify feature columns (exclude metadata and target)
    exclude_cols = [
        "PLAYER_NAME",
        "PLAYER_ID",
        "GAME_ID",
        "GAME_DATE",
        "SEASON",
        "TEAM_NAME",
        "TEAM_ID",
        "OPP_TEAM",
        "PRA",
        "PTS",
        "REB",
        "AST",  # Target and its components
    ]

    # Get all numeric columns that aren't in exclude list
    feature_cols = [
        col
        for col in df.columns
        if col not in exclude_cols and df[col].dtype in ["int64", "float64"]
    ]

    logger.info(f"Selected {len(feature_cols)} feature columns")

    # Create target (PRA)
    if "PRA" not in df.columns:
        if all(col in df.columns for col in ["PTS", "REB", "AST"]):
            df["PRA"] = df["PTS"] + df["REB"] + df["AST"]
            logger.info("Created PRA target from PTS + REB + AST")
        else:
            raise ValueError("Cannot create PRA target - missing PTS/REB/AST columns")

    # Remove rows with NaN in features or target
    valid_mask = ~(df[feature_cols].isna().any(axis=1) | df["PRA"].isna())
    df_clean = df[valid_mask].copy()

    logger.info(f"After removing NaN: {len(df_clean)} samples remain")

    return df_clean, feature_cols


def train_baseline_model():
    """Train baseline XGBoost model with MLflow tracking"""

    # Initialize MLflow tracker
    tracker = NBAPropsTracker(experiment_name="Phase1_Foundation")

    # Start run
    tracker.start_run(
        run_name="baseline_xgb_v1",
        tags={
            "model_type": "xgboost",
            "model_version": "baseline_v1.0",
            "description": "First baseline model using all available features",
            "phase": "foundation",
        },
    )

    try:
        # Load and prepare data
        df = load_data()
        df_clean, feature_cols = prepare_features(df)

        X = df_clean[feature_cols]
        y = df_clean["PRA"]

        # Log training config
        training_config = {
            "n_samples": len(X),
            "n_features": len(feature_cols),
            "data_source": "game_level_training_data.parquet",
            "target": "PRA",
            "validation_method": "time_series_cv",
        }
        tracker.log_training_config(training_config)

        # Log feature config
        feature_config = {
            "feature_set_version": "baseline_v1.0",
            "n_features": len(feature_cols),
            "feature_names": feature_cols[:50],  # Log first 50 to avoid clutter
        }
        tracker.log_feature_config(feature_config)

        # Baseline hyperparameters
        hyperparams = {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "objective": "reg:squarederror",
        }
        tracker.log_params(hyperparams)

        # Enable autologging
        enable_autologging("xgboost")

        # Time series cross-validation
        logger.info("Starting time series cross-validation...")
        tscv = TimeSeriesSplit(n_splits=5)

        cv_maes = []
        cv_rmses = []
        cv_r2s = []

        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
            logger.info(f"Training fold {fold_idx + 1}/5")

            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Train model
            model = xgb.XGBRegressor(**hyperparams)
            model.fit(X_train, y_train, verbose=False)

            # Predict
            y_pred = model.predict(X_val)

            # Calculate metrics
            mae = mean_absolute_error(y_val, y_pred)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            r2 = r2_score(y_val, y_pred)

            cv_maes.append(mae)
            cv_rmses.append(rmse)
            cv_r2s.append(r2)

            logger.info(f"  Fold {fold_idx + 1}: MAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2:.3f}")

        # Log validation metrics
        val_metrics = {
            "mae": np.mean(cv_maes),
            "mae_std": np.std(cv_maes),
            "rmse": np.mean(cv_rmses),
            "rmse_std": np.std(cv_rmses),
            "r2": np.mean(cv_r2s),
            "r2_std": np.std(cv_r2s),
        }
        tracker.log_validation_metrics(val_metrics)

        logger.info(f"\nCross-Validation Results:")
        logger.info(f"  MAE: {val_metrics['mae']:.2f} ± {val_metrics['mae_std']:.2f}")
        logger.info(f"  RMSE: {val_metrics['rmse']:.2f} ± {val_metrics['rmse_std']:.2f}")
        logger.info(f"  R²: {val_metrics['r2']:.3f} ± {val_metrics['r2_std']:.3f}")

        # Train final model on all data
        logger.info("\nTraining final model on full dataset...")
        final_model = xgb.XGBRegressor(**hyperparams)
        final_model.fit(X, y, verbose=False)

        # Get predictions
        y_pred_train = final_model.predict(X)

        # Calculate final metrics
        final_metrics = {
            "train_mae": mean_absolute_error(y, y_pred_train),
            "train_rmse": np.sqrt(mean_squared_error(y, y_pred_train)),
            "train_r2": r2_score(y, y_pred_train),
        }
        tracker.log_training_metrics(final_metrics)

        logger.info(f"\nFinal Model Performance:")
        logger.info(f"  Train MAE: {final_metrics['train_mae']:.2f}")
        logger.info(f"  Train RMSE: {final_metrics['train_rmse']:.2f}")
        logger.info(f"  Train R²: {final_metrics['train_r2']:.3f}")

        # Log feature importance
        importance = final_model.feature_importances_
        tracker.log_feature_importance(feature_cols, importance)

        # Log model
        tracker.log_model(
            final_model, model_type="xgboost", registered_model_name="NBAPropsModel_Baseline"
        )

        # End run
        tracker.end_run(status="FINISHED")

        logger.info(f"\n✅ Baseline experiment completed!")
        logger.info(f"Run ID: {tracker.run_id}")
        logger.info(f"\nTo view results:")
        logger.info(f"  uv run mlflow ui")
        logger.info(f"  Navigate to http://localhost:5000")

        return {
            "model": final_model,
            "features": feature_cols,
            "cv_metrics": val_metrics,
            "final_metrics": final_metrics,
            "run_id": tracker.run_id,
        }

    except Exception as e:
        logger.error(f"Training failed: {e}")
        tracker.end_run(status="FAILED")
        raise


if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("BASELINE EXPERIMENT - PHASE 1 FOUNDATION")
    logger.info("=" * 80)

    results = train_baseline_model()

    logger.info("\n" + "=" * 80)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("=" * 80)
