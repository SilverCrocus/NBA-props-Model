"""
Week 1 Day 5: Feature Selection and Hyperparameter Optimization

Based on Day 4 results showing minimal improvement from 47 features,
this script:
1. Removes low-importance features (importance < 0.01)
2. Trains with pruned feature set
3. Runs hyperparameter grid search
4. Compares results to Day 3 and Day 4

Expected Impact: MAE 6.10 → 5.5-5.8 points

Author: NBA Props Model - Week 1 Day 5
Date: October 14, 2025
"""

import logging
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent.parent / "utils"))
from ctg_feature_builder import CTGFeatureBuilder

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Initialize MLflow tracking
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.mlflow_integration.tracker import NBAPropsTracker, enable_autologging


# Load feature importance from Day 4
FEATURE_IMPORTANCE_PATH = "feature_importance_day4.csv"

# Selected features (importance > 0.01 from Day 4)
# Top features identified from analysis:
SELECTED_FEATURES = [
    "PRA_ewma10",      # 47.4%
    "PRA_L20_mean",    # 18.4%
    "PRA_ewma5",       # 8.7%
    "PRA_L5_mean",     # 2.3%
    "PRA_L10_mean",    # 2.0%
]

# Additional features with >0.5% importance
ADDITIONAL_FEATURES = [
    "CTG_USG", "MIN", "MIN_avg", "PRA_per_36", "CTG_PSA",
    "days_rest", "eFG_pct", "MIN_L5_mean", "MIN_lag1", "CTG_eFG",
    "FGA", "TS_pct", "CTG_AST_PCT", "PTS_per_36", "games_last_7d",
    "PRA_L5_std", "PRA_L10_std", "is_b2b", "CTG_TOV_PCT", "FG_PCT"
]

ALL_SELECTED_FEATURES = SELECTED_FEATURES + ADDITIONAL_FEATURES


# Import feature calculation functions from Day 4 script
def calculate_lag_features(player_history: pd.DataFrame, lags=[1, 3, 5, 7]) -> dict:
    """Calculate lag features using ONLY historical games."""
    features = {}
    if len(player_history) == 0:
        for lag in lags:
            features[f"PRA_lag{lag}"] = 0
            features[f"MIN_lag{lag}"] = 0
        return features

    history = player_history.sort_values("GAME_DATE", ascending=False)
    for lag in lags:
        if len(history) >= lag:
            features[f"PRA_lag{lag}"] = history.iloc[lag - 1]["PRA"]
            features[f"MIN_lag{lag}"] = history.iloc[lag - 1].get("MIN", 0)
        else:
            features[f"PRA_lag{lag}"] = 0
            features[f"MIN_lag{lag}"] = 0
    return features


def calculate_rolling_features(player_history: pd.DataFrame, windows=[5, 10, 20]) -> dict:
    """Calculate rolling average features."""
    features = {}
    if len(player_history) == 0:
        for window in windows:
            features[f"PRA_L{window}_mean"] = 0
            features[f"PRA_L{window}_std"] = 0
            features[f"MIN_L{window}_mean"] = 0
        return features

    history = player_history.sort_values("GAME_DATE", ascending=False)
    for window in windows:
        if len(history) >= window:
            recent_games = history.iloc[:window]
            features[f"PRA_L{window}_mean"] = recent_games["PRA"].mean()
            features[f"PRA_L{window}_std"] = recent_games["PRA"].std()
            features[f"MIN_L{window}_mean"] = recent_games.get("MIN", pd.Series([0])).mean()
        elif len(history) >= 3:
            features[f"PRA_L{window}_mean"] = history["PRA"].mean()
            features[f"PRA_L{window}_std"] = history["PRA"].std()
            features[f"MIN_L{window}_mean"] = history.get("MIN", pd.Series([0])).mean()
        else:
            features[f"PRA_L{window}_mean"] = 0
            features[f"PRA_L{window}_std"] = 0
            features[f"MIN_L{window}_mean"] = 0
    return features


def calculate_ewma_features(player_history: pd.DataFrame, spans=[5, 10]) -> dict:
    """Calculate EWMA features."""
    features = {}
    if len(player_history) < 3:
        for span in spans:
            features[f"PRA_ewma{span}"] = 0
        return features

    history = player_history.sort_values("GAME_DATE", ascending=True)
    for span in spans:
        ewma_value = history["PRA"].ewm(span=span, min_periods=1).mean().iloc[-1]
        features[f"PRA_ewma{span}"] = ewma_value
    return features


def calculate_rest_features(player_history: pd.DataFrame, current_date: pd.Timestamp) -> dict:
    """Calculate rest and schedule fatigue features."""
    features = {}
    if len(player_history) == 0:
        features["days_rest"] = 7
        features["is_b2b"] = 0
        features["games_last_7d"] = 0
        return features

    last_game = player_history.sort_values("GAME_DATE", ascending=False).iloc[0]
    last_game_date = last_game["GAME_DATE"]

    days_rest = (current_date - last_game_date).days
    features["days_rest"] = min(days_rest, 7)
    features["is_b2b"] = 1 if days_rest <= 1 else 0

    week_ago = current_date - timedelta(days=7)
    recent_games = player_history[player_history["GAME_DATE"] >= week_ago]
    features["games_last_7d"] = len(recent_games)
    return features


def calculate_efficiency_features(player_history: pd.DataFrame) -> dict:
    """Calculate efficiency features."""
    features = {}
    if len(player_history) < 5:
        features["TS_pct"] = 0
        features["eFG_pct"] = 0
        features["PTS_per_36"] = 0
        return features

    history = player_history.sort_values("GAME_DATE", ascending=False).iloc[:10]

    pts_total = history["PTS"].sum()
    fga_total = history.get("FGA", pd.Series([0])).sum()
    fta_total = history.get("FTA", pd.Series([0])).sum()

    ts_denominator = 2 * (fga_total + 0.44 * fta_total)
    features["TS_pct"] = pts_total / ts_denominator if ts_denominator > 0 else 0

    fgm_total = history.get("FGM", pd.Series([0])).sum()
    fg3m_total = history.get("FG3M", pd.Series([0])).sum()
    features["eFG_pct"] = (
        (fgm_total + 0.5 * fg3m_total) / fga_total if fga_total > 0 else 0
    )

    min_total = history.get("MIN", pd.Series([0])).sum()
    features["PTS_per_36"] = (pts_total / min_total * 36) if min_total > 0 else 0

    return features


def calculate_normalization_features(player_history: pd.DataFrame) -> dict:
    """Calculate per-36 stats."""
    features = {}
    if len(player_history) < 5:
        features["PRA_per_36"] = 0
        features["MIN_avg"] = 0
        return features

    history = player_history.sort_values("GAME_DATE", ascending=False).iloc[:10]
    min_total = history.get("MIN", pd.Series([0])).sum()

    if min_total == 0:
        features["PRA_per_36"] = 0
        features["MIN_avg"] = 0
        return features

    pra_total = history["PRA"].sum()
    features["PRA_per_36"] = (pra_total / min_total) * 36
    features["MIN_avg"] = min_total / len(history)
    return features


def calculate_all_features(
    player_history: pd.DataFrame,
    current_date: pd.Timestamp,
    player_name: str,
    season: str,
    ctg_builder: CTGFeatureBuilder,
    current_game_stats: dict = None,
) -> dict:
    """Calculate ALL features (we'll filter to selected ones later)."""
    features = {}

    # Calculate all feature types
    features.update(calculate_lag_features(player_history))
    features.update(calculate_rolling_features(player_history))
    features.update(calculate_ewma_features(player_history))
    features.update(calculate_rest_features(player_history, current_date))
    features.update(calculate_efficiency_features(player_history))
    features.update(calculate_normalization_features(player_history))

    # CTG features
    ctg_feats = ctg_builder.get_player_ctg_features(player_name, season)
    features.update(ctg_feats)

    # Current game stats
    if current_game_stats is None and len(player_history) > 0:
        last_game = player_history.sort_values("GAME_DATE", ascending=False).iloc[0]
        current_game_stats = {
            "MIN": last_game.get("MIN", 0),
            "FGA": last_game.get("FGA", 0),
            "FG_PCT": last_game.get("FG_PCT", 0),
            "FG3A": last_game.get("FG3A", 0),
            "FTA": last_game.get("FTA", 0),
        }

    if current_game_stats:
        features.update(current_game_stats)

    return features


def train_with_selected_features(
    train_season: str = "2023-24",
    val_season: str = "2024-25",
    feature_set: list = None,
    hyperparams: dict = None,
    run_name_suffix: str = "",
):
    """
    Train model with selected features and optional hyperparameters.
    """
    if feature_set is None:
        feature_set = ALL_SELECTED_FEATURES

    if hyperparams is None:
        hyperparams = {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "n_jobs": -1,
        }

    logger.info("=" * 80)
    logger.info(f"TRAINING WITH {len(feature_set)} SELECTED FEATURES")
    logger.info("=" * 80)

    # Initialize MLflow
    tracker = NBAPropsTracker(experiment_name="Phase1_Foundation")
    tracker.start_run(
        run_name=f"day5_optimized_{val_season}{run_name_suffix}",
        tags={
            "model_type": "xgboost",
            "validation_type": "walk_forward_feature_selected",
            "train_season": train_season,
            "val_season": val_season,
            "n_features": len(feature_set),
            "description": f"Day 5: Feature selection ({len(feature_set)} features)",
        },
    )

    try:
        # Initialize CTG
        logger.info("\n1. Initializing CTG feature builder...")
        ctg_builder = CTGFeatureBuilder()

        # Load data
        logger.info("\n2. Loading game logs...")
        game_logs_path = "data/game_logs/all_game_logs_with_opponent.csv"
        all_games_df = pd.read_csv(game_logs_path)
        all_games_df["GAME_DATE"] = pd.to_datetime(all_games_df["GAME_DATE"], format="mixed")
        all_games_df = all_games_df.sort_values("GAME_DATE").reset_index(drop=True)

        logger.info(f"   Loaded {len(all_games_df):,} games")

        # Split data
        train_start_date = pd.to_datetime("2023-10-01")
        train_end_date = pd.to_datetime("2024-06-30")
        val_start_date = pd.to_datetime("2024-10-01")
        val_end_date = all_games_df["GAME_DATE"].max()

        train_games = all_games_df[
            (all_games_df["GAME_DATE"] >= train_start_date)
            & (all_games_df["GAME_DATE"] <= train_end_date)
        ].copy()

        val_games = all_games_df[
            (all_games_df["GAME_DATE"] >= val_start_date)
            & (all_games_df["GAME_DATE"] <= val_end_date)
        ].copy()

        logger.info(f"\n3. Data split:")
        logger.info(f"   Training: {len(train_games):,} games")
        logger.info(f"   Validation: {len(val_games):,} games")

        # Build training dataset
        logger.info(f"\n4. Building training dataset with {len(feature_set)} selected features...")

        training_samples = []
        train_dates = sorted(train_games["GAME_DATE"].unique())

        for pred_date in tqdm(train_dates, desc="Building training data"):
            games_today = train_games[train_games["GAME_DATE"] == pred_date]
            past_games = train_games[train_games["GAME_DATE"] < pred_date]

            for _, row in games_today.iterrows():
                player_id = row["PLAYER_ID"]
                player_name = row.get("PLAYER_NAME", "")

                player_history = past_games[past_games["PLAYER_ID"] == player_id]

                if len(player_history) < 10:
                    continue

                # Calculate ALL features
                all_features = calculate_all_features(
                    player_history, pred_date, player_name, train_season, ctg_builder
                )

                # Filter to selected features only
                features = {k: all_features.get(k, 0) for k in feature_set}
                features["PRA"] = row["PRA"]
                features["GAME_DATE"] = pred_date
                features["PLAYER_ID"] = player_id

                training_samples.append(features)

        train_df = pd.DataFrame(training_samples)
        logger.info(f"\n✅ Training dataset built: {len(train_df):,} samples")

        # Prepare features
        X_train = train_df[feature_set].fillna(0)
        y_train = train_df["PRA"]

        logger.info(f"   Features: {len(feature_set)}")
        logger.info(f"   Samples: {len(X_train):,}")

        # Log config
        training_config = {
            "n_samples": len(X_train),
            "n_features": len(feature_set),
            "n_features_day4": 47,
            "features_removed": 47 - len(feature_set),
            "feature_set": feature_set[:10],  # Log first 10
        }
        tracker.log_training_config(training_config)
        tracker.log_params(hyperparams)

        # Train model
        logger.info("\n5. Training XGBoost model...")
        enable_autologging("xgboost")

        model = xgb.XGBRegressor(**hyperparams)
        model.fit(X_train, y_train, verbose=False)

        train_pred = model.predict(X_train)
        train_mae = mean_absolute_error(y_train, train_pred)

        logger.info(f"✅ Model trained - Train MAE: {train_mae:.2f}")

        # Walk-forward validation
        logger.info("\n6. Running walk-forward validation on 2024-25...")

        val_predictions = []
        val_dates = sorted(val_games["GAME_DATE"].unique())

        for pred_date in tqdm(val_dates, desc="Walk-forward validation"):
            games_today = val_games[val_games["GAME_DATE"] == pred_date]
            past_games = all_games_df[all_games_df["GAME_DATE"] < pred_date]

            for _, row in games_today.iterrows():
                player_id = row["PLAYER_ID"]
                player_name = row.get("PLAYER_NAME", "")

                player_history = past_games[past_games["PLAYER_ID"] == player_id]

                if len(player_history) < 5:
                    continue

                # Calculate features
                all_features = calculate_all_features(
                    player_history, pred_date, player_name, val_season, ctg_builder
                )

                # Filter to selected features
                feature_vector = [all_features.get(col, 0) for col in feature_set]
                pred_pra = model.predict([feature_vector])[0]

                val_predictions.append(
                    {
                        "PLAYER_NAME": player_name,
                        "PLAYER_ID": player_id,
                        "GAME_DATE": pred_date,
                        "PRA": row["PRA"],
                        "predicted_PRA": pred_pra,
                        "error": pred_pra - row["PRA"],
                        "abs_error": abs(pred_pra - row["PRA"]),
                    }
                )

        val_df = pd.DataFrame(val_predictions)
        logger.info(f"\n✅ Validation complete: {len(val_df):,} predictions")

        # Calculate metrics
        val_mae = mean_absolute_error(val_df["PRA"], val_df["predicted_PRA"])
        val_rmse = np.sqrt(mean_squared_error(val_df["PRA"], val_df["predicted_PRA"]))
        val_r2 = r2_score(val_df["PRA"], val_df["predicted_PRA"])

        val_metrics = {
            "mae": val_mae,
            "rmse": val_rmse,
            "r2": val_r2,
            "within_3pts_pct": (val_df["abs_error"] <= 3).mean() * 100,
            "within_5pts_pct": (val_df["abs_error"] <= 5).mean() * 100,
            "within_10pts_pct": (val_df["abs_error"] <= 10).mean() * 100,
            "improvement_from_day4": 6.10 - val_mae,
        }
        tracker.log_validation_metrics(val_metrics)

        # Log results
        logger.info("\n" + "=" * 80)
        logger.info(f"RESULTS - {len(feature_set)} SELECTED FEATURES")
        logger.info("=" * 80)
        logger.info(f"\nValidation Metrics:")
        logger.info(f"  MAE: {val_mae:.2f} points (Day 4: 6.10)")
        logger.info(f"  Improvement: {6.10 - val_mae:+.2f} points")
        logger.info(f"  RMSE: {val_rmse:.2f} points")
        logger.info(f"  R²: {val_r2:.3f}")
        logger.info(f"  Within ±3 pts: {val_metrics['within_3pts_pct']:.1f}%")
        logger.info(f"  Within ±5 pts: {val_metrics['within_5pts_pct']:.1f}%")
        logger.info(f"  Within ±10 pts: {val_metrics['within_10pts_pct']:.1f}%")

        # Log feature importance
        importance = model.feature_importances_
        tracker.log_feature_importance(feature_set, importance)

        # Log model
        tracker.log_model(model, model_type="xgboost")

        tracker.end_run(status="FINISHED")

        return {
            "model": model,
            "features": feature_set,
            "val_df": val_df,
            "metrics": val_metrics,
            "run_id": tracker.run_id,
        }

    except Exception as e:
        logger.error(f"Error: {e}")
        tracker.end_run(status="FAILED")
        raise


if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("WEEK 1 DAY 5: FEATURE SELECTION & OPTIMIZATION")
    logger.info("=" * 80)

    # Step 1: Train with top 25 features (importance > 0.005)
    logger.info("\n>>> STEP 1: Training with top 25 selected features")
    results = train_with_selected_features(
        feature_set=ALL_SELECTED_FEATURES,
        run_name_suffix="_top25",
    )

    logger.info("\n" + "=" * 80)
    logger.info("DAY 5 STEP 1 COMPLETE")
    logger.info("=" * 80)
    logger.info(f"\nRun ID: {results['run_id']}")
    logger.info(f"MAE: {results['metrics']['mae']:.2f} (Day 4: 6.10)")
    logger.info(f"Improvement: {results['metrics']['improvement_from_day4']:+.2f} points")
    logger.info("\nView in MLflow: uv run mlflow ui")
