"""
Walk-Forward Training & Validation - Advanced Features (Day 4)

Builds on Day 3's leak-free implementation with:
1. Opponent features: Defensive rating, pace, PRA allowed
2. Efficiency features: TS%, PER, usage per 36
3. Normalization features: Per-36 and per-100 stats

Expected Impact: MAE 6.11 → 5.2-5.5 points

Author: NBA Props Model - Week 1 Day 4
Date: October 14, 2025
"""

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent.parent / "utils"))
from ctg_feature_builder import CTGFeatureBuilder

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Initialize MLflow tracking
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import data_config, model_config, validation_config
from src.mlflow_integration.tracker import NBAPropsTracker, enable_autologging

# ==================== BASIC FEATURES (from Day 3) ====================


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


def calculate_trend_features(player_history: pd.DataFrame) -> dict:
    """Calculate trend features."""
    features = {}

    if len(player_history) < 10:
        features["PRA_trend"] = 0
        return features

    history = player_history.sort_values("GAME_DATE", ascending=False)

    if len(history) >= 15:
        l5_mean = history.iloc[:5]["PRA"].mean()
        l10_mean = history.iloc[5:15]["PRA"].mean()
        features["PRA_trend"] = l5_mean - l10_mean
    else:
        features["PRA_trend"] = 0

    return features


# ==================== NEW: EFFICIENCY FEATURES (Day 4) ====================


def calculate_efficiency_features(player_history: pd.DataFrame) -> dict:
    """
    Calculate efficiency features: TS%, PER, usage per 36, points per shot.

    Args:
        player_history: Player's games before current date

    Returns:
        Dictionary of efficiency features
    """
    features = {}

    if len(player_history) < 5:
        features["TS_pct"] = 0
        features["PER"] = 0
        features["USG_per_36"] = 0
        features["PTS_per_shot"] = 0
        features["eFG_pct"] = 0
        return features

    history = player_history.sort_values("GAME_DATE", ascending=False).iloc[:10]

    # True Shooting % = PTS / (2 * (FGA + 0.44 * FTA))
    pts_total = history["PTS"].sum()
    fga_total = history.get("FGA", pd.Series([0])).sum()
    fta_total = history.get("FTA", pd.Series([0])).sum()

    ts_denominator = 2 * (fga_total + 0.44 * fta_total)
    features["TS_pct"] = pts_total / ts_denominator if ts_denominator > 0 else 0

    # Effective FG% = (FGM + 0.5 * FG3M) / FGA
    fgm_total = history.get("FGM", pd.Series([0])).sum()
    fg3m_total = history.get("FG3M", pd.Series([0])).sum()
    features["eFG_pct"] = (fgm_total + 0.5 * fg3m_total) / fga_total if fga_total > 0 else 0

    # Points per shot attempt
    features["PTS_per_shot"] = pts_total / fga_total if fga_total > 0 else 0

    # Usage per 36 minutes (approximation: FGA + 0.44*FTA + TOV per 36)
    min_total = history.get("MIN", pd.Series([0])).sum()
    tov_total = history.get("TOV", pd.Series([0])).sum()
    usage_total = fga_total + 0.44 * fta_total + tov_total

    features["USG_per_36"] = (usage_total / min_total * 36) if min_total > 0 else 0

    # Simplified PER (approximate): (PTS + REB + AST + STL + BLK - TOV - (FGA-FGM) - (FTA-FTM)) / MIN * factor
    reb_total = history.get("REB", pd.Series([0])).sum()
    ast_total = history.get("AST", pd.Series([0])).sum()
    stl_total = history.get("STL", pd.Series([0])).sum()
    blk_total = history.get("BLK", pd.Series([0])).sum()
    ftm_total = history.get("FTM", pd.Series([0])).sum()

    per_numerator = (
        pts_total
        + reb_total
        + ast_total
        + stl_total
        + blk_total
        - tov_total
        - (fga_total - fgm_total)
        - (fta_total - ftm_total)
    )
    features["PER"] = (per_numerator / min_total * 15) if min_total > 0 else 0

    return features


# ==================== NEW: NORMALIZATION FEATURES (Day 4) ====================


def calculate_normalization_features(player_history: pd.DataFrame) -> dict:
    """
    Calculate per-36 and per-100 possession stats.

    Args:
        player_history: Player's games before current date

    Returns:
        Dictionary of normalized features
    """
    features = {}

    if len(player_history) < 5:
        features["PRA_per_36"] = 0
        features["PTS_per_36"] = 0
        features["REB_per_36"] = 0
        features["AST_per_36"] = 0
        features["MIN_avg"] = 0
        return features

    history = player_history.sort_values("GAME_DATE", ascending=False).iloc[:10]

    min_total = history.get("MIN", pd.Series([0])).sum()

    if min_total == 0:
        features["PRA_per_36"] = 0
        features["PTS_per_36"] = 0
        features["REB_per_36"] = 0
        features["AST_per_36"] = 0
        features["MIN_avg"] = 0
        return features

    # Per-36 stats
    pra_total = history["PRA"].sum()
    pts_total = history["PTS"].sum()
    reb_total = history.get("REB", pd.Series([0])).sum()
    ast_total = history.get("AST", pd.Series([0])).sum()

    features["PRA_per_36"] = (pra_total / min_total) * 36
    features["PTS_per_36"] = (pts_total / min_total) * 36
    features["REB_per_36"] = (reb_total / min_total) * 36
    features["AST_per_36"] = (ast_total / min_total) * 36

    # Average minutes played
    features["MIN_avg"] = min_total / len(history)

    return features


# ==================== NEW: OPPONENT FEATURES (Day 4) ====================


def calculate_opponent_features(
    opponent_team: str, all_games: pd.DataFrame, current_date: pd.Timestamp
) -> dict:
    """
    Calculate opponent DEFENSIVE features using games before current date.

    CRITICAL FIX: Calculate PRA ALLOWED by opponent (defensive stats),
    not PRA SCORED by opponent (offensive stats).

    Args:
        opponent_team: Name of opponent team
        all_games: All games data (for calculating opponent stats)
        current_date: Current game date

    Returns:
        Dictionary of opponent features
    """
    features = {}

    # Get games BEFORE current date
    past_games = all_games[all_games["GAME_DATE"] < current_date]

    # FIXED: Get games where OTHER teams played AGAINST this opponent
    # This tells us how much PRA the opponent ALLOWS (their defense)
    opponent_defense_games = past_games[past_games["OPP_TEAM"] == opponent_team]

    if len(opponent_defense_games) < 5:
        features["opp_DRtg"] = 110.0  # League average
        features["opp_pace"] = 100.0  # League average
        features["opp_PRA_allowed"] = 30.0  # League average
        return features

    # Use last 20 games for better sample (opponent plays ~2-3x per week)
    recent_def = opponent_defense_games.sort_values("GAME_DATE", ascending=False).iloc[:20]

    # FIXED: PRA allowed = average PRA scored BY OPPONENTS against this team
    features["opp_PRA_allowed"] = recent_def["PRA"].mean()

    # Defensive Rating: Higher PRA allowed = worse defense = higher DRtg
    # Scale: 100 (elite defense) to 120 (poor defense)
    features["opp_DRtg"] = 95.0 + (features["opp_PRA_allowed"] - 30.0) * 0.5

    # Pace calculation: Get opponent's own games for pace
    opp_games = past_games[past_games["TEAM_NAME"] == opponent_team]
    if len(opp_games) >= 5:
        recent_opp = opp_games.sort_values("GAME_DATE", ascending=False).iloc[:10]
        # Pace proxy: higher PRA output = faster pace
        features["opp_pace"] = 90.0 + (recent_opp["PRA"].mean() - 30.0) * 0.3
    else:
        features["opp_pace"] = 100.0

    return features


# ==================== MASTER FEATURE CALCULATION ====================


def calculate_all_features(
    player_history: pd.DataFrame,
    current_date: pd.Timestamp,
    player_name: str,
    opponent_team: str,
    season: str,
    ctg_builder: CTGFeatureBuilder,
    all_games: pd.DataFrame,
) -> dict:
    """
    Calculate ALL features including new Day 4 features.

    Args:
        player_history: Player's games before current date
        current_date: Current game date
        player_name: Player name for CTG lookup
        opponent_team: Opponent team name
        season: Current season
        ctg_builder: CTG feature builder
        all_games: All games data (for opponent features)

    Returns:
        Dictionary of all features
    """
    features = {}

    # Day 3 features (basic)
    features.update(calculate_lag_features(player_history))
    features.update(calculate_rolling_features(player_history))
    features.update(calculate_ewma_features(player_history))
    features.update(calculate_rest_features(player_history, current_date))
    features.update(calculate_trend_features(player_history))

    # Day 4 NEW features
    features.update(calculate_efficiency_features(player_history))
    features.update(calculate_normalization_features(player_history))
    features.update(calculate_opponent_features(opponent_team, all_games, current_date))

    # CTG season stats
    ctg_feats = ctg_builder.get_player_ctg_features(player_name, season)
    features.update(ctg_feats)

    # REMOVED: current_game_stats section (redundant with lag features)
    # - MIN duplicates MIN_lag1
    # - FGA, FG_PCT, FG3A, FTA duplicate lag/rolling features
    # - These stats aren't available in production (can't know current game stats before it happens)
    # - Model should rely on historical lag/rolling features instead

    return features


# ==================== MAIN TRAINING FUNCTION ====================


def walk_forward_train_and_validate(
    train_season: str = "2023-24",
    val_season: str = "2024-25",
    min_train_games: int = 10,
    min_history_for_prediction: int = 5,
):
    """
    Perform walk-forward training with advanced features (Day 4).
    """
    logger.info("=" * 80)
    logger.info("WALK-FORWARD TRAINING - ADVANCED FEATURES (DAY 4)")
    logger.info("=" * 80)

    # Initialize MLflow
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

    try:
        # Initialize CTG feature builder
        logger.info("\n1. Initializing CTG feature builder...")
        ctg_builder = CTGFeatureBuilder()

        # Load raw game logs
        logger.info("\n2. Loading raw game logs...")

        # Load combined game logs with opponent data
        game_logs_path = data_config.GAME_LOGS_PATH

        if not game_logs_path.exists():
            raise FileNotFoundError(f"Game logs not found: {game_logs_path}")

        all_games_df = pd.read_csv(game_logs_path)
        all_games_df["GAME_DATE"] = pd.to_datetime(all_games_df["GAME_DATE"], format="mixed")

        logger.info(f"   Loaded {game_logs_path}: {len(all_games_df):,} games")
        all_games_df = all_games_df.sort_values("GAME_DATE").reset_index(drop=True)

        logger.info(f"\n✅ Total games loaded: {len(all_games_df):,}")
        logger.info(
            f"   Date range: {all_games_df['GAME_DATE'].min()} to {all_games_df['GAME_DATE'].max()}"
        )

        # Split into training and validation
        train_start_date = pd.to_datetime(validation_config.TRAIN_START_DATE)
        train_end_date = pd.to_datetime(validation_config.TRAIN_END_DATE)
        val_start_date = pd.to_datetime(validation_config.VAL_START_DATE)
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
        logger.info(
            f"   Training: {len(train_games):,} games ({train_games['GAME_DATE'].min()} to {train_games['GAME_DATE'].max()})"
        )
        logger.info(
            f"   Validation: {len(val_games):,} games ({val_games['GAME_DATE'].min()} to {val_games['GAME_DATE'].max()})"
        )

        # Build training dataset
        logger.info("\n4. Building training dataset with ADVANCED features...")

        training_samples = []
        train_dates = sorted(train_games["GAME_DATE"].unique())

        logger.info(f"   Processing {len(train_dates)} training dates...")

        for pred_date in tqdm(train_dates, desc="Building training data"):
            games_today = train_games[train_games["GAME_DATE"] == pred_date]
            past_games = train_games[train_games["GAME_DATE"] < pred_date]

            for _, row in games_today.iterrows():
                player_id = row["PLAYER_ID"]
                player_name = row.get("PLAYER_NAME", "")
                opponent_team = row.get("OPP_TEAM", "")

                player_history = past_games[past_games["PLAYER_ID"] == player_id]

                if len(player_history) < min_train_games:
                    continue

                # Calculate features (NOW with opponent features!)
                features = calculate_all_features(
                    player_history,
                    pred_date,
                    player_name,
                    opponent_team,
                    train_season,
                    ctg_builder,
                    all_games_df,
                )

                features["PRA"] = row["PRA"]
                features["GAME_DATE"] = pred_date
                features["PLAYER_ID"] = player_id

                training_samples.append(features)

        train_df = pd.DataFrame(training_samples)
        logger.info(f"\n✅ Training dataset built: {len(train_df):,} samples")

        # Define feature columns
        exclude_cols = ["PRA", "GAME_DATE", "PLAYER_ID", "CTG_Available"]
        feature_cols = [col for col in train_df.columns if col not in exclude_cols]

        X_train = train_df[feature_cols].fillna(0)
        y_train = train_df["PRA"]

        logger.info(f"   Features: {len(feature_cols)} (was 34 in Day 3)")
        logger.info(f"   Samples: {len(X_train):,}")

        # Log training config
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

        # Train model
        logger.info("\n5. Training XGBoost model...")

        hyperparams = model_config.XGBOOST_PARAMS.copy()
        tracker.log_params(hyperparams)

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
                opponent_team = row.get("OPP_TEAM", "")

                player_history = past_games[past_games["PLAYER_ID"] == player_id]

                if len(player_history) < min_history_for_prediction:
                    continue

                # Calculate features with opponent data
                features = calculate_all_features(
                    player_history,
                    pred_date,
                    player_name,
                    opponent_team,
                    val_season,
                    ctg_builder,
                    all_games_df,
                )

                feature_vector = [features.get(col, 0) for col in feature_cols]
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
                        "CTG_Available": features.get("CTG_Available", 0),
                    }
                )

        val_df = pd.DataFrame(val_predictions)
        logger.info(f"\n✅ Validation complete: {len(val_df):,} predictions")

        # Calculate metrics
        val_mae = mean_absolute_error(val_df["PRA"], val_df["predicted_PRA"])
        val_rmse = np.sqrt(mean_squared_error(val_df["PRA"], val_df["predicted_PRA"]))
        val_r2 = r2_score(val_df["PRA"], val_df["predicted_PRA"])

        # Log metrics
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

        # Save predictions
        output_path = data_config.RESULTS_DIR / "walk_forward_advanced_features_2024_25.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        val_df.to_csv(output_path, index=False)

        logger.info("\n" + "=" * 80)
        logger.info("RESULTS - ADVANCED FEATURES (DAY 4)")
        logger.info("=" * 80)
        logger.info(f"\nValidation Metrics (2024-25 Season):")
        logger.info(f"  MAE: {val_mae:.2f} points (Day 3: 6.11)")
        logger.info(f"  Improvement: {6.11 - val_mae:+.2f} points")
        logger.info(f"  RMSE: {val_rmse:.2f} points")
        logger.info(f"  R²: {val_r2:.3f}")
        logger.info(f"  Within ±3 pts: {val_metrics['within_3pts_pct']:.1f}%")
        logger.info(f"  Within ±5 pts: {val_metrics['within_5pts_pct']:.1f}%")
        logger.info(f"  Within ±10 pts: {val_metrics['within_10pts_pct']:.1f}%")
        logger.info(f"\nFeature Coverage:")
        logger.info(f"  CTG data available: {val_metrics['ctg_coverage']:.1f}%")
        logger.info(f"\n✅ Results saved to {output_path}")

        # Log feature importance
        importance = model.feature_importances_
        tracker.log_feature_importance(feature_cols, importance)

        # Log model
        tracker.log_model(model, model_type="xgboost")

        tracker.end_run(status="FINISHED")

        return {
            "model": model,
            "features": feature_cols,
            "val_df": val_df,
            "metrics": val_metrics,
            "run_id": tracker.run_id,
        }

    except Exception as e:
        logger.error(f"Error: {e}")
        tracker.end_run(status="FAILED")
        raise


if __name__ == "__main__":
    results = walk_forward_train_and_validate(
        train_season="2023-24",
        val_season="2024-25",
        min_train_games=10,
        min_history_for_prediction=5,
    )

    logger.info("\n" + "=" * 80)
    logger.info("DAY 4 COMPLETE")
    logger.info("=" * 80)
    logger.info(f"\nRun ID: {results['run_id']}")
    logger.info(f"MAE Improvement: {results['metrics']['improvement_from_day3']:+.2f} points")
    logger.info("\nView in MLflow: uv run mlflow ui")
