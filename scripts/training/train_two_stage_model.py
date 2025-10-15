"""
Train Two-Stage Predictor (Minutes → PRA)

This script implements the KEY INNOVATION identified by all agents:
Minutes variance accounts for 40-50% of current MAE 6.10.

Architecture:
- Stage 1: Predict minutes played (using recent minutes, rest, pace)
- Stage 2: Predict PRA given predicted minutes (using efficiency + predicted_MIN)

Expected Impact: MAE 6.10 → 5.60-5.80 (7-8% improvement)

This is Phase 1 Weeks 2-3 of the strategic roadmap.

Author: NBA Props Model - Phase 1 Weeks 2-3
Date: October 14, 2025
"""

import logging
import sys
from datetime import timedelta
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from config import data_config, model_config
from src.models.two_stage_predictor import TwoStagePredictor

# Add utils for CTG features
sys.path.append(str(Path(__file__).parent.parent.parent / "utils"))
from ctg_feature_builder import CTGFeatureBuilder

# Import feature calculation functions from Day 4 script
sys.path.append(str(Path(__file__).parent))
from walk_forward_training_advanced_features import (
    calculate_all_features,
    calculate_efficiency_features,
    calculate_ewma_features,
    calculate_lag_features,
    calculate_normalization_features,
    calculate_opponent_features,
    calculate_rest_features,
    calculate_rolling_features,
    calculate_trend_features,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_training_data() -> pd.DataFrame:
    """
    Load raw game logs data (not pre-processed features).

    Returns:
        DataFrame with raw game logs
    """
    logger.info("Loading raw game logs...")

    # Use raw game logs (same as Day 4)
    game_logs_path = data_config.GAME_LOGS_PATH

    if not game_logs_path.exists():
        raise FileNotFoundError(f"Game logs not found: {game_logs_path}")

    df = pd.read_csv(game_logs_path)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], format="mixed")
    df = df.sort_values("GAME_DATE").reset_index(drop=True)

    # Check for required columns
    required_cols = ["PRA", "MIN", "GAME_DATE", "PLAYER_ID"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    logger.info(f"   Loaded {game_logs_path}")
    logger.info(f"   Games: {len(df):,}")
    logger.info(f"   Date range: {df['GAME_DATE'].min()} to {df['GAME_DATE'].max()}")
    logger.info(f"   Players: {df['PLAYER_ID'].nunique():,}")

    return df


def walk_forward_validation(
    all_games_df: pd.DataFrame,
    start_date: str = "2024-10-01",  # 2024-25 season start
    n_predictions: int = 100,  # Limit for Phase 1 testing
    min_history_for_prediction: int = 5,
    season: str = "2024-25",
) -> pd.DataFrame:
    """
    Walk-forward validation for two-stage predictor.

    This follows the Day 4 pattern of building features on-the-fly
    for each prediction date to maintain temporal isolation.

    Args:
        all_games_df: Full game logs (all historical data)
        start_date: Start of prediction period (2024-25 season)
        n_predictions: Number of prediction dates to test
        min_history_for_prediction: Minimum games needed to make prediction
        season: Current season

    Returns:
        DataFrame with predictions
    """
    logger.info("\n" + "=" * 80)
    logger.info("WALK-FORWARD VALIDATION (TWO-STAGE PREDICTOR)")
    logger.info("=" * 80)

    # Initialize CTG feature builder
    logger.info("   Initializing CTG feature builder...")
    ctg_builder = CTGFeatureBuilder()

    # Get prediction dates (2024-25 season)
    prediction_dates = sorted(
        all_games_df[all_games_df["GAME_DATE"] >= start_date]["GAME_DATE"].unique()
    )

    if len(prediction_dates) > n_predictions:
        logger.info(f"   Limiting to first {n_predictions} dates for testing")
        prediction_dates = prediction_dates[:n_predictions]

    logger.info(f"   Prediction dates: {len(prediction_dates)}")
    logger.info(f"   Date range: {prediction_dates[0]} to {prediction_dates[-1]}")

    # Storage for predictions
    all_predictions = []

    # MLflow experiment
    mlflow.set_experiment("Phase1_TwoStage")

    with mlflow.start_run(run_name="two_stage_walk_forward"):
        mlflow.log_param("n_prediction_dates", len(prediction_dates))
        mlflow.log_param("start_date", str(prediction_dates[0]))
        mlflow.log_param("end_date", str(prediction_dates[-1]))
        mlflow.log_param("season", season)

        # Track metrics across dates
        date_maes = []
        date_minutes_r2s = []

        # Initialize predictor (will be trained on first iteration)
        predictor = None
        feature_cols = None

        for i, pred_date in enumerate(tqdm(prediction_dates, desc="Walk-forward validation"), 1):
            # Games to predict today
            games_today = all_games_df[all_games_df["GAME_DATE"] == pred_date]

            if len(games_today) == 0:
                continue

            # Historical data (strictly before today)
            past_games = all_games_df[all_games_df["GAME_DATE"] < pred_date]

            if len(past_games) < 100:  # Need minimum training data
                continue

            # For Phase 1 testing, train model only on FIRST iteration
            # (In production, would retrain periodically)
            if i == 1:
                logger.info("\n   Training two-stage model on historical data...")
                logger.info(f"   Using training data: {len(past_games):,} games")

                # Build training dataset from recent historical data
                # Use last 30K games for efficiency (still covers ~2 seasons)
                training_data = past_games.tail(30000) if len(past_games) > 30000 else past_games
                training_samples = []

                for _, row in tqdm(
                    training_data.iterrows(),
                    total=len(training_data),
                    desc="Building training data",
                ):
                    player_id = row["PLAYER_ID"]
                    player_name = row.get("PLAYER_NAME", "")
                    opponent_team = row.get("OPP_TEAM", "")
                    game_date = row["GAME_DATE"]

                    # Get player history BEFORE this game
                    player_history = past_games[
                        (past_games["PLAYER_ID"] == player_id)
                        & (past_games["GAME_DATE"] < game_date)
                    ]

                    if len(player_history) < 5:  # Need minimum history
                        continue

                    # Calculate features
                    features = calculate_all_features(
                        player_history,
                        game_date,
                        player_name,
                        opponent_team,
                        "2023-24",  # Use 2023-24 season for training
                        ctg_builder,
                        all_games_df,
                    )

                    features["PRA"] = row["PRA"]
                    features["MIN"] = row["MIN"]

                    training_samples.append(features)

                train_df = pd.DataFrame(training_samples)

                # Define feature columns
                exclude_cols = ["PRA", "MIN", "CTG_Available"]
                feature_cols = [col for col in train_df.columns if col not in exclude_cols]

                X_train = train_df[feature_cols].fillna(0)
                y_train_pra = train_df["PRA"]
                y_train_minutes = train_df["MIN"]

                logger.info(f"   Training samples: {len(X_train):,}")
                logger.info(f"   Features: {len(feature_cols)}")

                # Initialize and train two-stage predictor
                predictor = TwoStagePredictor()
                train_metrics = predictor.fit(X_train, y_train_pra, y_train_minutes)

                logger.info("   ✅ Two-stage model trained!\n")

            # Skip if predictor not trained yet
            if predictor is None or feature_cols is None:
                continue

            # Now make predictions for today's games
            try:
                for _, row in games_today.iterrows():
                    player_id = row["PLAYER_ID"]
                    player_name = row.get("PLAYER_NAME", "")
                    opponent_team = row.get("OPP_TEAM", "")

                    player_history = past_games[past_games["PLAYER_ID"] == player_id]

                    if len(player_history) < min_history_for_prediction:
                        continue

                    # Calculate features for this game
                    features = calculate_all_features(
                        player_history,
                        pred_date,
                        player_name,
                        opponent_team,
                        season,
                        ctg_builder,
                        all_games_df,
                    )

                    # Convert to feature vector
                    feature_vector = [features.get(col, 0) for col in feature_cols]
                    X_test = pd.DataFrame([dict(zip(feature_cols, feature_vector))])

                    # Predict (returns PRA and minutes)
                    predicted_pra, predicted_minutes = predictor.predict_with_minutes(X_test)

                    # Store prediction
                    all_predictions.append(
                        {
                            "GAME_DATE": pred_date,
                            "PLAYER_ID": player_id,
                            "PLAYER_NAME": player_name,
                            "actual_PRA": row["PRA"],
                            "predicted_PRA": predicted_pra[0],
                            "actual_MIN": row["MIN"],
                            "predicted_MIN": predicted_minutes[0],
                            "error": abs(row["PRA"] - predicted_pra[0]),
                            "minutes_error": abs(row["MIN"] - predicted_minutes[0]),
                        }
                    )

                # Calculate metrics for this date
                date_predictions = [p for p in all_predictions if p["GAME_DATE"] == pred_date]
                if len(date_predictions) > 0:
                    date_df = pd.DataFrame(date_predictions)
                    mae = date_df["error"].mean()
                    minutes_r2 = r2_score(date_df["actual_MIN"], date_df["predicted_MIN"])

                    date_maes.append(mae)
                    date_minutes_r2s.append(minutes_r2)

                    # Progress logging
                    if i % 10 == 0 or i == len(prediction_dates):
                        logger.info(
                            f"   [{i}/{len(prediction_dates)}] {pred_date}: "
                            f"MAE {mae:.2f}, Minutes R² {minutes_r2:.3f}, "
                            f"Games {len(date_predictions)}, Train {len(train_df):,}"
                        )

            except Exception as e:
                logger.warning(f"   Skipping {pred_date}: {str(e)}")
                continue

        # Final metrics
        predictions_df = pd.DataFrame(all_predictions)

        overall_mae = mean_absolute_error(
            predictions_df["actual_PRA"], predictions_df["predicted_PRA"]
        )
        overall_minutes_mae = mean_absolute_error(
            predictions_df["actual_MIN"], predictions_df["predicted_MIN"]
        )
        overall_minutes_r2 = r2_score(predictions_df["actual_MIN"], predictions_df["predicted_MIN"])

        logger.info("\n" + "=" * 80)
        logger.info("WALK-FORWARD RESULTS")
        logger.info("=" * 80)
        logger.info(f"   Total Predictions: {len(predictions_df):,}")
        logger.info(f"   Prediction Dates: {len(prediction_dates)}")
        logger.info(f"\n   PRA Prediction:")
        logger.info(f"      MAE: {overall_mae:.2f} points")
        logger.info(
            f"      RMSE: {np.sqrt(mean_squared_error(predictions_df['actual_PRA'], predictions_df['predicted_PRA'])):.2f}"
        )
        logger.info(
            f"      R²: {r2_score(predictions_df['actual_PRA'], predictions_df['predicted_PRA']):.3f}"
        )
        logger.info(f"\n   Minutes Prediction (Stage 1):")
        logger.info(f"      MAE: {overall_minutes_mae:.2f} minutes")
        logger.info(f"      R²: {overall_minutes_r2:.3f}")
        logger.info(f"      (Target: R² > 0.85)")

        # Log to MLflow
        mlflow.log_metric("mae_pra", overall_mae)
        mlflow.log_metric("mae_minutes", overall_minutes_mae)
        mlflow.log_metric("r2_minutes", overall_minutes_r2)
        mlflow.log_metric(
            "r2_pra", r2_score(predictions_df["actual_PRA"], predictions_df["predicted_PRA"])
        )
        mlflow.log_metric("n_predictions", len(predictions_df))

        # Compare to baseline
        baseline_mae = 6.10  # From Day 4
        improvement = baseline_mae - overall_mae
        improvement_pct = (improvement / baseline_mae) * 100

        logger.info(f"\n   IMPROVEMENT vs Baseline (Day 4):")
        logger.info(f"      Baseline MAE: {baseline_mae:.2f}")
        logger.info(f"      Two-Stage MAE: {overall_mae:.2f}")
        logger.info(f"      Improvement: {improvement:+.2f} points ({improvement_pct:+.1f}%)")

        if improvement > 0:
            logger.info(f"      ✅ BETTER than baseline!")
        else:
            logger.info(f"      ⚠️ WORSE than baseline (need to investigate)")

        mlflow.log_metric("baseline_mae", baseline_mae)
        mlflow.log_metric("improvement_mae", improvement)
        mlflow.log_metric("improvement_pct", improvement_pct)

    return predictions_df


def analyze_predictions(predictions_df: pd.DataFrame):
    """
    Analyze prediction errors and patterns.
    """
    logger.info("\n" + "=" * 80)
    logger.info("ERROR ANALYSIS")
    logger.info("=" * 80)

    # Error distribution
    logger.info("\n   PRA Error Distribution:")
    logger.info(f"      Mean: {predictions_df['error'].mean():.2f}")
    logger.info(f"      Median: {predictions_df['error'].median():.2f}")
    logger.info(f"      95th percentile: {predictions_df['error'].quantile(0.95):.2f}")
    logger.info(f"      Max: {predictions_df['error'].max():.2f}")

    # Minutes error distribution
    logger.info("\n   Minutes Error Distribution:")
    logger.info(f"      Mean: {predictions_df['minutes_error'].mean():.2f}")
    logger.info(f"      Median: {predictions_df['minutes_error'].median():.2f}")
    logger.info(f"      95th percentile: {predictions_df['minutes_error'].quantile(0.95):.2f}")

    # Error by actual minutes played (key insight)
    logger.info("\n   Error by Minutes Played:")
    predictions_df["minutes_bin"] = pd.cut(
        predictions_df["actual_MIN"],
        bins=[0, 10, 20, 30, 40, 50],
        labels=["0-10", "10-20", "20-30", "30-40", "40+"],
    )

    for minutes_bin, group in predictions_df.groupby("minutes_bin"):
        mae = group["error"].mean()
        count = len(group)
        logger.info(f"      {minutes_bin} min: MAE {mae:.2f} ({count:,} games)")

    # Correlation between minutes error and PRA error
    minutes_pra_corr = predictions_df["minutes_error"].corr(predictions_df["error"])
    logger.info(f"\n   Correlation (Minutes Error ↔ PRA Error): {minutes_pra_corr:.3f}")
    logger.info(f"      (High correlation = minutes prediction drives PRA accuracy)")


def save_results(predictions_df: pd.DataFrame, predictor: TwoStagePredictor):
    """
    Save predictions and trained model.
    """
    logger.info("\n" + "=" * 80)
    logger.info("SAVING RESULTS")
    logger.info("=" * 80)

    # Save predictions
    output_path = data_config.RESULTS_DIR / "two_stage_predictions_2024_25.csv"
    predictions_df.to_csv(output_path, index=False)
    logger.info(f"   ✅ Predictions saved to {output_path}")

    # Save model
    model_path = data_config.MODELS_DIR / "two_stage_predictor"
    # Note: Can't save the last predictor from loop, need to retrain on full data
    # This is just a placeholder for now
    logger.info(f"   ⚠️ Model saving requires full dataset training (TODO)")

    # Feature importance (would need to retrain on full dataset)
    logger.info(f"\n   Feature importance available after full training")


def main():
    logger.info("=" * 80)
    logger.info("PHASE 1 WEEKS 2-3: TWO-STAGE PREDICTOR TRAINING")
    logger.info("=" * 80)
    logger.info("\n🎯 OBJECTIVE: Reduce MAE from 6.10 → 5.60-5.80 (7-8% improvement)")
    logger.info("   via two-stage prediction: Minutes → PRA\n")

    # Load data
    all_games_df = load_training_data()

    # Run walk-forward validation
    predictions_df = walk_forward_validation(
        all_games_df,
        start_date="2024-10-01",
        n_predictions=50,  # Test on first 50 dates (faster)
        min_history_for_prediction=5,
        season="2024-25",
    )

    # Analyze predictions
    analyze_predictions(predictions_df)

    # Save results
    save_results(predictions_df, None)

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 1 WEEKS 2-3 COMPLETE")
    logger.info("=" * 80)

    final_mae = mean_absolute_error(predictions_df["actual_PRA"], predictions_df["predicted_PRA"])
    baseline_mae = 6.10

    logger.info(f"\n📊 FINAL RESULTS:")
    logger.info(f"   Baseline (Day 4): {baseline_mae:.2f} MAE")
    logger.info(f"   Two-Stage: {final_mae:.2f} MAE")
    logger.info(f"   Improvement: {baseline_mae - final_mae:+.2f} points")

    if final_mae < 5.80:
        logger.info(f"\n🎉 TARGET ACHIEVED! MAE < 5.80")
        logger.info("   Expected win rate: 54-56% (barely profitable)")
        logger.info("   ✅ Ready for Phase 2: Tree Ensemble + Position-Specific Defense")
    elif final_mae < baseline_mae:
        logger.info(f"\n✅ IMPROVEMENT but target not reached")
        logger.info("   Need to investigate feature selection or hyperparameters")
    else:
        logger.info(f"\n⚠️ NO IMPROVEMENT")
        logger.info("   Need to diagnose: Minutes prediction quality, feature selection")

    logger.info(f"\n📁 Files Created:")
    logger.info(f"   data/results/two_stage_predictions_2024_25.csv")
    logger.info(f"\n🚀 Next Steps:")
    logger.info(f"   1. Combine calibration + two-stage for full Phase 1")
    logger.info(f"   2. Validate on full 2024-25 season (all 204 dates)")
    logger.info(f"   3. Proceed to Phase 2 if MAE < 5.80")


if __name__ == "__main__":
    main()
