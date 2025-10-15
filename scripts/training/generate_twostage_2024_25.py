"""
Generate Full 2024-25 Predictions with Two-Stage Model

Uses the trained two-stage model to generate predictions for all 2024-25 games
with proper walk-forward validation (no temporal leakage).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm

# Add project root
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.models.two_stage_predictor import TwoStagePredictor
from utils.ctg_feature_builder import CTGFeatureBuilder

# Import feature calculation from baseline script
sys.path.append(str(Path(__file__).parent))
from walk_forward_training_advanced_features import calculate_all_features

print("=" * 80)
print("TWO-STAGE MODEL - FULL 2024-25 PREDICTIONS")
print("=" * 80)

# Load trained two-stage model
print("\n1. Loading trained two-stage model...")
try:
    predictor = TwoStagePredictor.load("models/two_stage_model")
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("   Run: uv run python scripts/training/quick_two_stage_train.py")
    sys.exit(1)

# Load raw game logs
print("\n2. Loading game logs...")
game_logs_path = "data/game_logs/all_game_logs_with_opponent.csv"
if not Path(game_logs_path).exists():
    print(f"❌ Game logs not found: {game_logs_path}")
    sys.exit(1)

all_games = pd.read_csv(game_logs_path)
all_games["GAME_DATE"] = pd.to_datetime(all_games["GAME_DATE"], format="mixed")
all_games = all_games.sort_values("GAME_DATE").reset_index(drop=True)

print(f"✅ Loaded {len(all_games):,} games")
print(f"   Date range: {all_games['GAME_DATE'].min()} to {all_games['GAME_DATE'].max()}")

# Filter to 2024-25 season
season_2024_25 = all_games[
    (all_games["GAME_DATE"] >= "2024-10-01") & (all_games["GAME_DATE"] <= "2025-06-30")
].copy()

print(f"\n3. 2024-25 season games: {len(season_2024_25):,}")
print(f"   Date range: {season_2024_25['GAME_DATE'].min()} to {season_2024_25['GAME_DATE'].max()}")
print(f"   Unique dates: {season_2024_25['GAME_DATE'].nunique()}")

# Initialize CTG feature builder
print("\n4. Initializing CTG feature builder...")
ctg_builder = CTGFeatureBuilder()

# Get unique prediction dates
prediction_dates = sorted(season_2024_25["GAME_DATE"].unique())
print(f"   Prediction dates: {len(prediction_dates)}")

# Generate predictions with walk-forward
print("\n5. Generating predictions with walk-forward validation...")
predictions = []
failed_predictions = 0

for pred_date in tqdm(prediction_dates, desc="Walk-forward"):
    # Games to predict today
    games_today = all_games[all_games["GAME_DATE"] == pred_date]

    # Historical data (strictly before today)
    past_games = all_games[all_games["GAME_DATE"] < pred_date]

    if len(past_games) < 100:  # Need minimum history
        continue

    for _, row in games_today.iterrows():
        player_id = row["PLAYER_ID"]
        player_name = row.get("PLAYER_NAME", "")
        opponent_team = row.get("OPP_TEAM", "")

        # Get player history
        player_history = past_games[past_games["PLAYER_ID"] == player_id]

        if len(player_history) < 5:  # Need minimum player history
            continue

        try:
            # Calculate features (same as baseline)
            features = calculate_all_features(
                player_history,
                pred_date,
                player_name,
                opponent_team,
                "2024-25",
                ctg_builder,
                all_games,
            )

            # Convert to DataFrame for two-stage predictor
            # Get feature columns from trained model
            all_feature_cols = predictor.minutes_features + predictor.pra_features
            all_feature_cols = list(set(all_feature_cols))  # Remove duplicates

            # Build feature vector
            feature_dict = {}
            for col in all_feature_cols:
                feature_dict[col] = features.get(col, 0)

            X = pd.DataFrame([feature_dict])

            # Make prediction with two-stage model
            pred_pra, pred_min = predictor.predict_with_minutes(X)

            # Store prediction
            predictions.append(
                {
                    "GAME_DATE": pred_date,
                    "PLAYER_ID": player_id,
                    "PLAYER_NAME": player_name,
                    "PRA": row["PRA"],
                    "MIN": row["MIN"],
                    "predicted_PRA": pred_pra[0],
                    "predicted_MIN": pred_min[0],
                    "error": abs(row["PRA"] - pred_pra[0]),
                    "abs_error": abs(row["PRA"] - pred_pra[0]),
                    "minutes_error": abs(row["MIN"] - pred_min[0]),
                }
            )

        except Exception as e:
            failed_predictions += 1
            continue

# Convert to DataFrame
predictions_df = pd.DataFrame(predictions)

print(f"\n" + "=" * 80)
print("PREDICTION RESULTS")
print("=" * 80)
print(f"\nTotal predictions: {len(predictions_df):,}")
print(f"Failed predictions: {failed_predictions:,}")
print(f"Success rate: {len(predictions_df)/(len(predictions_df)+failed_predictions)*100:.1f}%")

if len(predictions_df) > 0:
    # Calculate metrics
    mae_pra = mean_absolute_error(predictions_df["PRA"], predictions_df["predicted_PRA"])
    mae_min = mean_absolute_error(predictions_df["MIN"], predictions_df["predicted_MIN"])

    print(f"\n📊 PERFORMANCE:")
    print(f"   PRA MAE: {mae_pra:.2f} points")
    print(f"   Minutes MAE: {mae_min:.2f} minutes")

    # Compare to baseline
    baseline_mae = 8.83  # From walk_forward_advanced_features_2024_25.csv
    improvement = baseline_mae - mae_pra
    improvement_pct = (improvement / baseline_mae) * 100

    print(f"\n   Baseline MAE: {baseline_mae:.2f} points")
    print(f"   Improvement: {improvement:+.2f} points ({improvement_pct:+.1f}%)")

    if mae_pra < baseline_mae:
        print(f"   ✅ TWO-STAGE IS BETTER!")
    else:
        print(f"   ⚠️  Two-stage did not improve over baseline")

    # Accuracy bands
    print(f"\n📊 ACCURACY:")
    print(f"   Within ±3 pts: {(predictions_df['abs_error'] <= 3).mean()*100:.1f}%")
    print(f"   Within ±5 pts: {(predictions_df['abs_error'] <= 5).mean()*100:.1f}%")
    print(f"   Within ±10 pts: {(predictions_df['abs_error'] <= 10).mean()*100:.1f}%")

    # Save predictions
    output_file = "data/results/two_stage_predictions_2024_25_FULL.csv"
    predictions_df.to_csv(output_file, index=False)
    print(f"\n✅ Saved to {output_file}")

    # Compare coverage to baseline
    baseline_df = pd.read_csv("data/results/walk_forward_advanced_features_2024_25.csv")
    print(f"\n📊 COVERAGE COMPARISON:")
    print(f"   Baseline: {len(baseline_df):,} predictions")
    print(f"   Two-stage: {len(predictions_df):,} predictions")
    print(f"   Difference: {len(predictions_df) - len(baseline_df):,} predictions")

    if len(predictions_df) < len(baseline_df) * 0.9:
        print(f"   ⚠️  Two-stage has significantly fewer predictions")
        print(f"      This may be due to missing features or stricter requirements")
else:
    print("\n❌ No predictions generated")

print("\n" + "=" * 80)
print("GENERATION COMPLETE")
print("=" * 80)
print(f"\nNext steps:")
print(f"  1. Apply calibration: uv run python scripts/calibration/calibrate_twostage.py")
print(f"  2. Backtest with odds: uv run python scripts/backtesting/backtest_twostage.py")
print(f"  3. Compare all approaches")
