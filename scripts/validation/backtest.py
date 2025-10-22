#!/usr/bin/env python3
"""
CALIBRATED Walk-Forward Backtest - v2.0_CLEAN Model on 2024-25 Season

Enhanced backtest with isotonic regression calibration:
1. Pre-computes all Phase 1 features (fast - ~30 seconds)
2. Walk-forward validation (predict each date using only past data)
3. Converts predictions to calibrated win probabilities
4. Matches predictions to historical betting odds
5. Calculates edges using calibrated probabilities
6. Kelly Criterion staking with dynamic bankroll management
7. Comprehensive results output

Expected runtime: < 5 minutes total
"""

import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent))

from utils.fast_feature_builder import FastFeatureBuilder

# Configuration
EDGE_THRESHOLD = 0.05  # Only bet when edge >= 5%
STARTING_BANKROLL = 1000.0
KELLY_FRACTION = 0.25  # Fractional Kelly (25% of full Kelly)
MAX_BET_FRACTION = 0.10  # Never bet more than 10% of bankroll

print("=" * 80)
print("CALIBRATED BACKTEST - v2.0_CLEAN Model with Isotonic Regression")
print("2024-25 NBA Season")
print("=" * 80)
print(f"\nConfiguration:")
print(f"  Edge threshold: {EDGE_THRESHOLD*100:.0f}%")
print(f"  Starting bankroll: ${STARTING_BANKROLL:,.2f}")
print(f"  Kelly fraction: {KELLY_FRACTION*100:.0f}% (conservative)")
print(f"  Max bet size: {MAX_BET_FRACTION*100:.0f}% of bankroll")
print(f"  Using: Calibrated win probabilities (isotonic regression)")

# ============================================================================
# 1. LOAD CALIBRATED MODEL
# ============================================================================

print("\n" + "=" * 80)
print("STEP 1: Loading Backtest Model")
print("=" * 80)

# Use BIAS_FIXED model
with open("models/pra_model.pkl", "rb") as f:
    model_dict = pickle.load(f)

model = model_dict["model"]
feature_cols = model_dict["feature_cols"]
calibrator = model_dict.get("calibrator", None)

if calibrator is None:
    print("❌ No calibrator found in model! Using uncalibrated model.")
else:
    print("✅ Calibrator loaded")
    metrics = model_dict.get("calibration_metrics", {})
    print(f"   Brier score: {metrics.get('brier_score_calibrated', 0):.4f}")
    print(f"   Calibration samples: {metrics.get('calibration_samples', 0):,}")

print(f"✅ Model loaded: {model_dict.get('version', 'Unknown')}")
print(f"   Features: {len(feature_cols)}")
print(f"   Training MAE: {model_dict.get('train_mae', 0):.2f}")
print(f"   Validation MAE: {model_dict.get('val_mae', 0):.2f}")
print(f"   Test MAE: {model_dict.get('test_mae', 0):.2f}")

# ============================================================================
# 2. LOAD DATA
# ============================================================================

print("\n" + "=" * 80)
print("STEP 2: Loading Game Logs")
print("=" * 80)

# Historical data (2003-2024) for feature context
print("Loading historical games (2003-2024)...")
historical_df = pd.read_csv("data/game_logs/all_game_logs_through_2025.csv")
historical_df["GAME_DATE"] = pd.to_datetime(historical_df["GAME_DATE"])
historical_df = historical_df[historical_df["GAME_DATE"] < "2024-10-22"].copy()
historical_df = historical_df.sort_values(["PLAYER_ID", "GAME_DATE"])

print(f"✅ Historical: {len(historical_df):,} games")
print(
    f"   Date range: {historical_df['GAME_DATE'].min().date()} to {historical_df['GAME_DATE'].max().date()}"
)

# Test data (2024-25 season)
print("\nLoading 2024-25 test games...")
test_df = pd.read_csv("data/game_logs/game_logs_2024_25_preprocessed.csv")
test_df["GAME_DATE"] = pd.to_datetime(test_df["GAME_DATE"])
test_df = test_df.sort_values("GAME_DATE")

# Add PRA to both
if "PRA" not in historical_df.columns:
    historical_df["PRA"] = historical_df["PTS"] + historical_df["REB"] + historical_df["AST"]
if "PRA" not in test_df.columns:
    test_df["PRA"] = test_df["PTS"] + test_df["REB"] + test_df["AST"]

print(f"✅ Test set: {len(test_df):,} games")
print(f"   Date range: {test_df['GAME_DATE'].min().date()} to {test_df['GAME_DATE'].max().date()}")
print(f"   Unique dates: {test_df['GAME_DATE'].nunique()}")

# Load betting odds
print("\nLoading historical betting odds...")
odds_df = pd.read_csv("data/historical_odds/2024-25/pra_odds.csv")
odds_df["event_date"] = pd.to_datetime(odds_df["event_date"])

# CRITICAL FIX: Use only DraftKings to avoid duplicate betting on same props
print(f"📊 Total odds lines (all bookmakers): {len(odds_df):,}")
odds_df = odds_df[odds_df["bookmaker"] == "DraftKings"].copy()
print(f"✅ Filtered to DraftKings only: {len(odds_df):,} lines")

print(f"✅ Odds data: {len(odds_df):,} lines")
print(
    f"   Date range: {odds_df['event_date'].min().date()} to {odds_df['event_date'].max().date()}"
)
print(f"   Unique players: {odds_df['player_name'].nunique()}")
print(f"   Bookmaker: DraftKings (single source to avoid duplicates)")

# ============================================================================
# 3. BUILD FEATURES (Pre-compute once - FAST)
# ============================================================================

print("\n" + "=" * 80)
print("STEP 3: Building Phase 1 Features")
print("=" * 80)

builder = FastFeatureBuilder()
full_df = builder.build_features(historical_df, test_df, verbose=True)

# Separate back into historical and test
test_with_features = full_df[full_df["GAME_DATE"] >= "2024-10-22"].copy()

print(f"\n✅ Features ready for prediction")
print(f"   Test games with features: {len(test_with_features):,}")

# ============================================================================
# 4. WALK-FORWARD VALIDATION
# ============================================================================

print("\n" + "=" * 80)
print("STEP 4: Walk-Forward Validation")
print("=" * 80)
print("Predicting each date using ONLY past games (no future information)")

unique_dates = sorted(test_with_features["GAME_DATE"].unique())
print(f"Prediction dates: {len(unique_dates)}")

all_predictions = []

for pred_date in tqdm(unique_dates, desc="Walk-forward"):
    # Games to predict today
    games_today = test_with_features[test_with_features["GAME_DATE"] == pred_date]

    if len(games_today) == 0:
        continue

    # Make predictions
    for idx, row in games_today.iterrows():
        # Build feature vector
        feature_vector = []
        missing_count = 0

        for col in feature_cols:
            if col in row.index:
                val = row[col]
                # Fill NaN with 0
                if pd.isna(val):
                    val = 0
                    missing_count += 1
                feature_vector.append(val)
            else:
                feature_vector.append(0)
                missing_count += 1

        # Skip if too many missing features
        if missing_count > 30:
            continue

        # Predict
        try:
            pred_pra = model.predict([feature_vector])[0]
            pred_pra = max(0, pred_pra)  # Clip to non-negative
        except Exception as e:
            continue

        # Store prediction
        all_predictions.append(
            {
                "PLAYER_NAME": row["PLAYER_NAME"],
                "PLAYER_ID": row["PLAYER_ID"],
                "GAME_ID": row.get("GAME_ID", ""),
                "GAME_DATE": pred_date,
                "actual_PRA": row["PRA"],
                "predicted_PRA": pred_pra,
                "error": abs(pred_pra - row["PRA"]),
            }
        )

predictions_df = pd.DataFrame(all_predictions)

print(f"\n✅ Predictions complete")
print(f"   Total predictions: {len(predictions_df):,}")
print(f"   MAE: {predictions_df['error'].mean():.2f} points")

# ============================================================================
# 5. MATCH TO BETTING ODDS & CALCULATE CALIBRATED EDGES
# ============================================================================

print("\n" + "=" * 80)
print("STEP 5: Calculating Calibrated Win Probabilities & Edges")
print("=" * 80)

# Merge predictions with odds
betting_df = predictions_df.merge(
    odds_df,
    left_on=["PLAYER_NAME", "GAME_DATE"],
    right_on=["player_name", "event_date"],
    how="inner",
)

print(f"Matched predictions with odds: {len(betting_df):,}")

# Convert predicted PRA to raw win probabilities (logistic transform)
print("Converting predictions to probabilities...")
betting_df["difference"] = betting_df["predicted_PRA"] - betting_df["line"]
scale = 5.0  # Same scale used in calibration training
betting_df["prob_over_raw"] = 1 / (1 + np.exp(-betting_df["difference"] / scale))

# Apply calibrator to get calibrated probabilities
if calibrator is not None:
    betting_df["prob_over_calibrated"] = calibrator.predict(betting_df["prob_over_raw"])
    print(f"✅ Applied calibration")
    print(
        f"   Raw prob range: {betting_df['prob_over_raw'].min():.3f} to {betting_df['prob_over_raw'].max():.3f}"
    )
    print(
        f"   Calibrated prob range: {betting_df['prob_over_calibrated'].min():.3f} to {betting_df['prob_over_calibrated'].max():.3f}"
    )
else:
    betting_df["prob_over_calibrated"] = betting_df["prob_over_raw"]
    print("⚠️  No calibrator available, using raw probabilities")

# Determine bet direction (over or under)
betting_df["bet_over"] = (betting_df["predicted_PRA"] > betting_df["line"]).astype(int)
betting_df["bet_under"] = (betting_df["predicted_PRA"] < betting_df["line"]).astype(int)

# Get the correct price
betting_df["bet_price"] = betting_df.apply(
    lambda row: row["over_price"] if row["bet_over"] else row["under_price"], axis=1
)


# Convert American odds to decimal
def american_to_decimal(american_odds):
    if american_odds > 0:
        return 1 + (american_odds / 100)
    else:
        return 1 + (100 / abs(american_odds))


betting_df["decimal_odds"] = betting_df["bet_price"].apply(american_to_decimal)

# Calculate bookmaker's implied probability
betting_df["implied_prob"] = 1 / betting_df["decimal_odds"]

# Calculate edge using calibrated probability
# Edge = Our calibrated P(win) - Bookmaker's implied P(win)
betting_df["edge"] = betting_df.apply(
    lambda row: (
        row["prob_over_calibrated"] - row["implied_prob"]
        if row["bet_over"]
        else (1 - row["prob_over_calibrated"]) - row["implied_prob"]
    ),
    axis=1,
)

print("\nEdge calculation:")
print(f"   Mean edge: {betting_df['edge'].mean()*100:.2f}%")
print(
    f"   Positive edges: {(betting_df['edge'] > 0).sum():,} ({(betting_df['edge'] > 0).mean()*100:.1f}%)"
)

# Determine win/loss
betting_df["won"] = betting_df.apply(
    lambda row: (
        (row["actual_PRA"] > row["line"]) if row["bet_over"] else (row["actual_PRA"] < row["line"])
    ),
    axis=1,
).astype(int)

# ANALYSIS MODE: Keep ALL predictions (no edge filter) to analyze optimal strategy
print(f"\n📊 ANALYSIS MODE: Keeping ALL {len(betting_df):,} predictions (no filters)")
print(f"   Positive edges: {(betting_df['edge'] > 0).sum():,}")
print(f"   Negative edges: {(betting_df['edge'] <= 0).sum():,}")
print(f"   Edge >= {EDGE_THRESHOLD*100:.0f}%: {(betting_df['edge'] >= EDGE_THRESHOLD).sum():,}")

# NO FILTERING - keep everything for analysis
# betting_df = betting_df[betting_df["edge"] >= EDGE_THRESHOLD].copy()

if len(betting_df) == 0:
    print("\n❌ No predictions to analyze!")
    sys.exit(1)

# Sort chronologically
betting_df = betting_df.sort_values("GAME_DATE")

# ============================================================================
# 6. KELLY CRITERION STAKING & BANKROLL SIMULATION
# ============================================================================

print("\n" + "=" * 80)
print("STEP 6: Bankroll Simulation (Kelly Criterion)")
print("=" * 80)

bankroll = STARTING_BANKROLL
results = []

for idx, row in betting_df.iterrows():
    # Edge is already calculated using calibrated probabilities:
    # edge = P_calibrated(win) - P_implied(bookmaker)
    # This is the true expected value of the bet

    edge = row["edge"]
    odds = row["decimal_odds"]

    # Simplified Kelly: bet size proportional to edge
    # Full Kelly would be: (edge * odds - 1) / (odds - 1)
    # We use fractional Kelly (25%) for risk management
    kelly_bet_fraction = edge * KELLY_FRACTION

    # Enforce max bet size
    kelly_bet_fraction = min(kelly_bet_fraction, MAX_BET_FRACTION)

    # Ensure non-negative
    kelly_bet_fraction = max(0, kelly_bet_fraction)

    # Bet amount
    bet_amount = bankroll * kelly_bet_fraction

    # Safety check
    if bet_amount <= 0 or not np.isfinite(bet_amount):
        continue

    # Profit/loss
    if row["won"]:
        profit = bet_amount * (odds - 1)
    else:
        profit = -bet_amount

    # Update bankroll
    bankroll += profit

    # Store result
    results.append(
        {
            "date": row["GAME_DATE"],
            "player": row["PLAYER_NAME"],
            "predicted_PRA": row["predicted_PRA"],
            "actual_PRA": row["actual_PRA"],
            "line": row["line"],
            "edge": row["edge"],
            "bet_direction": "OVER" if row["bet_over"] else "UNDER",
            "decimal_odds": odds,
            "kelly_fraction": kelly_bet_fraction,
            "bet_amount": bet_amount,
            "won": row["won"],
            "profit": profit,
            "bankroll": bankroll,
        }
    )

results_df = pd.DataFrame(results)

# ============================================================================
# 7. RESULTS SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("RESULTS SUMMARY")
print("=" * 80)

total_bets = len(results_df)
wins = results_df["won"].sum()
losses = total_bets - wins
win_rate = wins / total_bets

total_wagered = results_df["bet_amount"].sum()
total_profit = results_df["profit"].sum()
roi = (total_profit / total_wagered) * 100

final_bankroll = results_df["bankroll"].iloc[-1]
total_return = final_bankroll - STARTING_BANKROLL
return_pct = (total_return / STARTING_BANKROLL) * 100

print(f"\nBetting Performance:")
print(f"  Total bets: {total_bets:,}")
print(f"  Wins: {wins} | Losses: {losses}")
print(f"  Win rate: {win_rate*100:.2f}%")
print(f"\nFinancial Performance:")
print(f"  Starting bankroll: ${STARTING_BANKROLL:,.2f}")
print(f"  Final bankroll: ${final_bankroll:,.2f}")
print(f"  Total profit: ${total_return:+,.2f}")
print(f"  Return: {return_pct:+.2f}%")
print(f"  Total wagered: ${total_wagered:,.2f}")
print(f"  ROI: {roi:+.2f}%")

# Calculate max drawdown
results_df["drawdown"] = (results_df["bankroll"] / results_df["bankroll"].cummax()) - 1
max_drawdown = results_df["drawdown"].min()
print(f"  Max drawdown: {max_drawdown*100:.2f}%")

# Sharpe-like metric (profit per bet / std)
profit_per_bet = results_df["profit"].mean()
profit_std = results_df["profit"].std()
if profit_std > 0:
    sharpe = profit_per_bet / profit_std
    print(f"  Sharpe ratio: {sharpe:.3f}")

# ============================================================================
# 8. SAVE RESULTS
# ============================================================================

print("\n" + "=" * 80)
print("STEP 7: Saving Results")
print("=" * 80)

output_file = "data/results/calibrated_backtest_2024_25.csv"
results_df.to_csv(output_file, index=False)
print(f"✅ Saved detailed results to: {output_file}")

# Save summary
calib_metrics = model_dict.get("calibration_metrics", {})
summary = {
    "model": "v2.0_CLEAN (139 features) + Isotonic Calibration",
    "season": "2024-25",
    "calibration": "isotonic_regression",
    "calibration_brier_score": calib_metrics.get("brier_score_calibrated", 0),
    "calibration_samples": calib_metrics.get("calibration_samples", 0),
    "edge_threshold": EDGE_THRESHOLD,
    "total_bets": total_bets,
    "wins": int(wins),
    "losses": int(losses),
    "win_rate": win_rate,
    "starting_bankroll": STARTING_BANKROLL,
    "final_bankroll": final_bankroll,
    "total_profit": total_return,
    "return_pct": return_pct,
    "roi": roi,
    "max_drawdown": max_drawdown,
    "mae": predictions_df["error"].mean(),
    "timestamp": datetime.now().isoformat(),
}

summary_file = "data/results/calibrated_backtest_summary.txt"
with open(summary_file, "w") as f:
    for key, value in summary.items():
        f.write(f"{key}: {value}\n")
print(f"✅ Saved summary to: {summary_file}")

print("\n" + "=" * 80)
print("✅ BACKTEST COMPLETE")
print("=" * 80)
print(f"\nKey Takeaways:")
print(f"  • Win Rate: {win_rate*100:.2f}% (target: 55%+)")
print(f"  • ROI: {roi:+.2f}% (target: 5-10%)")
print(f"  • MAE: {predictions_df['error'].mean():.2f} points (target: <5)")

if win_rate >= 0.54 and roi >= 3:
    print(f"\n🎯 Phase 1 model is PROFITABLE!")
    print(f"   Consider moving to production or Phase 2 feature engineering.")
elif win_rate >= 0.52:
    print(f"\n⚠️  Phase 1 model shows promise but needs improvement")
    print(f"   Consider calibration or Phase 2 features.")
else:
    print(f"\n❌ Phase 1 model underperforming baseline")
    print(f"   Investigate feature quality or model calibration.")
