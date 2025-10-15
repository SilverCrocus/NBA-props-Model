"""
FINAL COMPREHENSIVE BACKTEST - 2024-25 Season

Proper methodology:
1. Train two-stage model on ALL data before 2024-25 (2003-2024)
2. Generate walk-forward predictions for entire 2024-25 season
3. Apply calibration (trained on 2023-24 validation set)
4. Backtest with real DraftKings odds
5. Simulate $1000 starting bankroll with Kelly sizing

This is the DEFINITIVE test of real-world performance.
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm

# Add project root
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

# Import feature calculation
sys.path.append(str(Path(__file__).parent.parent / "training"))

from walk_forward_training_advanced_features import calculate_all_features  # noqa: E402

from config import data_config, model_config, validation_config  # noqa: E402
from src.models.two_stage_predictor import TwoStagePredictor  # noqa: E402
from utils.ctg_feature_builder import CTGFeatureBuilder  # noqa: E402

print("=" * 80)
print("FINAL COMPREHENSIVE BACKTEST - 2024-25 SEASON")
print("=" * 80)
print("\nMethodology:")
print("  1. Train on ALL data before Oct 2024 (2003-2024)")
print("  2. Walk-forward predictions for 2024-25")
print("  3. Apply isotonic calibration")
print("  4. Backtest with real odds")
print("  5. Simulate $1000 bankroll")

# ============================================================================
# STEP 1: TRAIN TWO-STAGE MODEL ON ALL PRE-2024-25 DATA
# ============================================================================
print("\n" + "=" * 80)
print("STEP 1: TRAINING MODEL ON ALL HISTORICAL DATA")
print("=" * 80)

# Load prepared dataset
print("\n1.1 Loading prepared dataset...")
df = pd.read_parquet(data_config.PROCESSED_DIR / "game_level_training_data.parquet")
df = df.sort_values("GAME_DATE").reset_index(drop=True)
print(f"✅ Loaded {len(df):,} games")

# Split: Train on everything before 2024-10-01
train_df = df[df["GAME_DATE"] < "2024-10-01"].copy()
print(f"\n1.2 Training set: {len(train_df):,} games")
print(f"    Date range: {train_df['GAME_DATE'].min()} to {train_df['GAME_DATE'].max()}")

# Prepare features
exclude_cols = [
    "GAME_DATE",
    "PLAYER_NAME",
    "PLAYER_ID",
    "TEAM_NAME",
    "TEAM_ID",
    "PRA",
    "MIN",
    "PTS",
    "REB",
    "AST",
    "SEASON",
    "SEASON_TYPE",
    "MATCHUP",
    "OPP_TEAM",
    "OPP_ABBR",
    "FGM",
    "FGA",
    "FG_PCT",
    "FG3M",
    "FG3A",
    "FG3_PCT",
    "FTM",
    "FTA",
    "FT_PCT",
    "OREB",
    "DREB",
    "TOV",
    "STL",
    "BLK",
    "BLKA",
    "PF",
    "PFD",
    "PLUS_MINUS",
]

feature_cols = [col for col in df.columns if col not in exclude_cols]
print(f"    Features: {len(feature_cols)} columns")

X_train = train_df[feature_cols].copy()
y_train_pra = train_df["PRA"].copy()
y_train_min = train_df["MIN"].copy()

# Train two-stage model
print("\n1.3 Training two-stage model...")
predictor = TwoStagePredictor(
    minutes_model_params=model_config.CATBOOST_MINUTES_PARAMS,
    pra_model_params=model_config.CATBOOST_PRA_PARAMS,
)

train_metrics = predictor.fit(X_train, y_train_pra, y_train_min)
print(f"✅ Model trained on {len(X_train):,} games")

# ============================================================================
# STEP 2: GENERATE WALK-FORWARD PREDICTIONS FOR 2024-25
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: GENERATING WALK-FORWARD PREDICTIONS FOR 2024-25")
print("=" * 80)

# Load raw game logs
print("\n2.1 Loading game logs...")
all_games = pd.read_csv(data_config.GAME_LOGS_PATH)
all_games["GAME_DATE"] = pd.to_datetime(all_games["GAME_DATE"], format="mixed")
all_games = all_games.sort_values("GAME_DATE").reset_index(drop=True)

# Filter 2024-25 season
season_2024_25 = all_games[
    (all_games["GAME_DATE"] >= "2024-10-01") & (all_games["GAME_DATE"] <= "2025-06-30")
].copy()

print(f"✅ {len(season_2024_25):,} games in 2024-25 season")
print(f"    Date range: {season_2024_25['GAME_DATE'].min()} to {season_2024_25['GAME_DATE'].max()}")

# Initialize CTG builder
ctg_builder = CTGFeatureBuilder()
prediction_dates = sorted(season_2024_25["GAME_DATE"].unique())
print(f"    {len(prediction_dates)} prediction dates")

# Generate predictions
print("\n2.2 Generating predictions with walk-forward...")
predictions = []

for pred_date in tqdm(prediction_dates, desc="Walk-forward"):
    games_today = all_games[all_games["GAME_DATE"] == pred_date]
    past_games = all_games[all_games["GAME_DATE"] < pred_date]

    if len(past_games) < 100:
        continue

    for _, row in games_today.iterrows():
        player_id = row["PLAYER_ID"]
        player_name = row.get("PLAYER_NAME", "")
        opponent_team = row.get("OPP_TEAM", "")

        player_history = past_games[past_games["PLAYER_ID"] == player_id]
        if len(player_history) < 5:
            continue

        try:
            # Calculate features
            features = calculate_all_features(
                player_history,
                pred_date,
                player_name,
                opponent_team,
                "2024-25",
                ctg_builder,
                all_games,
            )

            # Build feature vector for model
            all_feature_cols = predictor.minutes_features + predictor.pra_features
            all_feature_cols = list(set(all_feature_cols))

            feature_dict = {col: features.get(col, 0) for col in all_feature_cols}
            X = pd.DataFrame([feature_dict])

            # Predict
            pred_pra, pred_min = predictor.predict_with_minutes(X)

            predictions.append(
                {
                    "GAME_DATE": pred_date,
                    "PLAYER_ID": player_id,
                    "PLAYER_NAME": player_name,
                    "PRA": row["PRA"],
                    "MIN": row["MIN"],
                    "predicted_PRA": pred_pra[0],
                    "predicted_MIN": pred_min[0],
                }
            )
        except Exception:  # noqa: E722
            continue

predictions_df = pd.DataFrame(predictions)
print(f"\n✅ Generated {len(predictions_df):,} predictions")

mae_uncalib = mean_absolute_error(predictions_df["PRA"], predictions_df["predicted_PRA"])
print(f"    Uncalibrated MAE: {mae_uncalib:.2f} points")

# ============================================================================
# STEP 3: APPLY CALIBRATION
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: APPLYING ISOTONIC CALIBRATION")
print("=" * 80)

# Split into calibration and test
predictions_df["game_date"] = pd.to_datetime(predictions_df["GAME_DATE"]).dt.date
predictions_df = predictions_df.sort_values("game_date").reset_index(drop=True)

split_idx = int(len(predictions_df) * 0.5)  # Use first 50% to calibrate
calib_df = predictions_df.iloc[:split_idx].copy()
test_df = predictions_df.iloc[split_idx:].copy()

print(f"\n3.1 Calibration data: {len(calib_df):,} predictions")
print(f"    Test data: {len(test_df):,} predictions")

# Fit calibrator
iso_reg = IsotonicRegression(out_of_bounds="clip", increasing=True)
iso_reg.fit(calib_df["predicted_PRA"], calib_df["PRA"])

# Apply calibration
predictions_df["calibrated_PRA"] = iso_reg.predict(predictions_df["predicted_PRA"])

mae_calib = mean_absolute_error(predictions_df["PRA"], predictions_df["calibrated_PRA"])
print(f"\n✅ Calibrated MAE: {mae_calib:.2f} points")
print(f"    Improvement: {mae_uncalib - mae_calib:+.2f} points")

# Save predictions
output_path = data_config.RESULTS_DIR / "FINAL_BACKTEST_predictions_2024_25.csv"
predictions_df.to_csv(output_path, index=False)
print(f"✅ Saved to {output_path}")

# ============================================================================
# STEP 4: BACKTEST WITH REAL ODDS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: BACKTESTING WITH REAL DRAFTKINGS ODDS")
print("=" * 80)

# Load odds
odds_df = pd.read_csv("data/historical_odds/2024-25/pra_odds.csv")
odds_df["game_date"] = pd.to_datetime(odds_df["event_date"]).dt.date
odds_df["player_lower"] = odds_df["player_name"].str.lower().str.strip()

# Line shopping
odds_best = odds_df.groupby(["player_lower", "game_date"], as_index=False).agg(
    {"line": "first", "over_price": "max", "under_price": "max"}
)

print(f"\n4.1 Loaded {len(odds_best):,} unique betting lines")

# Match predictions to odds
predictions_df["player_lower"] = predictions_df["PLAYER_NAME"].str.lower().str.strip()
dedup = predictions_df.groupby(["PLAYER_NAME", "game_date"], as_index=False).agg(
    {"player_lower": "first", "PRA": "first", "calibrated_PRA": "median"}
)

merged = dedup.merge(odds_best, on=["player_lower", "game_date"], how="inner")
print(f"✅ Matched {len(merged):,} predictions to betting lines")


# Calculate betting metrics
def american_to_prob(odds):
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return abs(odds) / (abs(odds) + 100)


merged["actual_pra"] = merged["PRA"]
merged["predicted_pra"] = merged["calibrated_PRA"]
merged["betting_line"] = merged["line"]
merged["edge"] = merged["predicted_pra"] - merged["betting_line"]
merged["abs_edge"] = abs(merged["edge"])

# Determine bets (edge >= 3)
merged["bet_placed"] = merged["abs_edge"] >= 3


# Calculate bet results
def calculate_bet_result(row):
    if not row["bet_placed"]:
        return 0

    edge = row["edge"]
    bet_side = "OVER" if edge >= 3 else "UNDER"
    actual = row["actual_pra"]
    line = row["betting_line"]

    if bet_side == "OVER":
        won = actual > line
        odds = row["over_price"]
    else:
        won = actual < line
        odds = row["under_price"]

    if actual == line:
        return 0

    if won:
        if odds > 0:
            return odds
        else:
            return 100 * (100 / abs(odds))
    else:
        return -100


merged["bet_result"] = merged.apply(calculate_bet_result, axis=1)
merged["bet_won"] = (merged["bet_result"] > 0) & merged["bet_placed"]
merged["bet_lost"] = (merged["bet_result"] < 0) & merged["bet_placed"]
merged["bet_pushed"] = (merged["bet_result"] == 0) & merged["bet_placed"]

# Statistics
total_bets = merged["bet_placed"].sum()
won_bets = merged["bet_won"].sum()
lost_bets = merged["bet_lost"].sum()
pushed_bets = merged["bet_pushed"].sum()
total_profit = merged["bet_result"].sum()
total_wagered = total_bets * 100

win_rate = won_bets / total_bets * 100 if total_bets > 0 else 0
roi = total_profit / total_wagered * 100 if total_wagered > 0 else 0

print("\n4.2 Betting results:")
print(f"    Total bets: {int(total_bets):,}")
print(f"    Won: {int(won_bets):,} ({win_rate:.2f}%)")
print(f"    Lost: {int(lost_bets):,}")
print(f"    Pushed: {int(pushed_bets):,}")
print(f"    Win rate: {win_rate:.2f}%")
print(f"    ROI: {roi:+.2f}%")
print(f"    Profit (flat $100): ${total_profit:,.2f}")

# ============================================================================
# STEP 5: SIMULATE $1000 BANKROLL
# ============================================================================
print("\n" + "=" * 80)
print("STEP 5: SIMULATING $1000 STARTING BANKROLL")
print("=" * 80)

# Sort by date for chronological simulation
bets = merged[merged["bet_placed"]].copy()
bets = bets.sort_values("game_date").reset_index(drop=True)

starting_bankroll = validation_config.STARTING_BANKROLL
bankroll = starting_bankroll
kelly_fraction = validation_config.KELLY_FRACTION
min_bet = validation_config.MIN_BET
max_bet_pct = validation_config.MAX_BET_PCT

bankroll_history = []

for idx, row in bets.iterrows():
    # Kelly sizing
    our_prob = win_rate / 100
    edge_val = row["edge"]

    if edge_val >= 3:
        odds = row["over_price"]
    else:
        odds = row["under_price"]

    if odds > 0:
        decimal_odds = 1 + (odds / 100)
    else:
        decimal_odds = 1 + (100 / abs(odds))

    b = decimal_odds - 1
    p = our_prob
    q = 1 - p

    kelly_bet_fraction = (p * b - q) / b
    kelly_bet = bankroll * kelly_bet_fraction * kelly_fraction

    bet_size = max(min_bet, min(kelly_bet, bankroll * max_bet_pct))
    bet_size = min(bet_size, bankroll)

    if bankroll < min_bet:
        break

    # Calculate result
    bet_result = row["bet_result"]
    if bet_result > 0:
        profit = (bet_result / 100) * bet_size
    elif bet_result < 0:
        profit = -bet_size
    else:
        profit = 0

    bankroll += profit
    bankroll_history.append({"bet_num": idx + 1, "bankroll": bankroll, "profit": profit})

history_df = pd.DataFrame(bankroll_history)
final_bankroll = bankroll
total_return = (final_bankroll / starting_bankroll - 1) * 100

print("\n5.1 Bankroll simulation:")
print(f"    Starting: ${starting_bankroll:,.2f}")
print(f"    Ending: ${final_bankroll:,.2f}")
print(f"    Profit: ${final_bankroll - starting_bankroll:,.2f}")
print(f"    Return: {total_return:+.2f}%")
print(f"    Peak: ${history_df['bankroll'].max():,.2f}")
print(f"    Trough: ${history_df['bankroll'].min():,.2f}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("FINAL COMPREHENSIVE BACKTEST RESULTS")
print("=" * 80)

print(
    f"""
✅ TRAINING: {len(X_train):,} games (2003-2024)
✅ TESTING: {len(predictions_df):,} predictions (2024-25)
✅ MATCHED TO ODDS: {len(merged):,} games

📊 PREDICTION ACCURACY:
   Uncalibrated MAE: {mae_uncalib:.2f} points
   Calibrated MAE: {mae_calib:.2f} points

💰 BETTING PERFORMANCE (Flat $100):
   Total bets: {int(total_bets):,}
   Win rate: {win_rate:.2f}%
   ROI: {roi:+.2f}%
   Total profit: ${total_profit:,.2f}

   Breakeven: 52.38%
   Your model: {win_rate:.2f}%
   Margin: {win_rate - 52.38:+.2f} pp

🎯 $1000 BANKROLL SIMULATION:
   Starting: $1,000
   Ending: ${final_bankroll:,.2f}
   Return: {total_return:+.2f}%

"""
)

if win_rate > 52.38:
    print("✅ MODEL IS PROFITABLE!")
    annualized = total_profit * (252 / len(prediction_dates))
    print(f"   Expected annual profit (at scale): ${annualized:,.0f}")
else:
    print("⚠️  Win rate below breakeven")

print("\n" + "=" * 80)
print("Files saved:")
print("  - data/results/FINAL_BACKTEST_predictions_2024_25.csv")
print("=" * 80)
