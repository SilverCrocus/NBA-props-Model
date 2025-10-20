#!/usr/bin/env python3
"""
Backtest V2 Model on 2024-25 Season

Walk-forward validation on 2024-25 season to validate:
1. Win rate (expected 52-55%)
2. ROI (expected 5-10%)
3. MAE (expected 6-8 pts)
4. Betting strategy performance

Uses ONLY pre-game features (no in-game stats).
"""

import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from pathlib import Path

# ======================================================================
# CONFIGURATION
# ======================================================================

MODEL_PATH = "models/production_model_FIXED_V2_latest.pkl"
GAME_LOGS_PATH = "data/game_logs/all_game_logs_through_2025.csv"
ODDS_PATH = "data/historical_odds/2024-25/pra_odds.csv"  # Historical odds for 2024-25
OUTPUT_PATH = "data/results/backtest_2024_25_FIXED_V2.csv"

# Betting strategy (Phase 1 fix: skip 10+ edges)
# Analysis showed 5-7 pt edges achieve 52.03% win rate
# while 10+ pt edges achieve only 49.74% (overconfidence issue)
MIN_EDGE = 5.0
MAX_EDGE = 7.0

# Star players to exclude (efficient markets)
STAR_PLAYERS = [
    'LeBron James', 'Stephen Curry', 'Kevin Durant', 'Giannis Antetokounmpo',
    'Luka Doncic', 'Joel Embiid', 'Nikola Jokic', 'Jayson Tatum', 'Anthony Davis',
    'Damian Lillard', 'Devin Booker', 'Jimmy Butler', 'Kawhi Leonard',
    'Paul George', 'Kyrie Irving', 'James Harden', 'Trae Young', 'Donovan Mitchell',
    'Jaylen Brown', 'LaMelo Ball', 'De\'Aaron Fox', 'Anthony Edwards', 'Pascal Siakam',
    'Tyrese Haliburton', 'DeMar DeRozan', 'Shai Gilgeous-Alexander', 'Bam Adebayo',
    'Domantas Sabonis', 'Julius Randle', 'Zion Williamson', 'Brandon Ingram',
    'Dejounte Murray', 'Fred VanVleet', 'Jrue Holiday', 'CJ McCollum',
    'Tobias Harris', 'Khris Middleton', 'Tyler Herro', 'Jarrett Allen',
    'Kristaps Porzingis', 'Clint Capela', 'Jaren Jackson Jr', 'Scottie Barnes',
    'Paolo Banchero', 'Franz Wagner', 'Cade Cunningham', 'Jalen Green',
    'Alperen Sengun', 'Lauri Markkanen', 'Walker Kessler', 'Evan Mobley',
    'Darius Garland', 'Derrick White', 'OG Anunoby', 'RJ Barrett', 'Immanuel Quickley',
    'Mikal Bridges', 'Cameron Johnson', 'Nic Claxton', 'Spencer Dinwiddie',
    'Cam Thomas'
]

print("=" * 80)
print("BACKTEST V2 MODEL ON 2024-25 SEASON")
print("=" * 80)
print()

# ======================================================================
# LOAD MODEL
# ======================================================================

print("1. Loading V2 model...")
with open(MODEL_PATH, 'rb') as f:
    model_dict = pickle.load(f)

model = model_dict['model']
feature_cols = model_dict['feature_cols']
scaler = model_dict.get('scaler', None)

print(f"   ✅ Model loaded: {len(feature_cols)} features")
print(f"   Training MAE: {model_dict.get('train_mae', 'N/A'):.2f} pts" if 'train_mae' in model_dict else f"   Training MAE: N/A")
if 'feature_importance' in model_dict:
    print(f"   Top 3 features: {model_dict['feature_importance'].head(3)['feature'].tolist()}")
print()

# ======================================================================
# LOAD DATA
# ======================================================================

print("2. Loading game logs...")
df_all = pd.read_csv(GAME_LOGS_PATH)
df_all['GAME_DATE'] = pd.to_datetime(df_all['GAME_DATE'])

# Filter to 2024-25 season (test set)
df_2024_25 = df_all[
    (df_all['GAME_DATE'] >= '2024-10-22') &
    (df_all['GAME_DATE'] <= '2025-04-30')
].copy()

# Get historical data (for feature calculation)
df_historical = df_all[df_all['GAME_DATE'] < '2024-10-22'].copy()

print(f"   Historical games: {len(df_historical):,}")
print(f"   2024-25 games: {len(df_2024_25):,}")
print(f"   Date range: {df_2024_25['GAME_DATE'].min()} to {df_2024_25['GAME_DATE'].max()}")
print()

# ======================================================================
# LOAD HISTORICAL ODDS (if available)
# ======================================================================

print("3. Loading historical odds...")
try:
    df_odds = pd.read_csv(ODDS_PATH)
    df_odds['event_date'] = pd.to_datetime(df_odds['event_date'])
    print(f"   ✅ Loaded {len(df_odds):,} prop lines")
    has_odds = True
except FileNotFoundError:
    print(f"   ⚠️  No historical odds found at {ODDS_PATH}")
    print(f"   Will proceed with backtest based on model predictions only")
    has_odds = False
print()

# ======================================================================
# WALK-FORWARD VALIDATION
# ======================================================================

print("4. Running walk-forward validation...")
print()

unique_dates = sorted(df_2024_25['GAME_DATE'].unique())
all_predictions = []

for i, pred_date in enumerate(unique_dates, 1):
    # Games to predict today
    games_today = df_2024_25[df_2024_25['GAME_DATE'] == pred_date].copy()

    if len(games_today) == 0:
        continue

    # Historical data up to (but not including) today
    past_games = df_all[df_all['GAME_DATE'] < pred_date].copy()

    # Calculate features for today's games using ONLY past data
    # (This mimics what predict_oct22_FIXED_V2.py will do)

    # For each player in today's games, get their historical stats
    for _, game in games_today.iterrows():
        player_id = game['PLAYER_ID']
        player_name = game['PLAYER_NAME']

        # Get player's past games
        player_past = past_games[past_games['PLAYER_ID'] == player_id].sort_values('GAME_DATE')

        if len(player_past) < 5:
            # Not enough data, skip
            continue

        # Calculate lag features (last N games)
        pra_values = player_past['PRA'].values

        features = {}

        # Lag features
        features['PRA_lag1'] = pra_values[-1] if len(pra_values) >= 1 else np.nan
        features['PRA_lag3'] = pra_values[-3] if len(pra_values) >= 3 else np.nan
        features['PRA_lag5'] = pra_values[-5] if len(pra_values) >= 5 else np.nan
        features['PRA_lag7'] = pra_values[-7] if len(pra_values) >= 7 else np.nan
        features['PRA_lag10'] = pra_values[-10] if len(pra_values) >= 10 else np.nan

        # Rolling means
        features['PRA_L3_mean'] = np.mean(pra_values[-3:]) if len(pra_values) >= 3 else np.nan
        features['PRA_L5_mean'] = np.mean(pra_values[-5:]) if len(pra_values) >= 5 else np.nan
        features['PRA_L10_mean'] = np.mean(pra_values[-10:]) if len(pra_values) >= 10 else np.nan
        features['PRA_L20_mean'] = np.mean(pra_values[-20:]) if len(pra_values) >= 20 else np.nan

        # EWMA (exponentially weighted moving average)
        if len(pra_values) >= 5:
            features['PRA_ewma5'] = pd.Series(pra_values).ewm(span=5).mean().iloc[-1]
        else:
            features['PRA_ewma5'] = np.nan

        if len(pra_values) >= 10:
            features['PRA_ewma10'] = pd.Series(pra_values).ewm(span=10).mean().iloc[-1]
        else:
            features['PRA_ewma10'] = np.nan

        if len(pra_values) >= 15:
            features['PRA_ewma15'] = pd.Series(pra_values).ewm(span=15).mean().iloc[-1]
        else:
            features['PRA_ewma15'] = np.nan

        # Rolling std
        features['PRA_L3_std'] = np.std(pra_values[-3:]) if len(pra_values) >= 3 else np.nan
        features['PRA_L5_std'] = np.std(pra_values[-5:]) if len(pra_values) >= 5 else np.nan
        features['PRA_L10_std'] = np.std(pra_values[-10:]) if len(pra_values) >= 10 else np.nan
        features['PRA_L20_std'] = np.std(pra_values[-20:]) if len(pra_values) >= 20 else np.nan

        # Trend (L5 vs L20)
        if len(pra_values) >= 20:
            features['PRA_trend'] = np.mean(pra_values[-5:]) - np.mean(pra_values[-20:])
        else:
            features['PRA_trend'] = np.nan

        # Minutes projected (L5 average)
        if 'MIN' in player_past.columns and len(player_past) >= 5:
            features['Minutes_Projected'] = player_past['MIN'].iloc[-5:].mean()
        else:
            features['Minutes_Projected'] = np.nan

        # Days rest
        if len(player_past) >= 1:
            days_rest = (pred_date - player_past['GAME_DATE'].iloc[-1]).days
            features['Days_Rest'] = min(days_rest, 7)  # Cap at 7
            features['Is_BackToBack'] = 1 if days_rest <= 1 else 0
        else:
            features['Days_Rest'] = 7
            features['Is_BackToBack'] = 0

        # Games in last 7 days
        last_7_days = player_past[player_past['GAME_DATE'] >= pred_date - pd.Timedelta(days=7)]
        features['Games_Last7'] = len(last_7_days)

        # CTG features (use most recent season stats)
        ctg_cols = ['CTG_USG', 'CTG_PSA', 'CTG_AST_PCT', 'CTG_TOV_PCT', 'CTG_eFG', 'CTG_REB_PCT']
        for col in ctg_cols:
            if col in game:
                features[col] = game[col]
            else:
                features[col] = np.nan

        # Create feature vector
        X_pred = pd.DataFrame([features])

        # Ensure all required features exist
        for col in feature_cols:
            if col not in X_pred.columns:
                X_pred[col] = np.nan

        X_pred = X_pred[feature_cols]

        # Fill missing values (same strategy as training)
        X_pred = X_pred.fillna(0)

        # Make prediction
        pred_pra = model.predict(X_pred)[0]
        actual_pra = game['PRA']

        # Store result
        result = {
            'GAME_DATE': pred_date,
            'PLAYER_ID': player_id,
            'PLAYER_NAME': player_name,
            'TEAM': game.get('TEAM_ABBREVIATION', ''),
            'MATCHUP': game.get('MATCHUP', ''),
            'predicted_PRA': pred_pra,
            'actual_PRA': actual_pra,
            'error': abs(pred_pra - actual_pra),
            'last_game_PRA': features['PRA_lag1'],
            'L5_mean_PRA': features['PRA_L5_mean'],
            'Minutes_Projected': features['Minutes_Projected']
        }

        all_predictions.append(result)

    if (i % 10) == 0 or i == len(unique_dates):
        print(f"   Processed {i}/{len(unique_dates)} dates ({len(all_predictions):,} predictions)")

print()
print(f"   ✅ Walk-forward validation complete")
print(f"   Total predictions: {len(all_predictions):,}")
print()

# ======================================================================
# CALCULATE PERFORMANCE METRICS
# ======================================================================

print("5. Calculating performance metrics...")
print()

df_pred = pd.DataFrame(all_predictions)

# Overall MAE
mae = df_pred['error'].mean()
print(f"   📊 Overall MAE: {mae:.2f} pts")
print()

# Save predictions
df_pred.to_csv(OUTPUT_PATH, index=False)
print(f"   ✅ Predictions saved to {OUTPUT_PATH}")
print()

# ======================================================================
# SIMULATE BETTING (if odds available)
# ======================================================================

if has_odds:
    print("6. Simulating betting strategy...")
    print()

    # Merge predictions with odds
    df_betting = df_pred.merge(
        df_odds,
        left_on=['GAME_DATE', 'PLAYER_NAME'],
        right_on=['event_date', 'player_name'],
        how='inner'
    )

    print(f"   Matched {len(df_betting):,} predictions to odds")

    # Calculate edge
    df_betting['edge'] = df_betting['predicted_PRA'] - df_betting['line']
    df_betting['abs_edge'] = df_betting['edge'].abs()

    # Apply optimal betting strategy (Phase 1: skip 10+ edges)
    optimal_bets = df_betting[
        # Only 5-7 pt edges (10+ edges removed due to overconfidence)
        (df_betting['abs_edge'] >= MIN_EDGE) &
        (df_betting['abs_edge'] <= MAX_EDGE) &
        # Exclude star players
        (~df_betting['PLAYER_NAME'].isin(STAR_PLAYERS))
    ].copy()

    print(f"   Optimal strategy bets: {len(optimal_bets):,}")

    # Determine bet type (OVER/UNDER)
    optimal_bets['bet_type'] = optimal_bets['edge'].apply(lambda x: 'OVER' if x > 0 else 'UNDER')

    # Simulate outcomes
    optimal_bets['bet_correct'] = (
        ((optimal_bets['bet_type'] == 'OVER') & (optimal_bets['actual_PRA'] > optimal_bets['line'])) |
        ((optimal_bets['bet_type'] == 'UNDER') & (optimal_bets['actual_PRA'] < optimal_bets['line']))
    )

    # Calculate win rate
    win_rate = optimal_bets['bet_correct'].mean()
    total_bets = len(optimal_bets)
    wins = optimal_bets['bet_correct'].sum()
    losses = total_bets - wins

    print(f"\n   📊 BETTING RESULTS:")
    print(f"      Total bets: {total_bets:,}")
    print(f"      Wins: {wins:,}")
    print(f"      Losses: {losses:,}")
    print(f"      Win rate: {win_rate*100:.2f}%")

    # Calculate ROI (assuming -110 odds, $100 bets)
    profit_per_win = 100 * (100/110)  # $90.91
    loss_per_loss = -100
    total_profit = (wins * profit_per_win) + (losses * loss_per_loss)
    total_wagered = total_bets * 100
    roi = (total_profit / total_wagered) * 100

    print(f"      Total wagered: ${total_wagered:,}")
    print(f"      Total profit: ${total_profit:,.2f}")
    print(f"      ROI: {roi:.2f}%")
    print()

    # Save betting results
    betting_output = OUTPUT_PATH.replace('.csv', '_betting.csv')
    optimal_bets.to_csv(betting_output, index=False)
    print(f"   ✅ Betting results saved to {betting_output}")
    print()

# ======================================================================
# SUMMARY
# ======================================================================

print("=" * 80)
print("BACKTEST SUMMARY")
print("=" * 80)
print()
print(f"📊 Model Performance:")
print(f"   MAE: {mae:.2f} pts (target: 6-8 pts)")
print(f"   Predictions: {len(df_pred):,}")
print()

if has_odds:
    print(f"💰 Betting Performance (Optimal Strategy):")
    print(f"   Win Rate: {win_rate*100:.2f}% (target: 52-55%)")
    print(f"   ROI: {roi:.2f}% (target: 5-10%)")
    print(f"   Total Bets: {total_bets:,}")
    print()

    validation_status = "✅ PASS" if (0.52 <= win_rate <= 0.58 and 5 <= roi <= 15) else "⚠️  NEEDS REVIEW"
    print(f"🎯 Validation Status: {validation_status}")
else:
    print(f"⚠️  No betting simulation (historical odds not available)")

print()
print("=" * 80)
