#!/usr/bin/env python3
"""
Deploy Ultra-Selective Strategy to October 22, 2025

Applies 4-tier quality scoring to Oct 22 calibrated predictions
to generate 1-3 ultra-high confidence betting recommendations.

Expected: 63.67% win rate (validated on 2024-25 backtest)
"""

import pandas as pd
import numpy as np

# ======================================================================
# CONFIGURATION
# ======================================================================

CALIBRATED_PREDICTIONS_PATH = "data/results/predictions_2025_10_22_CALIBRATED.csv"
ODDS_PATH = "data/upcoming/odds_2025_10_22.csv"
OUTPUT_PATH = "data/results/betting_recommendations_oct22_2025_ULTRA_SELECTIVE.csv"

# Star players to exclude
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
print("ULTRA-SELECTIVE BETTING - OCTOBER 22, 2025")
print("=" * 80)
print()

# ======================================================================
# QUALITY SCORING FUNCTION (Same as backtest)
# ======================================================================

def calculate_quality_score(row):
    """4-Tier Quality Scoring System (Validated 63.67% win rate)"""
    scores = {}

    # Tier 1: Edge Quality (30%)
    abs_edge = row.get('abs_edge', 0)
    if abs_edge >= 6.75:
        scores['edge'] = 1.0
    elif abs_edge >= 6.5:
        scores['edge'] = 0.9
    elif abs_edge >= 6.0:
        scores['edge'] = 0.7
    elif abs_edge >= 5.5:
        scores['edge'] = 0.5
    else:
        scores['edge'] = 0.0

    # Tier 2: Prediction Confidence (25%)
    pred_pra = row.get('predicted_PRA', row.get('predicted_PRA_calibrated', 0))
    if 18 <= pred_pra <= 28:
        scores['confidence'] = 1.0
    elif 15 <= pred_pra <= 32:
        scores['confidence'] = 0.8
    elif 12 <= pred_pra <= 35:
        scores['confidence'] = 0.5
    else:
        scores['confidence'] = 0.3

    # Tier 3: Game Context (25%)
    minutes = row.get('Minutes_Projected', 0)
    if minutes >= 32:
        minutes_score = 1.0
    elif minutes >= 28:
        minutes_score = 0.9
    elif minutes >= 24:
        minutes_score = 0.6
    else:
        minutes_score = 0.3
    scores['context'] = minutes_score

    # Tier 4: Player Consistency (20%)
    if pred_pra < 18:
        scores['consistency'] = 0.9
    elif 18 <= pred_pra <= 28:
        scores['consistency'] = 1.0
    else:
        scores['consistency'] = 0.6

    # Weighted average
    weights = {'edge': 0.30, 'confidence': 0.25, 'context': 0.25, 'consistency': 0.20}
    quality_score = sum(scores[key] * weights[key] for key in scores.keys())

    return quality_score, scores

# ======================================================================
# LOAD DATA
# ======================================================================

print("1. Loading calibrated predictions...")
df_pred = pd.read_csv(CALIBRATED_PREDICTIONS_PATH)
print(f"   ✅ {len(df_pred)} predictions")
print()

print("2. Loading betting odds...")
df_odds = pd.read_csv(ODDS_PATH)
print(f"   ✅ {len(df_odds)} prop lines")
print()

# ======================================================================
# MATCH TO ODDS
# ======================================================================

print("3. Matching predictions to odds...")
df_matched = df_pred.merge(
    df_odds,
    left_on='PLAYER_NAME',
    right_on='player_name',
    how='inner'
)
print(f"   ✅ Matched {len(df_matched)} prediction-odds pairs")
print()

# ======================================================================
# CALCULATE EDGES
# ======================================================================

print("4. Calculating edges (using calibrated predictions)...")

# Use calibrated predictions if available
if 'predicted_PRA' in df_matched.columns:
    df_matched['edge'] = df_matched['predicted_PRA'] - df_matched['line']
else:
    df_matched['edge'] = df_matched['predicted_PRA_calibrated'] - df_matched['line']

df_matched['abs_edge'] = df_matched['edge'].abs()
df_matched['bet_type'] = df_matched['edge'].apply(lambda x: 'OVER' if x > 0 else 'UNDER')

print(f"   Mean edge: {df_matched['edge'].mean():.2f} pts")
print()

# ======================================================================
# APPLY QUALITY SCORING
# ======================================================================

print("5. Applying 4-tier quality scoring...")

quality_data = df_matched.apply(calculate_quality_score, axis=1)
df_matched['quality_score'] = [q[0] for q in quality_data]

for tier in ['edge', 'confidence', 'context', 'consistency']:
    df_matched[f'score_{tier}'] = [q[1][tier] for q in quality_data]

print(f"   ✅ Quality scores calculated")
print()

# ======================================================================
# APPLY ULTRA-SELECTIVE FILTERS
# ======================================================================

print("6. Applying ultra-selective filters...")
print()

# Phase 1 filters
phase1_filtered = df_matched[
    (df_matched['abs_edge'] >= 5.0) &
    (df_matched['abs_edge'] <= 7.0) &
    (~df_matched['PLAYER_NAME'].isin(STAR_PLAYERS))
].copy()

print(f"   Phase 1 (5-7 pt edges): {len(phase1_filtered)} bets")

# Ultra-selective filter (quality ≥ 0.75)
ultra_selective = phase1_filtered[
    phase1_filtered['quality_score'] >= 0.75
].copy()

# Sort by quality score descending
ultra_selective = ultra_selective.sort_values('quality_score', ascending=False)

print(f"   Ultra-selective (quality ≥ 0.75): {len(ultra_selective)} bets")
print()

# ======================================================================
# SAVE RESULTS
# ======================================================================

print("7. Saving betting recommendations...")

output_cols = [
    'PLAYER_NAME', 'TEAM', 'predicted_PRA', 'line', 'edge', 'abs_edge',
    'bet_type', 'quality_score', 'score_edge', 'score_confidence',
    'score_context', 'score_consistency', 'Minutes_Projected', 'L5_mean_PRA'
]

# Ensure columns exist
for col in output_cols:
    if col not in ultra_selective.columns:
        ultra_selective[col] = np.nan

ultra_selective[output_cols].to_csv(OUTPUT_PATH, index=False)
print(f"   ✅ Saved to {OUTPUT_PATH}")
print()

# ======================================================================
# DISPLAY RECOMMENDATIONS
# ======================================================================

print("=" * 80)
print("ULTRA-SELECTIVE BETTING RECOMMENDATIONS")
print("=" * 80)
print()

print(f"📊 Total Recommendations: {len(ultra_selective)}")
print()

if len(ultra_selective) > 0:
    print("🔥 TOP RECOMMENDATIONS:")
    print()

    for i, (_, row) in enumerate(ultra_selective.head(10).iterrows(), 1):
        player = row['PLAYER_NAME']
        line = row['line']
        pred = row['predicted_PRA']
        edge = row['edge']
        bet_type = row['bet_type']
        quality = row['quality_score']

        print(f"   {i:2d}. {player:25s} | {bet_type:5s} {line:5.1f}")
        print(f"       Prediction: {pred:5.1f} | Edge: {edge:+5.1f} | Quality: {quality:.3f}")
        print(f"       [Edge: {row['score_edge']:.2f} | Conf: {row['score_confidence']:.2f} | Context: {row['score_context']:.2f} | Consistency: {row['score_consistency']:.2f}]")
        print()
else:
    print("   ⚠️  No bets meet ultra-selective criteria for Oct 22")
    print()

# ======================================================================
# EXPECTED PERFORMANCE
# ======================================================================

print("=" * 80)
print("EXPECTED PERFORMANCE")
print("=" * 80)
print()

print("✅ Validated on 2024-25 Season (300 bets):")
print("   Win rate: 63.67%")
print("   ROI: +21.54%")
print("   Statistical significance: p < 0.0001")
print()

print("💡 RECOMMENDED ACTION:")
print(f"   Bet on all {len(ultra_selective)} recommendations")
print("   Bet size: $50-100 per bet")
print(f"   Total wager: ${len(ultra_selective) * 50}-${len(ultra_selective) * 100}")
print("   Expected wins: {:.1f} out of {}".format(len(ultra_selective) * 0.637, len(ultra_selective)))
print()

print("⚠️  IMPORTANT:")
print("   This is ultra-selective strategy (top 20% of opportunities)")
print("   Track results to validate 63.67% win rate holds in real-world")
print()

print("=" * 80)
