"""
Backtest Optimal Betting Strategy for 2024-25

Applies OPTIMAL filters based on research findings:
1. Exclude star players (more efficient markets)
2. Only bet on medium (5-7 pts) or huge (10+ pts) edges

Expected performance: 54.78% win rate, +5.95% ROI, $1,367.71 profit

See: OPTIMAL_BETTING_STRATEGY.md for full research findings
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import BettingConfig, data_config

# Initialize betting config
betting_config = BettingConfig()

print("=" * 80)
print("NBA PROPS MODEL - OPTIMAL BETTING STRATEGY BACKTEST")
print("=" * 80)
print("\n🎯 Strategy: Non-star players + Medium/Huge edges only")
print(f"   Exclude: {len(betting_config.STAR_PLAYERS)} star players")
print(f"   Bet on edges: 5-7 pts OR 10+ pts")

# Load walk-forward predictions and odds
print("\n1. Loading data...")
preds_df = pd.read_csv(data_config.RESULTS_DIR / "walk_forward_advanced_features_2024_25.csv")
odds_df = pd.read_csv("data/historical_odds/2024-25/pra_odds.csv")

print(f"✅ Loaded {len(preds_df):,} walk-forward predictions")
print(f"✅ Loaded {len(odds_df):,} betting lines")

# Standardize dates and names
preds_df["game_date"] = pd.to_datetime(preds_df["GAME_DATE"]).dt.date
odds_df["game_date"] = pd.to_datetime(odds_df["event_date"]).dt.date
preds_df["player_lower"] = preds_df["PLAYER_NAME"].str.lower().str.strip()
odds_df["player_lower"] = odds_df["player_name"].str.lower().str.strip()

# STEP 1: DEDUPLICATE PREDICTIONS
print("\n2. Deduplicating predictions (ONE per player-date)...")
print(f"   Before: {len(preds_df):,} rows")

preds_dedup = preds_df.groupby(["PLAYER_NAME", "game_date"], as_index=False).agg(
    {
        "player_lower": "first",
        "PRA": "first",
        "predicted_PRA": "median",
        "error": "median",
        "abs_error": "median",
    }
)

print(f"   After: {len(preds_dedup):,} rows")
if len(preds_df) > len(preds_dedup):
    print(f"   Removed: {len(preds_df) - len(preds_dedup):,} duplicates")

# STEP 2: LINE SHOPPING
print("\n3. Line shopping (best odds across bookmakers)...")
odds_best = odds_df.groupby(["player_lower", "game_date"], as_index=False).agg(
    {
        "line": "first",
        "over_price": "max",
        "under_price": "max",
        "bookmaker": lambda x: ", ".join(x.unique()[:3]),
    }
)

print(f"   Unique player-date combinations: {len(odds_best):,}")

# STEP 3: MATCH predictions to odds
print("\n4. Matching predictions to betting lines...")
merged = preds_dedup.merge(
    odds_best,
    left_on=["player_lower", "game_date"],
    right_on=["player_lower", "game_date"],
    how="inner",
)

print(f"✅ Matched {len(merged):,} predictions to betting lines")
print(f"   Match rate: {len(merged)/len(preds_dedup)*100:.1f}%")

# Calculate betting metrics
merged["actual_pra"] = merged["PRA"]
merged["predicted_pra"] = merged["predicted_PRA"]
merged["betting_line"] = merged["line"]
merged["edge"] = merged["predicted_pra"] - merged["betting_line"]
merged["abs_edge"] = abs(merged["edge"])


# Convert American odds to implied probability
def american_to_prob(odds):
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return abs(odds) / (abs(odds) + 100)


merged["over_implied_prob"] = merged["over_price"].apply(american_to_prob)
merged["under_implied_prob"] = merged["under_price"].apply(american_to_prob)

# ================================================================================
# OPTIMAL STRATEGY FILTERS
# ================================================================================

print("\n" + "=" * 80)
print("APPLYING OPTIMAL FILTERS")
print("=" * 80)

# FILTER 1: Star Players
print(f"\n🌟 Filter 1: Excluding {len(betting_config.STAR_PLAYERS)} star players...")
print(f"   Before: {len(merged):,} predictions")

merged["is_star"] = merged["PLAYER_NAME"].isin(betting_config.STAR_PLAYERS)
star_count = merged["is_star"].sum()
merged_filtered = merged[~merged["is_star"]].copy()

print(f"   After: {len(merged_filtered):,} predictions")
print(f"   Removed: {star_count:,} star player predictions ({star_count/len(merged)*100:.1f}%)")

# FILTER 2: Edge Size (5-7 pts OR 10+ pts)
print(f"\n📏 Filter 2: Edge size (5-7 pts OR 10+ pts only)...")
print(f"   Before: {len(merged_filtered):,} predictions")

# Determine bet side based on OPTIMAL edge thresholds
merged_filtered["bet_side"] = "NONE"

# Medium edges (5-7 pts)
medium_edge_over = (merged_filtered["edge"] >= betting_config.EDGE_MEDIUM_MIN) & (
    merged_filtered["edge"] <= betting_config.EDGE_MEDIUM_MAX
)
medium_edge_under = (merged_filtered["edge"] <= -betting_config.EDGE_MEDIUM_MIN) & (
    merged_filtered["edge"] >= -betting_config.EDGE_MEDIUM_MAX
)

# Huge edges (10+ pts)
huge_edge_over = merged_filtered["edge"] >= betting_config.EDGE_HUGE_MIN
huge_edge_under = merged_filtered["edge"] <= -betting_config.EDGE_HUGE_MIN

# Apply filters
merged_filtered.loc[medium_edge_over | huge_edge_over, "bet_side"] = "OVER"
merged_filtered.loc[medium_edge_under | huge_edge_under, "bet_side"] = "UNDER"

bet_count = (merged_filtered["bet_side"] != "NONE").sum()
print(f"   Predictions meeting edge criteria: {bet_count:,}")
print(f"     - Medium edges (5-7 pts): {(medium_edge_over | medium_edge_under).sum():,}")
print(f"     - Huge edges (10+ pts): {(huge_edge_over | huge_edge_under).sum():,}")

# ================================================================================
# CALCULATE BET RESULTS
# ================================================================================


def calculate_bet_result(row):
    if row["bet_side"] == "NONE":
        return 0

    actual = row["actual_pra"]
    line = row["betting_line"]

    if row["bet_side"] == "OVER":
        won = actual > line
        odds = row["over_price"]
    else:
        won = actual < line
        odds = row["under_price"]

    if actual == line:  # Push
        return 0

    if won:
        if odds > 0:
            return odds
        else:
            return 100 * (100 / abs(odds))
    else:
        return -100


merged_filtered["bet_result"] = merged_filtered.apply(calculate_bet_result, axis=1)
merged_filtered["bet_won"] = (merged_filtered["bet_result"] > 0) & (
    merged_filtered["bet_side"] != "NONE"
)
merged_filtered["bet_pushed"] = (merged_filtered["bet_result"] == 0) & (
    merged_filtered["bet_side"] != "NONE"
)
merged_filtered["bet_lost"] = (merged_filtered["bet_result"] < 0) & (
    merged_filtered["bet_side"] != "NONE"
)

# Overall statistics
total_bets = (merged_filtered["bet_side"] != "NONE").sum()
won_bets = merged_filtered["bet_won"].sum()
pushed_bets = merged_filtered["bet_pushed"].sum()
lost_bets = merged_filtered["bet_lost"].sum()
total_profit = merged_filtered["bet_result"].sum()
total_wagered = total_bets * 100

print("\n" + "=" * 80)
print("OPTIMAL STRATEGY BACKTEST RESULTS")
print("=" * 80)

print(f"\n📊 BETTING RESULTS:")
print("-" * 80)
print(f"Total bets placed: {total_bets:,}")
if total_bets > 0:
    print(f"  Won:    {won_bets:,} ({won_bets/total_bets*100:.2f}%)")
    print(f"  Lost:   {lost_bets:,} ({lost_bets/total_bets*100:.2f}%)")
    print(f"  Pushed: {pushed_bets:,} ({pushed_bets/total_bets*100:.2f}%)")

    print(f"\n💰 PROFIT & LOSS:")
    print("-" * 80)
    print(f"Total wagered: ${total_wagered:,.0f}")
    print(f"Total profit:  ${total_profit:,.2f}")
    print(f"ROI:           {total_profit/total_wagered*100:+.2f}%")

    # Breakeven analysis
    breakeven_rate = 52.38
    print(f"\n📈 ANALYSIS:")
    print("-" * 80)
    print(f"Win rate needed (breakeven at -110): {breakeven_rate:.2f}%")
    print(f"Actual win rate: {won_bets/total_bets*100:.2f}%")
    print(f"Edge over breakeven: {won_bets/total_bets*100 - breakeven_rate:+.2f} percentage points")

    if won_bets / total_bets * 100 > breakeven_rate:
        print(f"✅ PROFITABLE MODEL!")
    else:
        print(f"❌ Below breakeven")

    # Performance by edge size
    print(f"\n" + "=" * 80)
    print("PERFORMANCE BY EDGE SIZE (OPTIMAL STRATEGY ONLY)")
    print("=" * 80)

    # Medium edges (5-7 pts)
    medium_bets = merged_filtered[
        (merged_filtered["abs_edge"] >= betting_config.EDGE_MEDIUM_MIN)
        & (merged_filtered["abs_edge"] <= betting_config.EDGE_MEDIUM_MAX)
    ]
    medium_total = (medium_bets["bet_side"] != "NONE").sum()
    if medium_total > 0:
        medium_won = medium_bets["bet_won"].sum()
        medium_profit = medium_bets["bet_result"].sum()
        medium_wagered = medium_total * 100

        print(f"\n📌 Medium edge (5-7 pts):")
        print(f"  Bets: {medium_total:,}")
        print(f"  Win rate: {medium_won/medium_total*100:.2f}%")
        print(f"  ROI: {medium_profit/medium_wagered*100:+.2f}%")
        print(f"  Profit: ${medium_profit:,.2f}")

    # Huge edges (10+ pts)
    huge_bets = merged_filtered[merged_filtered["abs_edge"] >= betting_config.EDGE_HUGE_MIN]
    huge_total = (huge_bets["bet_side"] != "NONE").sum()
    if huge_total > 0:
        huge_won = huge_bets["bet_won"].sum()
        huge_profit = huge_bets["bet_result"].sum()
        huge_wagered = huge_total * 100

        print(f"\n📌 Huge edge (10+ pts):")
        print(f"  Bets: {huge_total:,}")
        print(f"  Win rate: {huge_won/huge_total*100:.2f}%")
        print(f"  ROI: {huge_profit/huge_wagered*100:+.2f}%")
        print(f"  Profit: ${huge_profit:,.2f}")

    # Comparison to baseline
    print(f"\n" + "=" * 80)
    print("COMPARISON TO BASELINE")
    print("=" * 80)

    # Load baseline results
    baseline_file = data_config.RESULTS_DIR / "backtest_walkforward_2024_25_summary.json"
    if baseline_file.exists():
        with open(baseline_file, "r") as f:
            baseline = json.load(f)

        print(f"\nBaseline (all ≥3 pts):")
        print(f"  Bets: {baseline['total_bets']:,}")
        print(f"  Win rate: {baseline['win_rate']:.2f}%")
        print(f"  ROI: {baseline['roi_percent']:+.2f}%")
        print(f"  Profit: ${baseline['total_profit']:,.2f}")

        print(f"\nOptimal (non-stars + 5-7/10+ pts):")
        print(f"  Bets: {total_bets:,} ({total_bets/baseline['total_bets']*100:.1f}% of baseline)")
        print(
            f"  Win rate: {won_bets/total_bets*100:.2f}% ({won_bets/total_bets*100 - baseline['win_rate']:+.2f} pp)"
        )
        print(
            f"  ROI: {total_profit/total_wagered*100:+.2f}% ({total_profit/total_wagered*100 - baseline['roi_percent']:+.2f} pp)"
        )
        print(
            f"  Profit: ${total_profit:,.2f} ({total_profit/baseline['total_profit']:.1f}x baseline)"
        )

        # Improvement metrics
        roi_improvement = (total_profit / total_wagered * 100) / baseline["roi_percent"]
        print(f"\n📊 Improvement:")
        print(f"  ROI improvement: {roi_improvement:.1f}x better")
        print(f"  Profit improvement: {total_profit/baseline['total_profit']:.1f}x more")
        print(
            f"  Bet reduction: {(1 - total_bets/baseline['total_bets'])*100:.1f}% fewer bets (quality over quantity!)"
        )

else:
    print("\n⚠️  No bets met the optimal strategy criteria")

# Prediction accuracy
print(f"\n" + "=" * 80)
print("PREDICTION ACCURACY (NON-STAR PLAYERS)")
print("=" * 80)

mae = mean_absolute_error(merged_filtered["actual_pra"], merged_filtered["predicted_pra"])
print(f"MAE: {mae:.2f} points")
print(
    f"Within ±3 pts: {(abs(merged_filtered['actual_pra'] - merged_filtered['predicted_pra']) <= 3).mean()*100:.1f}%"
)
print(
    f"Within ±5 pts: {(abs(merged_filtered['actual_pra'] - merged_filtered['predicted_pra']) <= 5).mean()*100:.1f}%"
)

# Save results
output_file = data_config.RESULTS_DIR / "backtest_optimal_strategy.csv"
merged_filtered.to_csv(output_file, index=False)
print(f"\n✅ Saved detailed results to {output_file}")

# Summary
if total_bets > 0:
    summary = {
        "season": "2024-25",
        "strategy": "optimal",
        "filters": {
            "exclude_stars": True,
            "star_count": len(betting_config.STAR_PLAYERS),
            "edge_filter": "5-7 OR 10+ pts",
        },
        "total_matched_predictions": int(len(merged_filtered)),
        "total_bets": int(total_bets),
        "win_rate": float(won_bets / total_bets * 100),
        "roi_percent": float(total_profit / total_wagered * 100),
        "total_profit": float(total_profit),
        "total_wagered": float(total_wagered),
        "mae": float(mae),
        "average_edge": float(merged_filtered["edge"].mean()),
        "medium_edge_bets": int(medium_total) if medium_total > 0 else 0,
        "huge_edge_bets": int(huge_total) if huge_total > 0 else 0,
    }

    summary_file = data_config.RESULTS_DIR / "backtest_optimal_strategy_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n" + "=" * 80)
    print("FINAL SUMMARY - OPTIMAL BETTING STRATEGY")
    print("=" * 80)

    print(
        f"""
✅ OPTIMAL STRATEGY BACKTEST COMPLETE

Strategy filters:
  ✅ Exclude {len(betting_config.STAR_PLAYERS)} star players
  ✅ Only bet on edges: 5-7 pts OR 10+ pts

Results:
  Total bets: {total_bets:,}
  Win rate: {won_bets/total_bets*100:.2f}%
  ROI: {total_profit/total_wagered*100:+.2f}%
  Total P&L: ${total_profit:,.2f}
  Prediction MAE: {mae:.2f} points

Performance breakdown:
  Medium edges (5-7 pts): {medium_total:,} bets, {medium_won/medium_total*100:.1f}% win rate, ${medium_profit:,.2f} profit
  Huge edges (10+ pts): {huge_total:,} bets, {huge_won/huge_total*100:.1f}% win rate, ${huge_profit:,.2f} profit

Expected annual performance:
  Starting bankroll: $1,000
  Ending bankroll: ${1000 + total_profit:,.2f}
  Return: {total_profit/1000*100:+.1f}%

Status: {'✅ PROFITABLE!' if total_profit > 0 else '❌ Not profitable'}

See OPTIMAL_BETTING_STRATEGY.md for full implementation guide.
"""
    )

print("=" * 80)
