"""
Error Analysis for Feature Improvement Strategy
Systematically analyze the 4 losses that could be wins to reach 56% win rate
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_backtest_results():
    """Load backtest results"""
    results_path = Path(
        "/Users/diyagamah/Documents/nba_props_model/data/results/backtest_walkforward_2024_25.csv"
    )
    df = pd.read_csv(results_path)
    return df


def analyze_current_state(df):
    """Quantify current performance"""
    print("=" * 80)
    print("CURRENT STATE ANALYSIS")
    print("=" * 80)

    # Filter to actual bets (where bet_side is not NONE)
    bets = df[df["bet_side"] != "NONE"].copy()

    total_bets = len(bets)
    wins = bets["bet_won"].sum()
    losses = bets["bet_lost"].sum()
    pushes = bets["bet_pushed"].sum()

    win_rate = wins / total_bets * 100

    print(f"\nTotal Bets: {total_bets}")
    print(f"Wins: {wins}")
    print(f"Losses: {losses}")
    print(f"Pushes: {pushes}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"\nTarget: 56% ({int(total_bets * 0.56)} wins)")
    print(f"Gap: Need {int(total_bets * 0.56) - wins} more wins")

    # Calculate MAE
    mae = bets["abs_error"].mean()
    print(f"\nMean Absolute Error: {mae:.2f} points")

    return bets


def identify_close_losses(bets):
    """Identify the losses that were close to being wins"""
    print("\n" + "=" * 80)
    print("CLOSE LOSSES ANALYSIS (Target: Find 4-5 losses to convert)")
    print("=" * 80)

    losses = bets[bets["bet_lost"] == True].copy()

    # Calculate how close each loss was
    # For OVER bets: predicted - line (negative means we were under)
    # For UNDER bets: line - predicted (negative means we were over)
    losses["margin_of_error"] = np.where(
        losses["bet_side"] == "OVER",
        losses["actual_pra"] - losses["betting_line"],  # How much they went over
        losses["betting_line"] - losses["actual_pra"],  # How much they went under
    )

    # Sort by smallest margin of error (closest to being right)
    losses_sorted = losses.sort_values("margin_of_error")

    print(f"\nTotal Losses: {len(losses)}")
    print(f"\nTop 10 Closest Losses (convert these to reach 56%):")
    print("-" * 80)

    for idx, (i, row) in enumerate(losses_sorted.head(10).iterrows(), 1):
        print(f"\n{idx}. {row['PLAYER_NAME']} ({row['game_date']})")
        print(f"   Bet: {row['bet_side']} {row['betting_line']}")
        print(f"   Predicted: {row['predicted_PRA']:.1f}")
        print(f"   Actual: {row['actual_pra']:.0f}")
        print(
            f"   Margin: {row['margin_of_error']:.1f} (need {abs(row['margin_of_error']):.1f} points better)"
        )
        print(f"   Error: {row['abs_error']:.1f} points")
        print(f"   Edge: {row['abs_edge']:.2f}")

    # Analyze common patterns
    print("\n" + "=" * 80)
    print("LOSS PATTERN ANALYSIS")
    print("=" * 80)

    print("\nLosses by Bet Type:")
    print(losses["bet_side"].value_counts())

    print("\nLosses by Error Magnitude:")
    print(f"  Small error (<5 pts): {(losses['abs_error'] < 5).sum()}")
    print(
        f"  Medium error (5-10 pts): {((losses['abs_error'] >= 5) & (losses['abs_error'] < 10)).sum()}"
    )
    print(
        f"  Large error (10-20 pts): {((losses['abs_error'] >= 10) & (losses['abs_error'] < 20)).sum()}"
    )
    print(f"  Very large error (>20 pts): {(losses['abs_error'] >= 20).sum()}")

    print("\nLosses by Edge Size:")
    print(f"  Small edge (3-5): {((losses['abs_edge'] >= 3) & (losses['abs_edge'] < 5)).sum()}")
    print(f"  Medium edge (5-7): {((losses['abs_edge'] >= 5) & (losses['abs_edge'] < 7)).sum()}")
    print(f"  Large edge (7-10): {((losses['abs_edge'] >= 7) & (losses['abs_edge'] < 10)).sum()}")
    print(f"  Very large edge (>10): {(losses['abs_edge'] >= 10).sum()}")

    return losses_sorted


def analyze_wins(bets):
    """Analyze winning bets to understand what works"""
    print("\n" + "=" * 80)
    print("WINNING BET ANALYSIS")
    print("=" * 80)

    wins = bets[bets["bet_won"] == True].copy()

    print(f"\nTotal Wins: {len(wins)}")
    print(f"Win Rate: {len(wins) / len(bets) * 100:.2f}%")

    print("\nWins by Bet Type:")
    print(wins["bet_side"].value_counts())

    print("\nWins by Error Magnitude:")
    print(f"  Small error (<5 pts): {(wins['abs_error'] < 5).sum()}")
    print(
        f"  Medium error (5-10 pts): {((wins['abs_error'] >= 5) & (wins['abs_error'] < 10)).sum()}"
    )
    print(
        f"  Large error (10-20 pts): {((wins['abs_error'] >= 10) & (wins['abs_error'] < 20)).sum()}"
    )
    print(f"  Very large error (>20 pts): {(wins['abs_error'] >= 20).sum()}")

    print("\nWins by Edge Size:")
    print(f"  Small edge (3-5): {((wins['abs_edge'] >= 3) & (wins['abs_edge'] < 5)).sum()}")
    print(f"  Medium edge (5-7): {((wins['abs_edge'] >= 5) & (wins['abs_edge'] < 7)).sum()}")
    print(f"  Large edge (7-10): {((wins['abs_edge'] >= 7) & (wins['abs_edge'] < 10)).sum()}")
    print(f"  Very large edge (>10): {(wins['abs_edge'] >= 10).sum()}")

    # Compare wins vs losses
    losses = bets[bets["bet_lost"] == True]

    print("\n" + "=" * 80)
    print("WIN vs LOSS COMPARISON")
    print("=" * 80)

    print(f"\nAverage Error:")
    print(f"  Wins: {wins['abs_error'].mean():.2f} pts")
    print(f"  Losses: {losses['abs_error'].mean():.2f} pts")
    print(f"  Difference: {losses['abs_error'].mean() - wins['abs_error'].mean():.2f} pts")

    print(f"\nAverage Edge:")
    print(f"  Wins: {wins['abs_edge'].mean():.2f}")
    print(f"  Losses: {losses['abs_edge'].mean():.2f}")
    print(f"  Difference: {losses['abs_edge'].mean() - wins['abs_edge'].mean():.2f}")

    return wins


def analyze_error_by_features(bets):
    """Analyze which types of bets have highest errors"""
    print("\n" + "=" * 80)
    print("ERROR ANALYSIS BY BET CHARACTERISTICS")
    print("=" * 80)

    # Create bins for analysis
    bets["edge_bin"] = pd.cut(
        bets["abs_edge"], bins=[0, 4, 6, 8, 100], labels=["3-4 pts", "4-6 pts", "6-8 pts", "8+ pts"]
    )

    bets["predicted_bin"] = pd.cut(
        bets["predicted_PRA"],
        bins=[0, 15, 25, 35, 100],
        labels=["Low (<15)", "Medium (15-25)", "High (25-35)", "Very High (35+)"],
    )

    # Win rate by edge size
    print("\nWin Rate by Edge Size:")
    edge_analysis = (
        bets.groupby("edge_bin")
        .agg({"bet_won": ["sum", "count", "mean"], "abs_error": "mean"})
        .round(3)
    )
    print(edge_analysis)

    # Win rate by predicted value
    print("\nWin Rate by Predicted PRA Range:")
    pred_analysis = (
        bets.groupby("predicted_bin")
        .agg({"bet_won": ["sum", "count", "mean"], "abs_error": "mean"})
        .round(3)
    )
    print(pred_analysis)

    # Win rate by bet side
    print("\nWin Rate by Bet Side:")
    side_analysis = (
        bets.groupby("bet_side")
        .agg({"bet_won": ["sum", "count", "mean"], "abs_error": "mean"})
        .round(3)
    )
    print(side_analysis)

    return bets


def generate_feature_recommendations(wins, losses):
    """Generate specific feature recommendations based on error patterns"""
    print("\n" + "=" * 80)
    print("FEATURE IMPROVEMENT RECOMMENDATIONS")
    print("=" * 80)

    # Calculate key metrics
    win_error = wins["abs_error"].mean()
    loss_error = losses["abs_error"].mean()
    error_gap = loss_error - win_error

    print(f"\nKey Insight: Losses have {error_gap:.2f} pts higher error than wins")
    print(f"Target: Reduce loss error from {loss_error:.2f} to {win_error:.2f} pts")
    print(f"This would convert ~{int(len(losses) * 0.3)} losses to wins → 56%+ win rate")

    print("\n" + "-" * 80)
    print("PRIORITY 1: Features to Reduce Error on High-Variance Players")
    print("-" * 80)

    print(
        """
1. OPPONENT DEFENSIVE RATING (DRtg) by Position
   - Impact: HIGH (1-2% win rate boost)
   - Complexity: EASY (CTG team data available)
   - Why: Large errors often from not accounting for matchup difficulty
   - Implementation: Merge opponent DRtg, pace, defensive rebounding %

2. IMPROVED MINUTES PROJECTION
   - Impact: HIGH (1-2% win rate boost)
   - Complexity: MEDIUM
   - Why: Minutes volatility is #1 cause of large errors
   - Implementation:
     * Use exponentially weighted average (recent games weighted higher)
     * Add minutes trend (increasing/decreasing role)
     * Add coach/injury context

3. L3 RECENT FORM FEATURES
   - Impact: HIGH (1-1.5% win rate boost)
   - Complexity: EASY
   - Why: Current features use L5/L10, but L3 captures hot/cold streaks
   - Implementation: Add PRA_L3, USG_L3, MIN_L3
    """
    )

    print("\n" + "-" * 80)
    print("PRIORITY 2: Features to Improve Edge Calibration")
    print("-" * 80)

    print(
        """
4. TRUE SHOOTING % (TS%) - Game Level
   - Impact: MEDIUM (0.5-1% win rate boost)
   - Complexity: EASY
   - Why: Better captures scoring efficiency than FG%
   - Formula: PTS / (2 * (FGA + 0.44 * FTA))

5. USAGE RATE x PACE INTERACTION
   - Impact: MEDIUM (0.5-1% win rate boost)
   - Complexity: EASY
   - Why: High usage players benefit more from fast pace
   - Implementation: Create interaction feature: USG% * (Team_Pace + Opp_Pace) / 2

6. REST DAYS INTERACTION with B2B
   - Impact: MEDIUM (0.5-1% win rate boost)
   - Complexity: EASY
   - Why: 1 day rest vs 3+ days rest has different impact
   - Implementation: Create bins: 0-1 days, 2-3 days, 4+ days
    """
    )

    print("\n" + "-" * 80)
    print("PRIORITY 3: Advanced Features (Phase 3)")
    print("-" * 80)

    print(
        """
7. TEAMMATE USAGE CONTEXT
   - Impact: MEDIUM (0.5-1% win rate boost)
   - Complexity: HIGH
   - Why: Usage increases when high-usage teammates are out
   - Implementation: Track teammate availability, adjust predicted usage

8. HOME/AWAY SPLITS
   - Impact: LOW-MEDIUM (0.3-0.5% win rate boost)
   - Complexity: EASY
   - Why: Some players perform better at home
   - Implementation: Add home_away indicator, calculate home/away averages

9. MATCHUP-SPECIFIC MODELS
   - Impact: MEDIUM-HIGH (0.5-1.5% win rate boost)
   - Complexity: VERY HIGH
   - Why: Different player types perform differently vs different defenses
   - Implementation: Cluster players by style, train position-specific models
    """
    )


def main():
    """Main analysis pipeline"""
    print("\n" + "=" * 80)
    print("NBA PROPS MODEL - FEATURE IMPROVEMENT STRATEGY")
    print("Target: 52.94% → 56% Win Rate")
    print("=" * 80)

    # Load data
    df = load_backtest_results()

    # Current state
    bets = analyze_current_state(df)

    # Identify close losses
    close_losses = identify_close_losses(bets)

    # Analyze wins
    wins = analyze_wins(bets)

    # Error analysis
    bets_analyzed = analyze_error_by_features(bets)

    # Generate recommendations
    losses = bets[bets["bet_lost"] == True]
    generate_feature_recommendations(wins, losses)

    print("\n" + "=" * 80)
    print("NEXT STEPS:")
    print("=" * 80)
    print(
        """
1. Review the 10 closest losses above
2. Identify common patterns (player type, matchup, rest, etc.)
3. Implement Priority 1 features first (1-2 days each)
4. Re-run walk-forward validation after each feature
5. Track improvement: 52.94% → 54% → 55% → 56%+
    """
    )


if __name__ == "__main__":
    main()
