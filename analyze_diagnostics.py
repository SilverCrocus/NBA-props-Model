"""
Deep Statistical Analysis of Diagnostic Failures
Analyze root causes of negative ROI despite 54% win rate
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm
from sklearn.metrics import brier_score_loss

# Load ledger
df = pd.read_csv("data/clv_ledger.csv")

# Rename columns to match expected format
df = df.rename(
    columns={
        "won": "win",
        "entry_odds_dec": "entry_odds",
        "result": "actual_pra",
        "player_sigma": "player_sigma",
        "ev_entry": "entry_ev",
    }
)

# Calculate model edge
df["model_edge"] = np.abs(df["predicted_pra"] - df["line"])

# Calculate predicted win probability (using normal CDF)
df["pred_win_prob"] = df.apply(
    lambda row: (
        norm.cdf((row["predicted_pra"] - row["line"]) / row["player_sigma"])
        if row["side"] == "OVER"
        else norm.cdf((row["line"] - row["predicted_pra"]) / row["player_sigma"])
    ),
    axis=1,
)

print("=" * 80)
print("STATISTICAL DEEP DIVE: DIAGNOSTIC FAILURES")
print("=" * 80)

print(f"\nTotal bets: {len(df)}")
print(f'Win rate: {df["win"].mean():.2%}')

print(f'Total profit: {df["profit"].sum():.2f} units')
print(f'Total staked: {df["stake"].sum():.2f} units')
print(f'ROI: {(df["profit"].sum() / df["stake"].sum() * 100):.2f}%')

# Breakeven calculation
avg_implied_prob = (1 / df["entry_odds"]).mean()
print(f"\nBreakeven WR: {avg_implied_prob:.2%}")
print(f'Actual WR: {df["win"].mean():.2%}')
print(f'WR Edge: {(df["win"].mean() - avg_implied_prob):.2%}')

print("\n" + "=" * 80)
print("ROOT CAUSE #1: ODDS IMBALANCE (Winning on Favorites, Losing on Dogs)")
print("=" * 80)

wins = df[df["win"] == 1]
losses = df[df["win"] == 0]

print(f'\nAverage odds on WINS: {wins["entry_odds"].mean():.3f}')
print(f'Average odds on LOSSES: {losses["entry_odds"].mean():.3f}')
print(
    f'Odds imbalance: {
        wins["entry_odds"].mean() -
        losses["entry_odds"].mean():.4f}'
)
print(
    "  → If negative: Winning on shorter odds (favorites), losing on longer odds (dogs)"
)  # noqa: E501
print("  → This DESTROYS ROI even with 54% WR")

print("\n" + "=" * 80)
print("ROOT CAUSE #2: KELLY OVERCONFIDENCE (Heavy Stakes on Losses)")
print("=" * 80)

print(f'\nAverage stake on WINS: {wins["stake"].mean():.2f} units')
print(f'Average stake on LOSSES: {losses["stake"].mean():.2f} units')
print(
    f'Stake imbalance: {
        losses["stake"].mean() -
        wins["stake"].mean():.2f} units'
)
print("  → If positive: We're betting HEAVIER on losses (Kelly overconfidence)")  # noqa: E501

# Weighted win rate
weighted_wr = (wins["stake"].sum()) / df["stake"].sum()
print(f"\nOdds-weighted WR: {weighted_wr:.2%}")
print(f'Traditional WR: {df["win"].mean():.2%}')
print(f'OWWR Gap: {(weighted_wr - df["win"].mean()):.2%}')
print("  → If gap < 0: Heavy staking on losses")

print("\n" + "=" * 80)
print("ROOT CAUSE #3: PROFIT/LOSS ASYMMETRY")
print("=" * 80)

total_wins_profit = (wins["stake"] * (wins["entry_odds"] - 1)).sum()
total_losses_loss = losses["stake"].sum()

print(f"\nTotal profit from {len(wins)} wins: +{total_wins_profit:.2f} units")
print(f"Total loss from {len(losses)} losses: -{total_losses_loss:.2f} units")
print(f'Net: {df["profit"].sum():.2f} units')

avg_win_profit = (wins["stake"] * (wins["entry_odds"] - 1)).mean()
avg_loss = losses["stake"].mean()

print(f"\nAverage profit per WIN: +{avg_win_profit:.2f} units")
print(f"Average loss per LOSS: -{avg_loss:.2f} units")
print(f"Win/Loss ratio: {avg_win_profit / avg_loss:.3f}")
print("  → Need ratio > 0.85 at 54% WR to be profitable")
print("  → Required at 54% WR: 0.46 / 0.54 = 0.852")
print(f"  → Actual: {avg_win_profit / avg_loss:.3f}")

print("\n" + "=" * 80)
print("ROOT CAUSE #4: SIDE BIAS (OVER vs UNDER)")
print("=" * 80)

for side in ["OVER", "UNDER"]:
    side_df = df[df["side"] == side]
    side_wins = side_df[side_df["win"] == 1]
    side_losses = side_df[side_df["win"] == 0]

    print(f"\n{side}:")
    print(f"  Bets: {len(side_df)} ({len(side_df) / len(df):.1%})")
    print(f'  WR: {side_df["win"].mean():.2%}')
    print(f'  ROI: {(side_df["profit"].sum() / side_df["stake"].sum() * 100):.2f}%')  # noqa: E501
    print(f'  Avg edge: {side_df["model_edge"].mean():.2f} pts')
    print(f'  Avg model prob: {side_df["pred_win_prob"].mean():.3f}')
    print(f'  Avg stake: {side_df["stake"].mean():.2f} units')
    print(f'  Total profit: {side_df["profit"].sum():.2f} units')
    print(f'  Avg odds (wins): {side_wins["entry_odds"].mean():.3f}')
    print(
        f'  Avg odds (losses): {side_losses["entry_odds"].mean():.3f}'
        if len(side_losses) > 0
        else "  Avg odds (losses): N/A"
    )

print("\n" + "=" * 80)
print("ROOT CAUSE #5: MODEL EDGE CORRELATION FAILURE")
print("=" * 80)

corr = np.corrcoef(df["model_edge"], df["win"])[0, 1]
print(f"\nModel edge vs win correlation: {corr:.4f}")
print(
    f'Statistical significance: {
        "YES" if abs(corr) > 0.15 else "NO"} (threshold: 0.15)'
)

# Spearman rank correlation
spearman_corr, spearman_p = stats.spearmanr(df["model_edge"], df["win"])
print(f"Spearman correlation: {spearman_corr:.4f} (p={spearman_p:.3f})")

# Edge buckets
bins = [0, 2, 4, 6, 8, np.inf]
labels = ["0 - 2", "2 - 4", "4 - 6", "6 - 8", "8+"]
df["edge_bucket"] = pd.cut(df["model_edge"], bins=bins, labels=labels)

print("\nWin rate by edge bucket (should be monotonically increasing):")
for label in labels:
    bucket_df = df[df["edge_bucket"] == label]
    if len(bucket_df) > 0:
        wr = bucket_df["win"].mean()
        roi = bucket_df["profit"].sum() / bucket_df["stake"].sum() * 100
        print(
            f"  {label} pts: WR={
                wr:.2%}, ROI={
                roi:.2f}%, n={
                len(bucket_df)}"
        )

print("\n" + "=" * 80)
print("ROOT CAUSE #6: CALIBRATION FAILURE")
print("=" * 80)

bins_prob = [0.5, 0.55, 0.60, 0.65, 0.70, 1.0]
labels_prob = ["50 - 55%", "55 - 60%", "60 - 65%", "65 - 70%", "70%+"]
df["prob_bin"] = pd.cut(df["pred_win_prob"], bins=bins_prob, labels=labels_prob)

print("\nPredicted probability vs actual win rate:")
ece = 0
for label in labels_prob:
    bin_df = df[df["prob_bin"] == label]
    if len(bin_df) > 0:
        pred_prob = bin_df["pred_win_prob"].mean()
        actual_wr = bin_df["win"].mean()
        calib_error = abs(pred_prob - actual_wr)
        weight = len(bin_df) / len(df)
        ece += weight * calib_error

        print(
            f"  {label}: Predicted={pred_prob:.1%}, Actual={actual_wr:.1%}, "
            f"n={len(bin_df)}, Error={calib_error:.3f}"
        )

print(f"\nExpected Calibration Error (ECE): {ece:.4f}")
print("  ✓ GOOD if < 0.05    ✗ BAD if > 0.10")

# Brier score (remove NaNs)
df_valid = df.dropna(subset=["win", "pred_win_prob"])
brier = brier_score_loss(df_valid["win"], df_valid["pred_win_prob"])
print(f"Brier Score: {brier:.4f}")
print("  ✓ GOOD if < 0.20    ✗ BAD if > 0.25")

print("\n" + "=" * 80)
print("SIDE-SPECIFIC CALIBRATION ANALYSIS")
print("=" * 80)

for side in ["OVER", "UNDER"]:
    side_df = df[df["side"] == side]
    print(f"\n{side} Calibration:")

    for label in labels_prob:
        bin_df = side_df[side_df["prob_bin"] == label]
        if len(bin_df) > 0:
            pred_prob = bin_df["pred_win_prob"].mean()
            actual_wr = bin_df["win"].mean()
            calib_error = abs(pred_prob - actual_wr)

            print(
                f"  {label}: Predicted={
                    pred_prob:.1%}, Actual={
                    actual_wr:.1%}, "
                f"Error={
                    calib_error:.3f}, n={
                    len(bin_df)}"
            )

    ece_side = 0
    for label in labels_prob:
        bin_df = side_df[side_df["prob_bin"] == label]
        if len(bin_df) > 0:
            pred_prob = bin_df["pred_win_prob"].mean()
            actual_wr = bin_df["win"].mean()
            calib_error = abs(pred_prob - actual_wr)
            weight = len(bin_df) / len(side_df)
            ece_side += weight * calib_error

    print(f"  ECE: {ece_side:.4f}")

print("\n" + "=" * 80)
print("SUMMARY: KEY FINDINGS")
print("=" * 80)

print("\n1. Odds Imbalance:")
print(
    f'   Winning at {wins["entry_odds"].mean():.3f} odds, '
    f'losing at {losses["entry_odds"].mean():.3f} odds'
)
print(f'   Gap: {wins["entry_odds"].mean() - losses["entry_odds"].mean():.4f}')

print("\n2. Kelly Overconfidence:")
print(
    f'   Betting {losses["stake"].mean():.2f} units on losses '
    f'vs {wins["stake"].mean():.2f} on wins'
)
print(
    f'   Imbalance: +{losses["stake"].mean() -
                        wins["stake"].mean():.2f} '
    "units heavier on losses"
)

print("\n3. Side Bias:")
over_wr = df[df["side"] == "OVER"]["win"].mean()
under_wr = df[df["side"] == "UNDER"]["win"].mean()
print(f"   OVER: {over_wr:.2%} WR, UNDER: {under_wr:.2%} WR")
print(f"   Gap: {abs(over_wr - under_wr):.2%}")

print("\n4. Edge Correlation:")
print(f"   Correlation: {corr:.4f} (threshold: 0.15)")
print(f'   Status: {"PASS" if abs(corr) > 0.15 else "FAIL"}')

print("\n5. Calibration:")
print(f"   ECE: {ece:.4f} (threshold: 0.05)")
print(f'   Status: {"PASS" if ece < 0.05 else "FAIL"}')
print(f"   Brier: {brier:.4f} (threshold: 0.20)")
print(f'   Status: {"PASS" if brier < 0.20 else "FAIL"}')

print("\n" + "=" * 80)
