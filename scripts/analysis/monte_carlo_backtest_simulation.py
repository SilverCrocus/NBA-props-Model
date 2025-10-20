#!/usr/bin/env python3
"""
Monte Carlo Simulation of Backtest Results

Takes the actual bet outcomes and randomly reorders them to test:
1. Is profitability due to skill or lucky bet ordering?
2. What's the probability of going bust?
3. What's the expected value and variance?
4. Is the Kelly Criterion sizing appropriate?

This simulates 10,000 different bet orderings to understand:
- If we're just lucky (order matters a lot) vs skilled (order doesn't matter)
- True expected return and confidence intervals
- Risk of ruin
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

print("=" * 80)
print("MONTE CARLO SIMULATION - Bet Reordering Analysis")
print("=" * 80)
print()

# Configuration
STARTING_BANKROLL = 1000.0
KELLY_FRACTION = 0.25
MAX_BET_FRACTION = 0.10
N_SIMULATIONS = 10000

# Load actual backtest results
print("Loading backtest results...")
results_df = pd.read_csv("data/results/calibrated_backtest_2024_25.csv")

print(f"✅ Loaded {len(results_df):,} bets")
print(f"   Actual win rate: {results_df['won'].mean() * 100:.2f}%")
print(f"   Actual final bankroll: ${results_df['bankroll'].iloc[-1]:,.2f}")
print()

# Extract bet characteristics (not outcomes yet)
bets = []
for idx, row in results_df.iterrows():
    bets.append(
        {
            "edge": row["edge"],
            "decimal_odds": row["decimal_odds"],
            "won": row["won"],  # We'll shuffle this
        }
    )

print(f"Running {N_SIMULATIONS:,} Monte Carlo simulations...")
print("(Randomly reordering bet outcomes to test if order matters)")
print()

# Run simulations
simulation_results = []

for sim in tqdm(range(N_SIMULATIONS), desc="Simulating"):
    # Randomly shuffle the bet outcomes
    # Keep edge and odds the same, but shuffle win/loss
    shuffled_bets = bets.copy()
    outcomes = [bet["won"] for bet in bets]
    np.random.shuffle(outcomes)

    for i, bet in enumerate(shuffled_bets):
        bet["won"] = outcomes[i]

    # Simulate bankroll with Kelly Criterion
    bankroll = STARTING_BANKROLL
    max_bankroll = STARTING_BANKROLL
    min_bankroll = STARTING_BANKROLL

    for bet in shuffled_bets:
        # Kelly sizing
        kelly_bet_fraction = bet["edge"] * KELLY_FRACTION
        kelly_bet_fraction = min(kelly_bet_fraction, MAX_BET_FRACTION)
        kelly_bet_fraction = max(0, kelly_bet_fraction)

        bet_amount = bankroll * kelly_bet_fraction

        if bet_amount <= 0 or not np.isfinite(bet_amount):
            continue

        # Calculate profit/loss
        if bet["won"]:
            profit = bet_amount * (bet["decimal_odds"] - 1)
        else:
            profit = -bet_amount

        bankroll += profit

        # Track extremes
        max_bankroll = max(max_bankroll, bankroll)
        min_bankroll = min(min_bankroll, bankroll)

        # Check for bust
        if bankroll <= 0:
            bankroll = 0
            break

    # Store results
    final_return = ((bankroll - STARTING_BANKROLL) / STARTING_BANKROLL) * 100
    max_drawdown = ((min_bankroll - max_bankroll) / max_bankroll) * 100

    simulation_results.append(
        {
            "final_bankroll": bankroll,
            "final_return_pct": final_return,
            "max_drawdown_pct": max_drawdown,
            "went_bust": bankroll <= 0,
        }
    )

sim_df = pd.DataFrame(simulation_results)

# Analysis
print("\n" + "=" * 80)
print("MONTE CARLO RESULTS")
print("=" * 80)
print()

print("PROFITABILITY ANALYSIS:")
profitable_sims = (sim_df["final_bankroll"] > STARTING_BANKROLL).sum()
bust_sims = sim_df["went_bust"].sum()

print(
    f"  Profitable simulations: {profitable_sims:,} / {N_SIMULATIONS:,} ({profitable_sims/N_SIMULATIONS*100:.1f}%)"
)
print(
    f"  Bust (bankroll → $0): {bust_sims:,} / {N_SIMULATIONS:,} ({bust_sims/N_SIMULATIONS*100:.1f}%)"
)
print(
    f"  Break-even or worse: {N_SIMULATIONS - profitable_sims:,} ({(N_SIMULATIONS - profitable_sims)/N_SIMULATIONS*100:.1f}%)"
)
print()

print("EXPECTED VALUE:")
print(f"  Mean final bankroll: ${sim_df['final_bankroll'].mean():,.2f}")
print(f"  Median final bankroll: ${sim_df['final_bankroll'].median():,.2f}")
print(f"  Std dev: ${sim_df['final_bankroll'].std():,.2f}")
print()

print("RETURN STATISTICS:")
print(f"  Mean return: {sim_df['final_return_pct'].mean():+.1f}%")
print(f"  Median return: {sim_df['final_return_pct'].median():+.1f}%")
print(f"  Best return: {sim_df['final_return_pct'].max():+.1f}%")
print(f"  Worst return: {sim_df['final_return_pct'].min():+.1f}%")
print()

# Confidence intervals
percentiles = [5, 25, 50, 75, 95]
print("CONFIDENCE INTERVALS:")
for p in percentiles:
    value = np.percentile(sim_df["final_return_pct"], p)
    print(f"  {p}th percentile: {value:+.1f}%")
print()

print("DRAWDOWN ANALYSIS:")
print(f"  Mean max drawdown: {sim_df['max_drawdown_pct'].mean():.1f}%")
print(f"  Median max drawdown: {sim_df['max_drawdown_pct'].median():.1f}%")
print(f"  Worst drawdown: {sim_df['max_drawdown_pct'].min():.1f}%")
print()

# Compare to actual result
actual_final = results_df["bankroll"].iloc[-1]
actual_return = ((actual_final - STARTING_BANKROLL) / STARTING_BANKROLL) * 100
percentile_rank = (sim_df["final_return_pct"] < actual_return).sum() / N_SIMULATIONS * 100

print("=" * 80)
print("ACTUAL RESULT vs MONTE CARLO")
print("=" * 80)
print(f"  Actual final bankroll: ${actual_final:,.2f}")
print(f"  Actual return: {actual_return:+.1f}%")
print(f"  Percentile rank: {percentile_rank:.1f}th percentile")
print()

if percentile_rank > 95:
    print("  🎲 VERY LUCKY! Your result is in the top 5% of simulations.")
    print("     This suggests bet ordering played a major role.")
elif percentile_rank > 75:
    print("  🍀 LUCKY! Your result is better than 75% of simulations.")
    print("     Some luck involved, but model has edge.")
elif percentile_rank > 50:
    print("  ✅ ABOVE AVERAGE! Result is better than median.")
    print("     Model has positive expected value.")
elif percentile_rank > 25:
    print("  ⚠️  BELOW AVERAGE! Result is worse than median.")
    print("     You may have been unlucky with bet ordering.")
else:
    print("  ❌ UNLUCKY! Result is in bottom 25% of simulations.")
    print("     Bet ordering worked against you.")
print()

# Skill vs Luck interpretation
print("=" * 80)
print("SKILL vs LUCK INTERPRETATION")
print("=" * 80)
print()

if bust_sims < N_SIMULATIONS * 0.10:
    print("✅ MODEL HAS SKILL:")
    print(f"   - Only {bust_sims/N_SIMULATIONS*100:.1f}% of simulations went bust")
    print("   - Bet outcomes have positive expected value")
    print("   - Profitability is NOT just lucky ordering")
else:
    print("⚠️  HIGH RISK MODEL:")
    print(f"   - {bust_sims/N_SIMULATIONS*100:.1f}% of simulations went bust")
    print("   - Significant risk of ruin")
    print("   - Consider reducing Kelly fraction")

if percentile_rank > 90:
    print()
    print("⚠️  YOUR SPECIFIC RESULT WAS LUCKY:")
    print(f"   - You're in the {percentile_rank:.0f}th percentile")
    print("   - Expect future results closer to median")
    print(
        f"   - Median return: {sim_df['final_return_pct'].median():+.1f}% (not {actual_return:+.1f}%)"
    )

# Save results
print()
print("=" * 80)
print("SAVING RESULTS")
print("=" * 80)

# Save distribution
output_file = "data/results/monte_carlo_simulation_results.csv"
sim_df.to_csv(output_file, index=False)
print(f"✅ Saved {N_SIMULATIONS:,} simulation results to: {output_file}")

# Create visualization
print("Creating visualization...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Return distribution
ax1 = axes[0, 0]
ax1.hist(sim_df["final_return_pct"], bins=50, alpha=0.7, edgecolor="black")
ax1.axvline(
    actual_return, color="red", linestyle="--", linewidth=2, label=f"Actual: {actual_return:+.0f}%"
)
ax1.axvline(
    sim_df["final_return_pct"].median(),
    color="green",
    linestyle="--",
    linewidth=2,
    label=f'Median: {sim_df["final_return_pct"].median():+.0f}%',
)
ax1.set_xlabel("Final Return (%)")
ax1.set_ylabel("Frequency")
ax1.set_title("Distribution of Returns (10,000 Simulations)")
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Final bankroll distribution
ax2 = axes[0, 1]
# Remove bust simulations for clearer visualization
non_bust = sim_df[~sim_df["went_bust"]]
ax2.hist(non_bust["final_bankroll"], bins=50, alpha=0.7, edgecolor="black")
ax2.axvline(
    actual_final, color="red", linestyle="--", linewidth=2, label=f"Actual: ${actual_final:,.0f}"
)
ax2.axvline(
    non_bust["final_bankroll"].median(),
    color="green",
    linestyle="--",
    linewidth=2,
    label=f'Median: ${non_bust["final_bankroll"].median():,.0f}',
)
ax2.set_xlabel("Final Bankroll ($)")
ax2.set_ylabel("Frequency")
ax2.set_title("Distribution of Final Bankroll (Non-Bust Only)")
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Cumulative probability
ax3 = axes[1, 0]
sorted_returns = np.sort(sim_df["final_return_pct"])
cumulative_prob = np.arange(1, len(sorted_returns) + 1) / len(sorted_returns)
ax3.plot(sorted_returns, cumulative_prob * 100, linewidth=2)
ax3.axvline(
    actual_return, color="red", linestyle="--", linewidth=2, label=f"Actual: {actual_return:+.0f}%"
)
ax3.axhline(50, color="gray", linestyle=":", alpha=0.5)
ax3.set_xlabel("Return (%)")
ax3.set_ylabel("Cumulative Probability (%)")
ax3.set_title("Cumulative Distribution Function")
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Risk of ruin by percentile
ax4 = axes[1, 1]
bins = [-np.inf, -50, -25, 0, 25, 50, 100, 500, np.inf]
labels = [
    "< -50%",
    "-50 to -25%",
    "-25 to 0%",
    "0 to 25%",
    "25 to 50%",
    "50 to 100%",
    "100 to 500%",
    "> 500%",
]
sim_df["return_bin"] = pd.cut(sim_df["final_return_pct"], bins=bins, labels=labels)
counts = sim_df["return_bin"].value_counts().sort_index()
colors = ["darkred", "red", "orange", "yellow", "lightgreen", "green", "darkgreen", "blue"]
ax4.bar(range(len(counts)), counts.values, color=colors[: len(counts)], edgecolor="black")
ax4.set_xticks(range(len(counts)))
ax4.set_xticklabels(labels, rotation=45, ha="right")
ax4.set_ylabel("Number of Simulations")
ax4.set_title("Distribution of Outcomes")
ax4.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plot_file = "data/results/monte_carlo_simulation.png"
plt.savefig(plot_file, dpi=150, bbox_inches="tight")
print(f"✅ Saved visualization to: {plot_file}")

print()
print("=" * 80)
print("✅ MONTE CARLO SIMULATION COMPLETE")
print("=" * 80)
print()
print("KEY TAKEAWAY:")
if bust_sims < N_SIMULATIONS * 0.10 and sim_df["final_return_pct"].median() > 0:
    print("  ✅ Model has POSITIVE expected value")
    print("  ✅ Profitability is NOT just luck")
    if percentile_rank > 90:
        print("  ⚠️  But your specific result WAS lucky (top 10%)")
        print(f"      Expect future returns closer to {sim_df['final_return_pct'].median():+.0f}%")
else:
    print("  ⚠️  Model has HIGH RISK or NEGATIVE expected value")
    print("  ⚠️  Profitability may have been due to lucky ordering")
    print("      Consider more conservative Kelly fraction")
