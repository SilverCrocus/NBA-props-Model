"""
Monte Carlo Simulation - Test Bet Sequence Sensitivity

Runs 10,000 simulations with different bet orderings to see if the
134% return was luck or if it's typical for this win rate.

This answers: "Would I have made similar returns if bets came in a different order?"
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

print("=" * 80)
print("MONTE CARLO SIMULATION - BET SEQUENCE SENSITIVITY")
print("=" * 80)

# Load the backtest results
print("\n1. Loading backtest results...")
df = pd.read_csv("data/results/FINAL_BACKTEST_predictions_2024_25.csv")
df["game_date_obj"] = pd.to_datetime(df["game_date"])
df["game_date_key"] = df["game_date_obj"].dt.date

# Merge with odds data to get full bet info
print("   Loading odds data...")
odds_df = pd.read_csv("data/historical_odds/2024-25/pra_odds.csv")
odds_df["game_date"] = pd.to_datetime(odds_df["event_date"]).dt.date
odds_df["player_lower"] = odds_df["player_name"].str.lower().str.strip()

# Line shopping
odds_best = odds_df.groupby(["player_lower", "game_date"], as_index=False).agg(
    {"line": "first", "over_price": "max", "under_price": "max"}
)

# Match predictions to odds
df["player_lower"] = df["PLAYER_NAME"].str.lower().str.strip()

merged = df.merge(
    odds_best,
    left_on=["player_lower", "game_date_key"],
    right_on=["player_lower", "game_date"],
    how="inner",
)

# Calculate edge
merged["edge"] = merged["calibrated_PRA"] - merged["line"]
merged["abs_edge"] = abs(merged["edge"])

# Filter to only bets (edge >= 3)
merged = merged[merged["abs_edge"] >= 3].copy()

print(f"✅ Loaded {len(merged):,} bets with edge >= 3 points")


# Calculate bet results for each bet
def calculate_bet_outcomes(bets_df):
    """Pre-calculate the outcome of each bet."""
    outcomes = []

    for _, row in bets_df.iterrows():
        edge = row["edge"]
        actual = row["PRA"]
        line = row["line"]

        if edge >= 3:
            odds = row["over_price"]
            won = actual > line
        else:
            odds = row["under_price"]
            won = actual < line

        # Push
        if actual == line:
            result = 0
        elif won:
            if odds > 0:
                result = odds  # Profit per $100 bet
            else:
                result = 100 * (100 / abs(odds))
        else:
            result = -100  # Loss

        outcomes.append({"won": won, "result": result, "odds": odds, "edge": edge})

    return pd.DataFrame(outcomes)


# Pre-calculate outcomes
print("\n2. Calculating bet outcomes...")
outcomes_df = calculate_bet_outcomes(merged)
print(f"   Win rate: {outcomes_df['won'].mean() * 100:.2f}%")
print(f"   Total bets: {len(outcomes_df):,}")


# Simulation function
def simulate_bankroll(
    bet_results,
    starting_bankroll=1000,
    kelly_fraction=0.25,
    min_bet=10,
    max_bet_pct=0.05,
    win_rate=0.52,
):
    """Simulate bankroll with given bet sequence."""
    bankroll = starting_bankroll

    for _, row in bet_results.iterrows():
        # Kelly sizing
        odds = row["odds"]

        if odds > 0:
            decimal_odds = 1 + (odds / 100)
        else:
            decimal_odds = 1 + (100 / abs(odds))

        b = decimal_odds - 1
        p = win_rate
        q = 1 - p

        kelly_bet_fraction = (p * b - q) / b
        kelly_bet = bankroll * kelly_bet_fraction * kelly_fraction

        # Apply constraints
        bet_size = max(min_bet, min(kelly_bet, bankroll * max_bet_pct))
        bet_size = min(bet_size, bankroll)

        if bankroll < min_bet:
            break

        # Apply result
        bet_result = row["result"]
        if bet_result > 0:
            profit = (bet_result / 100) * bet_size
        elif bet_result < 0:
            profit = -bet_size
        else:
            profit = 0

        bankroll += profit

        if bankroll <= 0:
            bankroll = 0
            break

    return bankroll


# Run Monte Carlo simulation
print("\n3. Running Monte Carlo simulation...")
print("   Simulations: 10,000")
print("   Method: Shuffle bet order, keep outcomes same")

n_simulations = 10000
win_rate = outcomes_df["won"].mean()
ending_bankrolls = []

for i in tqdm(range(n_simulations), desc="Simulating"):
    # Shuffle bet order
    shuffled_bets = outcomes_df.sample(frac=1, random_state=i).reset_index(drop=True)

    # Simulate
    final_bankroll = simulate_bankroll(shuffled_bets, win_rate=win_rate)
    ending_bankrolls.append(final_bankroll)

ending_bankrolls = np.array(ending_bankrolls)

# Calculate statistics
print("\n" + "=" * 80)
print("MONTE CARLO RESULTS")
print("=" * 80)

# Original chronological result
original_result = 2346.87  # From the actual backtest

print("\n💰 BANKROLL DISTRIBUTION:")
print(f"   Mean:     ${ending_bankrolls.mean():,.2f}")
print(f"   Median:   ${np.median(ending_bankrolls):,.2f}")
print(f"   Std Dev:  ${ending_bankrolls.std():,.2f}")

print("\n📊 PERCENTILES:")
percentiles = [5, 10, 25, 50, 75, 90, 95]
for p in percentiles:
    value = np.percentile(ending_bankrolls, p)
    print(f"   {p:2d}th percentile: ${value:>8,.2f}")

print(f"\n🎯 YOUR ACTUAL RESULT: ${original_result:,.2f}")

# Where does actual result rank?
better_than_actual = (ending_bankrolls < original_result).sum()
percentile_rank = (better_than_actual / n_simulations) * 100
print(f"   Better than {percentile_rank:.1f}% of simulations")

if percentile_rank > 75:
    print(f"   ✅ You got LUCKY! Top {100 - percentile_rank:.0f}% outcome")
elif percentile_rank > 50:
    print("   ⚠️  Above average outcome")
elif percentile_rank > 25:
    print("   ✅ Typical outcome")
else:
    print("   ⚠️  Below average outcome")

print("\n📈 PROFITABILITY:")
profitable = (ending_bankrolls > 1000).sum()
prob_profit = profitable / n_simulations * 100
print(f"   Profitable simulations: {profitable:,} / {n_simulations:,} ({prob_profit:.1f}%)")

breakeven = (ending_bankrolls >= 1000).sum()
prob_breakeven = breakeven / n_simulations * 100
print(f"   Break-even or better: {breakeven:,} / {n_simulations:,} ({prob_breakeven:.1f}%)")

busted = (ending_bankrolls < 100).sum()
prob_bust = busted / n_simulations * 100
print(f"   Went nearly bust (<$100): {busted:,} / {n_simulations:,} ({prob_bust:.1f}%)")

print("\n💸 OUTCOMES:")
best_return = (ending_bankrolls.max() / 1000 - 1) * 100
worst_return = (ending_bankrolls.min() / 1000 - 1) * 100
print(f"   Best case:  ${ending_bankrolls.max():,.2f} ({best_return:+.0f}% return)")
print(f"   Worst case: ${ending_bankrolls.min():,.2f} ({worst_return:+.0f}% return)")
print(f"   Range:      ${ending_bankrolls.max() - ending_bankrolls.min():,.2f}")

# Expected value
mean_return = (ending_bankrolls.mean() / 1000 - 1) * 100
median_return = (np.median(ending_bankrolls) / 1000 - 1) * 100
print("\n📊 EXPECTED RETURNS:")
print(f"   Mean return:   {mean_return:+.1f}%")
print(f"   Median return: {median_return:+.1f}%")
print(f"   Your return:   {(original_result / 1000 - 1) * 100:+.1f}%")

# Risk metrics
sharpe_returns = ending_bankrolls / 1000 - 1
sharpe_ratio = sharpe_returns.mean() / sharpe_returns.std()
print("\n📉 RISK METRICS:")
print(f"   Sharpe ratio (approx): {sharpe_ratio:.2f}")
print(f"   Volatility: {sharpe_returns.std() * 100:.1f}%")

# Save results
results_df = pd.DataFrame(
    {
        "simulation": range(n_simulations),
        "ending_bankroll": ending_bankrolls,
        "return_pct": (ending_bankrolls / 1000 - 1) * 100,
    }
)
results_df.to_csv("data/results/monte_carlo_results.csv", index=False)
print("\n✅ Results saved to data/results/monte_carlo_results.csv")

# Create visualization
print("\n4. Creating visualization...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Histogram
axes[0, 0].hist(ending_bankrolls, bins=100, alpha=0.7, edgecolor="black")
axes[0, 0].axvline(
    original_result,
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"Your Result: ${original_result:,.0f}",
)
axes[0, 0].axvline(
    ending_bankrolls.mean(),
    color="green",
    linestyle="--",
    linewidth=2,
    label=f"Mean: ${ending_bankrolls.mean():,.0f}",
)
axes[0, 0].axvline(1000, color="gray", linestyle=":", linewidth=2, label="Break-even")
axes[0, 0].set_xlabel("Ending Bankroll ($)")
axes[0, 0].set_ylabel("Frequency")
axes[0, 0].set_title("Distribution of Ending Bankrolls")
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Return distribution
returns = (ending_bankrolls / 1000 - 1) * 100
axes[0, 1].hist(returns, bins=100, alpha=0.7, edgecolor="black", color="orange")
axes[0, 1].axvline(
    (original_result / 1000 - 1) * 100,
    color="red",
    linestyle="--",
    linewidth=2,
    label="Your Return",
)
axes[0, 1].axvline(returns.mean(), color="green", linestyle="--", linewidth=2, label="Mean")
axes[0, 1].axvline(0, color="gray", linestyle=":", linewidth=2, label="Break-even")
axes[0, 1].set_xlabel("Return (%)")
axes[0, 1].set_ylabel("Frequency")
axes[0, 1].set_title("Distribution of Returns")
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Box plot
axes[1, 0].boxplot(ending_bankrolls, vert=True)
axes[1, 0].axhline(original_result, color="red", linestyle="--", linewidth=2, label="Your Result")
axes[1, 0].axhline(1000, color="gray", linestyle=":", linewidth=2, label="Starting Bankroll")
axes[1, 0].set_ylabel("Ending Bankroll ($)")
axes[1, 0].set_title("Bankroll Spread")
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Cumulative probability
sorted_bankrolls = np.sort(ending_bankrolls)
cumulative_prob = np.arange(1, len(sorted_bankrolls) + 1) / len(sorted_bankrolls) * 100
axes[1, 1].plot(sorted_bankrolls, cumulative_prob, linewidth=2)
axes[1, 1].axvline(
    original_result,
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"Your Result ({percentile_rank:.0f}th %ile)",
)
axes[1, 1].axvline(1000, color="gray", linestyle=":", linewidth=2, label="Break-even")
axes[1, 1].set_xlabel("Ending Bankroll ($)")
axes[1, 1].set_ylabel("Cumulative Probability (%)")
axes[1, 1].set_title("Cumulative Distribution")
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("data/results/monte_carlo_distribution.png", dpi=300, bbox_inches="tight")
print("✅ Visualization saved to data/results/monte_carlo_distribution.png")

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

# Determine verdict
if percentile_rank > 75:
    verdict = "You got LUCKY!"
elif percentile_rank > 25:
    verdict = "Typical result"
else:
    verdict = "Below average"

print(
    f"""
Starting with $1,000 and your model's 52.00% win rate:

TYPICAL OUTCOME:
• Median ending bankroll: ${np.median(ending_bankrolls):,.0f}
• Mean ending bankroll: ${ending_bankrolls.mean():,.0f}
• Median return: {median_return:+.1f}%

YOUR ACTUAL OUTCOME:
• Ending bankroll: ${original_result:,.0f}
• Return: {(original_result / 1000 - 1) * 100:+.1f}%
• Better than {percentile_rank:.0f}% of simulations

RISK ASSESSMENT:
• Probability of profit: {prob_profit:.1f}%
• Probability of near-bust: {prob_bust:.1f}%
• Best case: ${ending_bankrolls.max():,.0f} ({(ending_bankrolls.max() / 1000 - 1) * 100:+.0f}%)
• Worst case: ${ending_bankrolls.min():,.0f} ({(ending_bankrolls.min() / 1000 - 1) * 100:+.0f}%)

VERDICT: {verdict}
"""
)

if prob_profit < 60:
    print("⚠️  WARNING: Less than 60% chance of profit - model is marginal")
elif prob_profit < 75:
    print("⚠️  CAUTION: 60-75% chance of profit - risky but viable")
elif prob_profit < 90:
    print("✅ GOOD: 75-90% chance of profit - solid model")
else:
    print("🎉 EXCELLENT: 90%+ chance of profit - strong model")

print("\n" + "=" * 80)
