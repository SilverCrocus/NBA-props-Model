"""
Real $1000 Starting Bankroll Simulation

Simulates starting with $1000 and using Kelly Criterion bet sizing
to show actual bankroll growth over the 2024-25 season.
"""

import pandas as pd

print("=" * 80)
print("REAL BANKROLL SIMULATION - $1000 STARTING")
print("=" * 80)

# Load winner's predictions with bet results
df = pd.read_csv("data/results/WINNER_two_stage_calib_2024_25.csv")
df["game_date"] = pd.to_datetime(df["game_date"])
df = df.sort_values("game_date").reset_index(drop=True)

# Filter to only bets (edge >= 3)
bets = df[df["bet_placed"]].copy()
print(f"\n1. Loaded {len(bets):,} bets from {bets['game_date'].min()} to {bets['game_date'].max()}")

# Simulation parameters
starting_bankroll = 1000
bankroll = starting_bankroll
kelly_fraction = 0.25  # Use 1/4 Kelly for safety
min_bet = 10  # Minimum bet size
max_bet_pct = 0.05  # Max 5% of bankroll per bet

# Track bankroll over time
bankroll_history = []
bet_history = []

print("\n2. Simulation parameters:")
print(f"   Starting bankroll: ${starting_bankroll:,.0f}")
print(f"   Kelly fraction: {kelly_fraction} (1/4 Kelly)")
print(f"   Min bet: ${min_bet}")
print(f"   Max bet %: {max_bet_pct * 100:.0f}% of bankroll")


# Helper function
def american_to_prob(odds):
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return abs(odds) / (abs(odds) + 100)


# Simulate each bet
for idx, row in bets.iterrows():
    # Determine bet side and odds
    edge = row["edge"]
    if edge >= 3:
        bet_side = "OVER"
        odds = row["over_price"]
    else:
        bet_side = "UNDER"
        odds = row["under_price"]

    # Calculate implied probability and our estimated probability
    implied_prob = american_to_prob(odds)

    # Our estimated probability (based on our prediction)
    # For over: prob = how often we think actual > line
    # Simplified: use win rate from validation (52.72%)
    our_prob = 0.5272

    # Kelly criterion: f = (p*b - q) / b
    # where p = our probability, q = 1-p, b = decimal odds - 1
    if odds > 0:
        decimal_odds = 1 + (odds / 100)
    else:
        decimal_odds = 1 + (100 / abs(odds))

    b = decimal_odds - 1
    p = our_prob
    q = 1 - p

    kelly_bet_fraction = (p * b - q) / b
    kelly_bet = bankroll * kelly_bet_fraction * kelly_fraction  # Apply fractional Kelly

    # Apply constraints
    bet_size = max(min_bet, min(kelly_bet, bankroll * max_bet_pct))
    bet_size = min(bet_size, bankroll)  # Can't bet more than we have

    # Check if we can afford the bet
    if bankroll < min_bet:
        print(f"\n❌ BANKRUPT at bet #{idx + 1}")
        break

    # Calculate result
    bet_result = row["bet_result"]

    # Convert bet result from $100 basis to our bet size
    if bet_result > 0:
        profit = (bet_result / 100) * bet_size
    elif bet_result < 0:
        profit = -bet_size
    else:  # Push
        profit = 0

    # Update bankroll
    bankroll += profit

    # Record
    bankroll_history.append(
        {
            "bet_num": idx + 1,
            "date": row["game_date"],
            "bet_size": bet_size,
            "bet_result": "WIN" if profit > 0 else ("LOSS" if profit < 0 else "PUSH"),
            "profit": profit,
            "bankroll": bankroll,
        }
    )

# Convert to DataFrame
history_df = pd.DataFrame(bankroll_history)

# Calculate statistics
final_bankroll = bankroll
total_profit = final_bankroll - starting_bankroll
total_return = (final_bankroll / starting_bankroll - 1) * 100
total_bets_placed = len(history_df)
winning_bets = (history_df["bet_result"] == "WIN").sum()
losing_bets = (history_df["bet_result"] == "LOSS").sum()
win_rate = winning_bets / total_bets_placed * 100 if total_bets_placed > 0 else 0

print("\n" + "=" * 80)
print("SIMULATION RESULTS")
print("=" * 80)

print("\n💰 BANKROLL GROWTH:")
print(f"   Starting: ${starting_bankroll:,.2f}")
print(f"   Ending:   ${final_bankroll:,.2f}")
print(f"   Profit:   ${total_profit:,.2f}")
print(f"   Return:   {total_return:+.2f}%")

print("\n📊 BETTING ACTIVITY:")
print(f"   Total bets: {total_bets_placed:,}")
print(f"   Won:  {winning_bets:,} ({win_rate:.2f}%)")
print(f"   Lost: {losing_bets:,}")
print(f"   Win rate: {win_rate:.2f}%")

print("\n📈 BANKROLL JOURNEY:")
print(f"   Peak: ${history_df['bankroll'].max():,.2f}")
print(f"   Trough: ${history_df['bankroll'].min():,.2f}")
print(f"   Avg bet size: ${history_df['bet_size'].mean():.2f}")
print(f"   Max bet size: ${history_df['bet_size'].max():.2f}")

# Calculate max drawdown
running_max = history_df["bankroll"].expanding().max()
drawdown = (history_df["bankroll"] - running_max) / running_max * 100
max_drawdown = drawdown.min()

print(f"   Max drawdown: {max_drawdown:.2f}%")

# Show key milestones
milestones = [1500, 2000, 2500, 3000, 4000, 5000]
print("\n🎯 MILESTONES:")
for milestone in milestones:
    reached = history_df[history_df["bankroll"] >= milestone]
    if len(reached) > 0:
        first_reach = reached.iloc[0]
        bet_num = int(first_reach["bet_num"])
        reach_date = first_reach["date"].date()
        print(f"   ${milestone:,}: Reached on {reach_date} (bet #{bet_num})")
    else:
        print(f"   ${milestone:,}: Not reached")

# Final summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

if final_bankroll > starting_bankroll * 1.5:
    verdict = "🎉 EXCELLENT GROWTH!"
elif final_bankroll > starting_bankroll * 1.2:
    verdict = "✅ STRONG GROWTH"
elif final_bankroll > starting_bankroll:
    verdict = "✅ PROFITABLE"
else:
    verdict = "❌ LOSING"

print("\nStarting with $1,000:")
print(f"  After {total_bets_placed:,} bets: ${final_bankroll:,.2f}")
print(f"  {verdict}")

print("\nThis simulation uses:")
print("  • 1/4 Kelly bet sizing (conservative)")
print(f"  • Min bet: ${min_bet}")
print(f"  • Max bet: {max_bet_pct * 100:.0f}% of bankroll")
print("  • Actual chronological bet sequence from 2024-25")

# Save history
history_df.to_csv("data/results/bankroll_simulation_1000.csv", index=False)
print("\n✅ Bankroll history saved to data/results/bankroll_simulation_1000.csv")

print("\n" + "=" * 80)
