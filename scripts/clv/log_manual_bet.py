#!/usr/bin/env python3
"""
Manual Bet Logger - For Bet365 and Custom Bookmakers
====================================================
Log bets manually with custom bookmaker and odds when your actual bookmaker
differs from the recommended odds source.

Use case: Model uses DraftKings odds, but you bet on Bet365. Log the actual
Bet365 odds/lines you placed.

Usage:
    # Log a single bet interactively
    uv run scripts/clv/log_manual_bet.py

    # Log from command line
    uv run scripts/clv/log_manual_bet.py \
        --date 2025 - 10 - 23 \
        --player "Stephen Curry" \
        --line 33.5 \
        --side UNDER \
        --odds -110 \
        --stake 50.00 \
        --bookmaker "Bet365" \
        --prediction 29.0

Examples:
    # Interactive mode (recommended)
    uv run scripts/clv/log_manual_bet.py

    # Quick command-line logging with decimal odds
    uv run scripts/clv/log_manual_bet.py \\
        --date 2025 - 10 - 23 \\
        --player "Nikola Jokic" \\
        --line 47.5 \\
        --side UNDER \\
        --decimal-odds 1.87 \\
        --stake 37.30 \\
        --bookmaker "Bet365"

    # Or with American odds
    uv run scripts/clv/log_manual_bet.py \\
        --date 2025 - 10 - 23 \\
        --player "Stephen Curry" \\
        --line 33.5 \\
        --side UNDER \\
        --odds -110 \\
        --stake 50.00 \\
        --bookmaker "Bet365"
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from utils.clv_tracker import CLVTracker

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))


def american_to_decimal(american_odds: int) -> float:
    """Convert American odds to decimal odds"""
    if american_odds > 0:
        return (american_odds / 100) + 1
    else:
        return (100 / abs(american_odds)) + 1


def interactive_bet_entry():
    """Interactive prompt for bet entry"""
    print("\n" + "=" * 80)
    print("📝 MANUAL BET ENTRY")
    print("=" * 80)
    print("\nEnter bet details (press Ctrl+C to cancel):\n")

    try:
        # Date
        date = input("📅 Game date (YYYY-MM-DD): ").strip()
        datetime.strptime(date, "%Y-%m-%d")  # Validate format

        # Player
        player = input("🏀 Player name: ").strip()

        # Market
        print("\n📊 Market type:")
        print("   1. PRA (Points + Rebounds + Assists)")
        print("   2. Points")
        print("   3. Rebounds")
        print("   4. Assists")
        market_choice = input("   Select (1 - 4): ").strip()
        market_map = {"1": "PRA", "2": "Points", "3": "Rebounds", "4": "Assists"}
        market = market_map.get(market_choice, "PRA")

        # Line
        line = float(input(f"\n📏 {market} line (e.g., 33.5): ").strip())

        # Side
        print("\n⬆️⬇️  Direction:")
        print("   1. OVER")
        print("   2. UNDER")
        side_choice = input("   Select (1 - 2): ").strip()
        side = "OVER" if side_choice == "1" else "UNDER"

        # Odds
        print("\n💰 Odds format:")
        print("   1. Decimal (e.g., 1.91)")
        print("   2. American (e.g., -110)")
        odds_format = input("   Select (1 - 2): ").strip()

        if odds_format == "1":
            while True:
                decimal_odds_input = input("\n💰 Decimal odds (e.g., 1.91): ").strip()
                try:
                    decimal_odds = float(decimal_odds_input)
                    # Validate: typical betting odds are 1.01 to 10.0
                    if decimal_odds < 1.01 or decimal_odds > 10.0:
                        print(f"   ⚠️  Unusual odds: {decimal_odds:.2f}")
                        print(
                            "   💡 Typical odds are 1.01 to 10.0 (e.g., 1.50, 1.91, 2.50)"
                        )  # noqa: E501
                        confirm = (
                            input("   Are you sure this is correct? (y/n): ").strip().lower()
                        )  # noqa: E501
                        if confirm != "y":
                            continue
                    break
                except ValueError:
                    print("   ❌ Invalid input. Please enter a number (e.g., 1.91)")  # noqa: E501

            # Calculate American odds for display
            if decimal_odds >= 2.0:
                american_odds = int((decimal_odds - 1) * 100)
            else:
                american_odds = int(-100 / (decimal_odds - 1))
            print(f"   → American: {american_odds:+d}")
        else:
            american_odds = int(input("\n💰 American odds (e.g., -110): ").strip())
            decimal_odds = american_to_decimal(american_odds)
            print(f"   → Decimal: {decimal_odds:.3f}")

        # Stake
        stake = float(input("\n💵 Bet amount ($): ").strip())

        # Bookmaker (default to Bet365, skip prompt)
        bookmaker = "Bet365"

        # Skip optional fields (these come from existing bet if updating)
        predicted_value = None
        player_sigma = None

        # Confirm
        print("\n" + "=" * 80)
        print("📋 BET SUMMARY")
        print("=" * 80)
        print(f"\n📅 Date: {date}")
        print(f"🏀 Player: {player}")
        print(f"📊 Market: {market} {side} {line}")
        if odds_format == "1":
            print(f"💰 Decimal Odds: {decimal_odds:.3f}")
        else:
            print(
                f"💰 American Odds: {american_odds} (Decimal: {
                    decimal_odds:.3f})"
            )
        print(f"💵 Stake: ${stake:.2f}")
        print(f"🏦 Bookmaker: {bookmaker}")

        confirm = input("\n✅ Update/Log this bet? (y/n): ").strip().lower()

        if confirm != "y":
            print("\n❌ Bet entry cancelled")
            return None

        return {
            "date": date,
            "player": player,
            "market": market,
            "side": side,
            "line": line,
            "decimal_odds": decimal_odds,
            "stake": stake,
            "bookmaker": bookmaker,
            "predicted_value": predicted_value,
            "player_sigma": player_sigma,
        }

    except KeyboardInterrupt:
        print("\n\n❌ Cancelled by user")
        return None
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Manually log bets with custom bookmaker/odds")

    # Optional command-line arguments
    parser.add_argument("--date", help="Game date (YYYY-MM-DD)")
    parser.add_argument("--player", help="Player name")
    parser.add_argument("--market", default="PRA", help="Market type (default: PRA)")
    parser.add_argument("--line", type=float, help="Betting line (e.g., 33.5)")
    parser.add_argument("--side", choices=["OVER", "UNDER"], help="OVER or UNDER")
    parser.add_argument("--odds", type=int, help="American odds (e.g., -110)")
    parser.add_argument(
        "--decimal-odds", type=float, help="Decimal odds (e.g., 1.91) - alternative to --odds"
    )
    parser.add_argument("--stake", type=float, help="Bet amount in dollars")
    parser.add_argument("--bookmaker", default="Bet365", help="Bookmaker name")
    parser.add_argument("--prediction", type=float, help="Model prediction (optional)")
    parser.add_argument("--sigma", type=float, help="Player variance (optional)")

    args = parser.parse_args()

    # Check if all required args provided (command-line mode)
    # Need either --odds OR --decimal-odds
    has_odds = args.odds is not None or args.decimal_odds is not None
    if all([args.date, args.player, args.line, args.side, has_odds, args.stake]):  # noqa: E501
        # Command-line mode
        # Determine decimal odds
        if args.decimal_odds is not None:
            decimal_odds = args.decimal_odds
        else:
            decimal_odds = american_to_decimal(args.odds)

        bet_data = {
            "date": args.date,
            "player": args.player,
            "market": args.market,
            "side": args.side,
            "line": args.line,
            "decimal_odds": decimal_odds,
            "stake": args.stake,
            "bookmaker": args.bookmaker,
            "predicted_value": args.prediction,
            "player_sigma": args.sigma,
        }
    else:
        # Interactive mode
        bet_data = interactive_bet_entry()

    if not bet_data:
        return 1

    # Log to CLV tracker
    tracker = CLVTracker()

    try:
        import unicodedata

        # Normalize player name for matching
        def normalize_name(name):
            name = unicodedata.normalize("NFKD", name)
            name = name.encode("ASCII", "ignore").decode("ASCII")
            return name.lower().strip()

        def get_last_name(name):
            """Extract last name from full name"""
            return normalize_name(name).split()[-1]

        # Search for existing bets with same player/date/side (IGNORE line - it
        # might be different!)
        user_last_name = get_last_name(bet_data["player"])

        # Find all matching bets (by player name or last name + date + side)
        matching_bets = tracker.ledger[
            (tracker.ledger["date"] == bet_data["date"])
            & (
                (
                    tracker.ledger["player"].apply(normalize_name)
                    == normalize_name(bet_data["player"])
                )
                | (tracker.ledger["player"].apply(get_last_name) == user_last_name)  # noqa: E501
            )
            & (tracker.ledger["side"] == bet_data["side"])
        ]

        existing_bet = pd.DataFrame()

        if not matching_bets.empty:
            if len(matching_bets) == 1:
                # Only one match - use it automatically
                existing_bet = matching_bets
                matched_player = existing_bet.iloc[0]["player"]
                old_line = existing_bet.iloc[0]["line"]
                print(
                    f"\n📝 Found: {matched_player} {
                        bet_data['side']} {old_line}"
                )
                if old_line != bet_data["line"]:
                    print(
                        f"   🔄 Line moved: {old_line} → {
                            bet_data['line']} (you got a better line!)"
                    )
            else:
                # Multiple matches - let user choose
                print(
                    f"\n📋 Found {
                        len(matching_bets)} bets for {
                        bet_data['player']} {
                        bet_data['side']} on {
                        bet_data['date']}:"
                )
                for idx, bet in matching_bets.iterrows():
                    print(
                        f"   {
                            idx +
                            1}. {
                            bet['player']} {
                            bet['side']} {
                            bet['line']} @ {
                            bet['entry_odds_dec']:.3f}"
                    )

                choice = input(
                    f"\n❓ Which bet do you want to update? (1-{len(matching_bets)}, or 0 for new bet): "  # noqa: E501
                ).strip()

                if choice != "0":
                    try:
                        selected_idx = int(choice) - 1
                        if 0 <= selected_idx < len(matching_bets):
                            existing_bet = matching_bets.iloc[[selected_idx]]
                            old_line = existing_bet.iloc[0]["line"]
                            if old_line != bet_data["line"]:
                                print(
                                    f"   🔄 Updating line: {old_line} → {
                                        bet_data['line']}"
                                )
                    except ValueError:
                        pass

        if not existing_bet.empty:
            # UPDATE existing bet with actual Bet365 line/odds/stake
            bet_id = existing_bet.iloc[0]["bet_id"]
            old_line = existing_bet.iloc[0]["line"]
            old_odds = existing_bet.iloc[0]["entry_odds_dec"]

            print("\n" + "=" * 80)
            print("🔄 UPDATING EXISTING BET")
            print("=" * 80)
            print(f"\n📝 Found existing bet: {bet_id}")
            if old_line != bet_data["line"]:
                print(f"🔄 Updating line: {old_line} → {bet_data['line']}")
            print(
                f"🔄 Updating odds: {
                    old_odds:.3f} → {
                    bet_data['decimal_odds']:.3f}"
            )
            print(
                f"🔄 Updating stake: ${
                    existing_bet.iloc[0]['stake']:.2f} → ${
                    bet_data['stake']:.2f}"
            )

            # Update the existing bet (line, odds, and stake) FIRST using old
            # bet_id
            tracker.ledger.loc[tracker.ledger["bet_id"] == bet_id, "line"] = bet_data["line"]
            tracker.ledger.loc[tracker.ledger["bet_id"] == bet_id, "entry_odds_dec"] = bet_data[
                "decimal_odds"
            ]
            tracker.ledger.loc[tracker.ledger["bet_id"] == bet_id, "stake"] = bet_data["stake"]

            # Generate new bet_id if line changed (update AFTER other fields)
            if old_line != bet_data["line"]:
                # Extract player slug from old bet_id
                pass

                # bet_id format: YYYY-MM-DD_{player_slug}_{line}_{side}
                parts = bet_id.split("_")
                date_part = parts[0]  # YYYY-MM-DD
                side_part = parts[-1]  # OVER/UNDER
                # Reconstruct player slug (everything between date and line)
                player_slug = "_".join(parts[1:-2])

                # Create new bet_id with new line
                new_bet_id = f"{date_part}_{player_slug}_{
                    bet_data['line']}_{side_part}"

                print(f"🔄 Updating bet_id: {bet_id} → {new_bet_id}")
                tracker.ledger.loc[tracker.ledger["bet_id"] == bet_id, "bet_id"] = new_bet_id

            # Save updated ledger
            tracker.ledger.to_csv(tracker.ledger_path, index=False)

            print("\n✅ Bet updated successfully!")
            print(f"📋 Ledger: {tracker.ledger_path}")
            print("\n💡 Next steps:")
            print("   1. Wait for game to complete")
            print(f"   2. Run: uv run scripts/clv/log_results.py {bet_data['date']}")  # noqa: E501
            print("   3. Results will auto-populate in ledger")
            print("\n" + "=" * 80)

        else:
            # CREATE new bet (no existing bet found)
            from datetime import datetime

            player_slug = bet_data["player"].lower().replace(" ", "_")
            bet_id = f"{bet_data['date']}_{player_slug}_{bet_data['line']}_{bet_data['side']}_manual"  # noqa: E501

            # Generate game_id (simplified - won't have actual team matchup)
            game_id = f"{bet_data['date']}_manual"
            entry_time = datetime.now().isoformat()

            tracker.log_bet_entry(
                bet_id=bet_id,
                date=bet_data["date"],
                player=bet_data["player"],
                market=bet_data["market"],
                side=bet_data["side"],
                line=bet_data["line"],
                entry_odds_dec=bet_data["decimal_odds"],
                entry_time=entry_time,
                game_id=game_id,
                predicted_pra=bet_data["predicted_value"],
                player_sigma=bet_data["player_sigma"],
                ev=None,  # Can't calculate without prediction
                stake=bet_data["stake"],
            )

            print("\n" + "=" * 80)
            print("✅ NEW BET LOGGED")
            print("=" * 80)
            print(f"\n📝 Bet ID: {bet_id}")
            print("⚠️  No existing bet found - created new entry")
            print("📋 Ledger: data/clv_ledger.csv")
            print("\n💡 Next steps:")
            print("   1. Wait for game to complete")
            print(f"   2. Run: uv run scripts/clv/log_results.py {bet_data['date']}")  # noqa: E501
            print("   3. Results will auto-populate in ledger")
            print("\n" + "=" * 80)

        return 0

    except Exception as e:
        print("\n" + "=" * 80)
        print("❌ ERROR LOGGING BET")
        print("=" * 80)
        print(f"\n{e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
