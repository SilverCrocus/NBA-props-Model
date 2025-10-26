#!/usr/bin/env python3
"""
Automatic Results Logging for CLV Tracking
===========================================
Fetches actual player stats from NBA API and updates CLV ledger with results.

Usage:
    uv run scripts/clv/log_results.py 2025 - 10 - 23
    uv run scripts/clv/log_results.py 2025 - 10 - 23 --verify  # Show results without saving  # noqa: E501

The script:
1. Loads all bets for the target date from CLV ledger
2. Fetches actual game stats from NBA API
3. Calculates PRA (Points + Rebounds + Assists)
4. Determines WIN/LOSS/PUSH for each bet
5. Updates CLV ledger with results and profit/loss

Note:
    - Run this 2 - 4 hours after games complete (stats may not be immediate)
    - Uses NBA API (already integrated in codebase)
    - Handles player name variations automatically
    - Rate limited (600ms between requests)
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder
from nba_api.stats.static import players

from utils.clv_tracker import CLVTracker

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))


def normalize_player_name(name: str, remove_suffix: bool = False) -> str:
    """
    Normalize player names for matching

    Handles:
    - Dots: R.J. → RJ
    - Suffixes: Jr, Sr, II, III, IV (optional)
    - Accents: Jokić → Jokic, Šarić → Saric
    - Extra spaces
    - Case insensitivity

    Args:
        name: Player name (e.g., "R.J. Barrett", "Nikola Jokić")
        remove_suffix: If True, removes Jr/Sr/II/III/IV suffixes

    Returns:
        Normalized name (e.g., "rj barrett", "nikola jokic")
    """
    import re
    import unicodedata

    # Remove accents/diacritics (Jokić → Jokic, Šarić → Saric)
    name = unicodedata.normalize("NFKD", name)
    name = name.encode("ASCII", "ignore").decode("ASCII")

    # Remove dots
    name = name.replace(".", "")

    # Optionally remove suffixes (Jr., Sr., II, III, IV)
    if remove_suffix:
        name = re.sub(r"\s+(Jr\.?|Sr\.?|II|III|IV)$", "", name, flags=re.IGNORECASE)

    # Remove extra spaces and lowercase
    name = " ".join(name.split()).lower()
    return name


def get_player_stats(player_name: str, game_date: str, verbose: bool = True) -> dict:
    """
    Fetch actual player stats from NBA API

    Args:
        player_name: Player name (e.g., "LeBron James")
        game_date: Game date (YYYY-MM-DD format)
        verbose: Print progress messages

    Returns:
        Dict with keys: PTS, REB, AST, PRA, MIN, GAME_ID
        Returns None if player didn't play or stats unavailable
    """
    try:
        # Get all players (cached locally by nba_api)
        all_players = players.get_players()

        # Normalize input name (keep suffix like "II" for exact match)
        normalized_input = normalize_player_name(player_name, remove_suffix=False)

        # Try exact match first (WITH suffix)
        player_match = None
        for p in all_players:
            if normalize_player_name(p["full_name"], remove_suffix=False) == normalized_input:
                player_match = p
                break

        # Fallback 1: exact match WITHOUT suffix (e.g., "Gary Payton" → "Gary
        # Payton II")
        if not player_match:
            normalized_no_suffix = normalize_player_name(player_name, remove_suffix=True)
            candidates = []
            for p in all_players:
                if (
                    normalize_player_name(p["full_name"], remove_suffix=True)
                    == normalized_no_suffix
                ):
                    candidates.append(p)

            if candidates:
                # If multiple matches, prefer the one with HIGHER ID (more recent/active player)  # noqa: E501
                # Gary Payton (ID: 56) vs Gary Payton II (ID: 1627780)
                player_match = max(candidates, key=lambda x: x["id"])
                if verbose and len(candidates) > 1:
                    print(
                        f"   📝 Multiple matches found, using active player: '{
                            player_match['full_name']}'"
                    )
                elif verbose:
                    print(
                        f"   📝 Matched: '{player_name}' → '{
                            player_match['full_name']}'"
                    )

        # Fallback 2: last name match
        if not player_match:
            last_name = normalized_input.split()[-1]
            for p in all_players:
                if last_name in normalize_player_name(p["full_name"], remove_suffix=False):
                    player_match = p
                    if verbose:
                        print(
                            f"   📝 Fuzzy matched: '{player_name}' → '{
                                p['full_name']}'"
                        )
                    break

        if not player_match:
            if verbose:
                print(f"   ❌ Player not found: {player_name}")
            return None

        # Fetch game stats for date
        player_id = player_match["id"]

        # Convert date format: YYYY-MM-DD → MM/DD/YYYY for NBA API
        date_parts = game_date.split("-")
        nba_date = f"{date_parts[1]}/{date_parts[2]}/{date_parts[0]}"

        gamefinder = leaguegamefinder.LeagueGameFinder(
            player_id_nullable=player_id, date_from_nullable=nba_date, date_to_nullable=nba_date
        )

        games = gamefinder.get_data_frames()[0]

        if games.empty:
            if verbose:
                print(
                    f"   ⏳ No game found for {
                        player_match['full_name']} on {game_date}"
                )
                print("      (Player may not have played or stats not yet available)")  # noqa: E501
            return None

        # CRITICAL: Verify the game date actually matches what we requested
        # NBA API sometimes returns games from other dates if stats aren't
        # available yet
        game = games.iloc[0]
        game_date_str = game["GAME_DATE"]

        # Parse GAME_DATE (format: YYYY-MM-DD or similar)
        # Compare with our target date
        if isinstance(game_date_str, str):
            actual_date = game_date_str[:10]  # Get YYYY-MM-DD portion
        else:
            # If GAME_DATE is datetime object
            actual_date = pd.to_datetime(game_date_str).strftime("%Y-%m-%d")

        if actual_date != game_date:
            if verbose:
                print(
                    f"   ⚠️  Found game from {actual_date}, but looking for {game_date}"
                )  # noqa: E501
                print(
                    f"      Stats for {game_date} not yet available (game may be in progress)"
                )  # noqa: E501
            return None

        return {
            "PTS": int(game["PTS"]),
            "REB": int(game["REB"]),
            "AST": int(game["AST"]),
            "PRA": int(game["PTS"]) + int(game["REB"]) + int(game["AST"]),
            "MIN": game["MIN"],
            "GAME_ID": game["GAME_ID"],
            "MATCHUP": game["MATCHUP"],
        }

    except Exception as e:
        if verbose:
            print(f"   ❌ Error fetching {player_name}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Log actual game results to CLV tracker")
    parser.add_argument("date", help="Game date (YYYY-MM-DD)")
    parser.add_argument(
        "--verify", action="store_true", help="Verify results without saving (dry run)"
    )
    parser.add_argument(
        "--ledger-path",
        default="data/clv_ledger.csv",
        help="Path to CLV ledger (default: data/clv_ledger.csv)",
    )

    args = parser.parse_args()
    target_date = args.date

    print("=" * 80)
    print("📊 AUTOMATIC RESULTS LOGGING")
    print("=" * 80)
    print(f"\n📅 Date: {target_date}")
    print(f"📋 Ledger: {args.ledger_path}")
    if args.verify:
        print("🔍 Mode: VERIFY ONLY (no changes will be saved)")
    print()

    # Initialize CLV tracker
    try:
        clv_tracker = CLVTracker(ledger_path=args.ledger_path)
    except Exception as e:
        print(f"❌ Error loading CLV ledger: {e}")
        print("   Run betting recommendations first to create ledger")
        return 1

    # Get bets for target date that don't have results yet
    ledger = clv_tracker.ledger
    pending_bets = ledger[(ledger["date"] == target_date) & (ledger["result"].isna())].copy()

    if pending_bets.empty:
        print(f"✅ All bets for {target_date} already have results")
        print("   (or no bets logged for this date)")
        return 0

    print(f"🔄 Found {len(pending_bets)} pending bets for {target_date}\n")

    # Fetch stats for each player
    settled_count = 0
    skipped_count = 0
    results_summary = []

    for idx, bet in pending_bets.iterrows():
        player_name = bet["player"]
        line = bet["line"]
        side = bet["side"]
        stake = bet.get("stake", 0.0)
        entry_odds = bet["entry_odds_dec"]

        print(f"📊 {player_name} {side} {line}")

        # Fetch actual stats
        stats = get_player_stats(player_name, target_date)

        if stats is None:
            print("   ⏳ Skipping - stats not available yet\n")
            skipped_count += 1
            continue

        # Calculate result
        actual_pra = stats["PRA"]

        if actual_pra == line:
            result_text = "PUSH"
            won = 0  # noqa: F841
        elif side == "OVER":
            result_text = "WIN" if actual_pra > line else "LOSS"
            _won = 1 if actual_pra > line else 0  # noqa: F841
        else:  # UNDER
            result_text = "WIN" if actual_pra < line else "LOSS"
            __won = 1 if actual_pra < line else 0  # noqa: F841

        # Calculate profit/loss
        if result_text == "PUSH":
            profit = 0.0
        elif result_text == "WIN":
            profit = stake * (entry_odds - 1)
        else:
            profit = -stake

        # Print result
        emoji = (
            "✅" if result_text == "WIN" else "❌" if result_text == "LOSS" else "〰️"
        )  # noqa: E501
        print(
            f"   {emoji} {actual_pra:.1f} PRA ({stats['PTS']}P + {stats['REB']}R + {stats['AST']}A)"  # noqa: E501
        )
        print(f"   💵 {result_text} | ${profit:+.2f}\n")

        # Update ledger if not in verify mode
        if not args.verify:
            clv_tracker.log_result(
                bet_id=bet["bet_id"], result=actual_pra, stake=stake if stake > 0 else None
            )

        results_summary.append(
            {
                "player": player_name,
                "side": side,
                "line": line,
                "actual": actual_pra,
                "result": result_text,
                "profit": profit,
            }
        )

        settled_count += 1

        # Rate limiting (NBA API recommends 600ms between requests)
        time.sleep(0.6)

    # Print summary
    print("=" * 80)
    print("📊 SUMMARY")
    print("=" * 80)

    if settled_count > 0:
        wins = len([r for r in results_summary if r["result"] == "WIN"])
        losses = len([r for r in results_summary if r["result"] == "LOSS"])
        pushes = len([r for r in results_summary if r["result"] == "PUSH"])
        total_profit = sum(r["profit"] for r in results_summary)

        print(f"\n✅ Processed {settled_count} bets:")
        print(f"   🎯 Record: {wins}W - {losses}L - {pushes}P")
        if wins + losses > 0:
            win_rate = (wins / (wins + losses)) * 100
            print(f"   📈 Win Rate: {win_rate:.1f}%")
        print(f"   💰 Total P/L: ${total_profit:+.2f}")

        if args.verify:
            print("\n⚠️  VERIFY MODE: No changes saved to ledger")
        else:
            print(f"\n✅ Results saved to: {args.ledger_path}")

    if skipped_count > 0:
        print(f"\n⏳ Skipped {skipped_count} bets (stats not yet available)")
        print("   💡 Run this script again in 1 - 2 hours")

    print("\n" + "=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
