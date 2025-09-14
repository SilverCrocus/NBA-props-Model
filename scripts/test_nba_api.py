#!/usr/bin/env python3
"""
Quick test to verify NBA API is working before running full fetch
"""

import sys
try:
    from nba_api.stats.endpoints import leaguegamelog
    print("✓ NBA API imported successfully")
except ImportError:
    print("✗ NBA API not installed. Please run: uv add nba-api")
    sys.exit(1)

import time

def test_api():
    """Test NBA API with a small request"""
    print("\nTesting NBA API connection...")

    try:
        # Fetch just one day of games as a test
        game_log = leaguegamelog.LeagueGameLog(
            season='2023-24',
            season_type_all_star='Regular Season',
            player_or_team_abbreviation='P',
            date_from_nullable='03/01/2024',
            date_to_nullable='03/01/2024'
        )

        df = game_log.get_data_frames()[0]

        print(f"✓ API call successful!")
        print(f"✓ Retrieved {len(df)} player performances from March 1, 2024")

        if len(df) > 0:
            # Show sample data
            print("\nSample data (first 3 rows):")
            sample_cols = ['PLAYER_NAME', 'TEAM_ABBREVIATION', 'PTS', 'REB', 'AST']
            print(df[sample_cols].head(3).to_string())

            # Calculate PRA
            df['PRA'] = df['PTS'] + df['REB'] + df['AST']
            print(f"\nPRA Statistics for March 1, 2024:")
            print(f"  Mean PRA: {df['PRA'].mean():.2f}")
            print(f"  Max PRA: {df['PRA'].max():.2f}")
            best_game = df.loc[df['PRA'].idxmax()]
            print(f"  Best: {best_game['PLAYER_NAME']} - {best_game['PRA']:.0f} PRA")

        print("\n✅ NBA API is working correctly!")
        print("You can now run: uv run scripts/fetch_all_game_logs.py")
        return True

    except Exception as e:
        print(f"\n✗ API test failed: {str(e)}")
        print("\nPossible issues:")
        print("1. No internet connection")
        print("2. NBA API servers are down")
        print("3. Rate limiting (wait a few minutes)")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("NBA API CONNECTION TEST")
    print("=" * 60)

    success = test_api()
    sys.exit(0 if success else 1)