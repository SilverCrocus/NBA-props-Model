"""
Feature Inventory Analysis
Detailed examination of all available features in the CTG data
"""

import pandas as pd
from pathlib import Path
from collections import defaultdict


def analyze_feature_inventory():
    """Comprehensive analysis of all features available in CTG data"""

    base_dir = Path('/Users/diyagamah/Documents/nba_props_model')
    player_data_dir = base_dir / 'data' / 'ctg_data_organized' / 'players'

    # Use most recent complete regular season
    season_dir = player_data_dir / '2024-25' / 'regular_season'

    print("\n" + "="*80)
    print("FEATURE INVENTORY - CTG PLAYER DATA")
    print("="*80)

    all_features = {}

    # 1. Offensive Overview
    print("\n### 1. OFFENSIVE OVERVIEW")
    file_path = season_dir / 'offensive_overview' / 'offensive_overview.csv'
    if file_path.exists():
        df = pd.read_csv(file_path, nrows=3)
        features = [col for col in df.columns if col not in ['Player', 'Age', 'Team', 'Pos']]
        all_features['offensive_overview'] = features
        print(f"Features ({len(features)}):")
        for feat in features:
            print(f"  - {feat}")

    # 2. Shooting Overall
    print("\n### 2. SHOOTING OVERALL")
    file_path = season_dir / 'shooting_overall' / 'shooting_overall.csv'
    if file_path.exists():
        df = pd.read_csv(file_path, nrows=3)
        features = [col for col in df.columns if col not in ['Player', 'Age', 'Team', 'Pos']]
        all_features['shooting_overall'] = features
        print(f"Features ({len(features)}):")
        for feat in features:
            print(f"  - {feat}")

    # 3. Shooting Frequency
    print("\n### 3. SHOOTING FREQUENCY (Shot Selection)")
    file_path = season_dir / 'shooting_frequency' / 'shooting_frequency.csv'
    if file_path.exists():
        df = pd.read_csv(file_path, nrows=3)
        features = [col for col in df.columns if col not in ['Player', 'Age', 'Team', 'Pos']]
        all_features['shooting_frequency'] = features
        print(f"Features ({len(features)}):")
        for feat in features:
            print(f"  - {feat}")

    # 4. Shooting Accuracy
    print("\n### 4. SHOOTING ACCURACY")
    file_path = season_dir / 'shooting_accuracy' / 'shooting_accuracy.csv'
    if file_path.exists():
        df = pd.read_csv(file_path, nrows=3)
        features = [col for col in df.columns if col not in ['Player', 'Age', 'Team', 'Pos']]
        all_features['shooting_accuracy'] = features
        print(f"Features ({len(features)}):")
        for feat in features:
            print(f"  - {feat}")

    # 5. Defense & Rebounding
    print("\n### 5. DEFENSE & REBOUNDING")
    file_path = season_dir / 'defense_rebounding' / 'defense_rebounding.csv'
    if file_path.exists():
        df = pd.read_csv(file_path, nrows=3)
        features = [col for col in df.columns if col not in ['Player', 'Age', 'Team', 'Pos']]
        all_features['defense_rebounding'] = features
        print(f"Features ({len(features)}):")
        for feat in features:
            print(f"  - {feat}")

    # 6. Foul Drawing
    print("\n### 6. FOUL DRAWING")
    file_path = season_dir / 'foul_drawing' / 'foul_drawing.csv'
    if file_path.exists():
        df = pd.read_csv(file_path, nrows=3)
        features = [col for col in df.columns if col not in ['Player', 'Age', 'Team', 'Pos']]
        all_features['foul_drawing'] = features
        print(f"Features ({len(features)}):")
        for feat in features:
            print(f"  - {feat}")

    # 7. On/Off - Efficiency & Four Factors
    print("\n### 7. ON/OFF - EFFICIENCY & FOUR FACTORS")
    file_path = season_dir / 'on_off' / 'efficiency_four_factors' / 'efficiency_four_factors.csv'
    if file_path.exists():
        df = pd.read_csv(file_path, nrows=3)
        features = [col for col in df.columns if col not in ['Player', 'Age', 'Team', 'Pos']]
        all_features['on_off_efficiency'] = features
        print(f"Features ({len(features)}):")
        for feat in features[:15]:
            print(f"  - {feat}")
        if len(features) > 15:
            print(f"  ... (+{len(features)-15} more)")

    # 8. On/Off - Team Shooting
    print("\n### 8. ON/OFF - TEAM SHOOTING FREQUENCY")
    file_path = season_dir / 'on_off' / 'team_shooting_frequency' / 'team_shooting_frequency.csv'
    if file_path.exists():
        df = pd.read_csv(file_path, nrows=3)
        features = [col for col in df.columns if col not in ['Player', 'Age', 'Team', 'Pos']]
        all_features['on_off_team_shooting_freq'] = features
        print(f"Features ({len(features)}):")
        for feat in features[:10]:
            print(f"  - {feat}")
        if len(features) > 10:
            print(f"  ... (+{len(features)-10} more)")

    # Summary
    print("\n" + "="*80)
    print("FEATURE SUMMARY")
    print("="*80)

    total_features = sum(len(v) for v in all_features.values())
    print(f"\nTotal unique features available: {total_features}")
    print(f"\nBreakdown by category:")
    for category, features in all_features.items():
        print(f"  {category}: {len(features)} features")

    # Identify PRA-relevant features
    print("\n" + "="*80)
    print("PRA-RELEVANT FEATURES")
    print("="*80)

    print("\n### Direct PRA Components:")
    print("  - Points: Available from game logs (PTS)")
    print("  - Rebounds: Available from game logs (REB)")
    print("  - Assists: Available from game logs (AST)")

    print("\n### Key Predictor Features from CTG:")
    print("\n1. Usage & Scoring Opportunity:")
    print("  - Usage (% of team possessions)")
    print("  - PSA (Points Scored per 100 Shooting Attempts)")
    print("  - MIN (Minutes played)")

    print("\n2. Assist Generation:")
    print("  - AST% (Assist percentage)")
    print("  - AST:Usg (Assist to usage ratio)")

    print("\n3. Rebounding:")
    print("  - fgOR% (Field goal offensive rebound %)")
    print("  - fgDR% (Field goal defensive rebound %)")
    print("  - ftOR% (Free throw offensive rebound %)")
    print("  - ftDR% (Free throw defensive rebound %)")

    print("\n4. Shooting Efficiency:")
    print("  - eFG% (Effective field goal %)")
    print("  - Rim accuracy")
    print("  - Three-point accuracy")
    print("  - Shot frequency by zone")

    print("\n5. Foul Drawing (Points via FT):")
    print("  - SFLD% (Shooting foul drawn %)")
    print("  - FFLD% (Free throw foul drawn %)")
    print("  - FT% (Free throw percentage)")

    print("\n6. On/Off Context:")
    print("  - Team offensive rating with/without player")
    print("  - Team pace with/without player")
    print("  - Team shooting impact")

    # Game Log Features
    print("\n### Game Log Features:")
    game_logs_file = base_dir / 'data' / 'game_logs' / 'all_game_logs_combined.csv'
    if game_logs_file.exists():
        df = pd.read_csv(game_logs_file, nrows=1)
        game_log_features = list(df.columns)
        print(f"\nTotal game log columns: {len(game_log_features)}")
        print("\nKey game log features:")
        key_features = ['PTS', 'REB', 'AST', 'MIN', 'FGM', 'FGA', 'FG3M', 'FG3A',
                       'FTM', 'FTA', 'OREB', 'DREB', 'STL', 'BLK', 'TOV', 'PF', 'PLUS_MINUS']
        for feat in key_features:
            if feat in game_log_features:
                print(f"  - {feat}")

    # Feature Engineering Opportunities
    print("\n" + "="*80)
    print("FEATURE ENGINEERING OPPORTUNITIES")
    print("="*80)

    print("\n### Temporal Features (from game logs):")
    print("  - Rolling averages (3, 5, 10 game windows)")
    print("  - Exponentially weighted moving averages")
    print("  - Trend indicators (improving/declining)")
    print("  - Recent form (last 5 games)")
    print("  - Days of rest")
    print("  - Back-to-back game indicator")

    print("\n### Contextual Features:")
    print("  - Home/Away")
    print("  - Opponent strength (defensive rating)")
    print("  - Team pace (possessions per game)")
    print("  - Player position matchup")
    print("  - Season phase (early/mid/late)")

    print("\n### Interaction Features:")
    print("  - Usage × Minutes (opportunity index)")
    print("  - Shooting efficiency × Shot frequency")
    print("  - Assist rate × Team pace")
    print("  - Rebounding % × Minutes")

    print("\n### Season Aggregates (from CTG):")
    print("  - Season-to-date averages")
    print("  - Per-100 possession stats")
    print("  - Percentile ranks vs position")
    print("  - Home vs away splits")

    return all_features


if __name__ == "__main__":
    analyze_feature_inventory()
