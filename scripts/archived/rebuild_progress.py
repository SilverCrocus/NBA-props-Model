#!/usr/bin/env python3
"""
Rebuild progress tracker from existing downloaded files
"""

import json
from pathlib import Path
from datetime import datetime

def rebuild_progress():
    """Scan existing files and rebuild progress tracker"""
    
    base_dir = Path("/Users/diyagamah/Documents/nba_props_model/data/ctg_data_organized")
    tracking_dir = base_dir / "tracking"
    progress_file = tracking_dir / "download_progress.json"
    
    # All possible combinations
    categories = [
        "Offensive Overview",
        "Shooting: Overall",
        "Shooting: Frequency", 
        "Shooting: Accuracy",
        "Foul Drawing",
        "Defense and Rebounding",
        "On/Off Efficiency & Four Factors",
        "On/Off Team Shooting: Frequency",
        "On/Off Team Shooting: Accuracy",
        "On/Off Team Halfcourt & Putbacks",
        "On/Off Team Transition",
        "On/Off Opponent Shooting: Frequency",
        "On/Off Opponent Shooting: Accuracy",
        "On/Off Opponent Halfcourt & Putbacks",
        "On/Off Opponent Transition"
    ]
    
    seasons = [
        "2024-25", "2023-24", "2022-23", "2021-22", "2020-21",
        "2019-20", "2018-19", "2017-18", "2016-17", "2015-16",
        "2014-15", "2013-14", "2012-13", "2011-12", "2010-11",
        "2009-10", "2008-09", "2007-08", "2006-07", "2005-06",
        "2004-05", "2003-04"
    ]
    
    season_types = ["Regular Season", "Playoffs"]
    
    print("🔍 Scanning existing CTG files...")
    print("=" * 60)
    
    # Find all CSV files
    csv_files = list(base_dir.glob("players/**/*.csv"))
    print(f"Found {len(csv_files)} CSV files")
    
    # Build completed list from existing files
    completed = []
    file_mapping = {}
    
    for csv_file in csv_files:
        # Parse the path to get season, type, and category
        parts = csv_file.relative_to(base_dir).parts
        
        if len(parts) >= 4 and parts[0] == "players":
            season = parts[1]  # e.g., "2024-25"
            season_type_folder = parts[2]  # e.g., "regular_season" or "playoffs"
            
            # Convert folder name to proper season type
            if season_type_folder == "regular_season":
                season_type = "Regular Season"
            elif season_type_folder == "playoffs":
                season_type = "Playoffs"
            else:
                continue
            
            # Get category from filename/path
            if parts[3] == "on_off" and len(parts) >= 5:
                # On/Off subcategory
                subcategory = parts[4]
                # Map back to original category name
                category_map = {
                    "efficiency_four_factors": "On/Off Efficiency & Four Factors",
                    "team_shooting_frequency": "On/Off Team Shooting: Frequency",
                    "team_shooting_accuracy": "On/Off Team Shooting: Accuracy",
                    "team_halfcourt_putbacks": "On/Off Team Halfcourt & Putbacks",
                    "team_transition": "On/Off Team Transition",
                    "opponent_shooting_frequency": "On/Off Opponent Shooting: Frequency",
                    "opponent_shooting_accuracy": "On/Off Opponent Shooting: Accuracy",
                    "opponent_halfcourt_putbacks": "On/Off Opponent Halfcourt & Putbacks",
                    "opponent_transition": "On/Off Opponent Transition"
                }
                category = category_map.get(subcategory)
            else:
                # Regular category
                category_folder = parts[3]
                category_map = {
                    "offensive_overview": "Offensive Overview",
                    "shooting_overall": "Shooting: Overall",
                    "shooting_frequency": "Shooting: Frequency",
                    "shooting_accuracy": "Shooting: Accuracy",
                    "foul_drawing": "Foul Drawing",
                    "defense_rebounding": "Defense and Rebounding"
                }
                category = category_map.get(category_folder)
            
            if category and season in seasons:
                completed_entry = (category, season, season_type)
                completed.append(list(completed_entry))
                
                key = f"{season}_{season_type}_{category}"
                file_mapping[key] = str(csv_file)
    
    # Calculate what's missing
    all_combinations = []
    for season in seasons:
        for season_type in season_types:
            for category in categories:
                all_combinations.append((category, season, season_type))
    
    completed_set = set(tuple(x) for x in completed)
    all_set = set(all_combinations)
    missing_set = all_set - completed_set
    
    # Save progress file
    progress_data = {
        "completed": completed,
        "total_expected": len(all_combinations),
        "last_updated": datetime.now().isoformat(),
        "completion_rate": f"{len(completed) * 100 / len(all_combinations):.1f}%"
    }
    
    with open(progress_file, 'w') as f:
        json.dump(progress_data, f, indent=2)
    
    print(f"\n✅ Progress rebuilt successfully!")
    print(f"   Completed: {len(completed)}/{len(all_combinations)} files ({progress_data['completion_rate']})")
    
    # Show breakdown by season
    print(f"\n📊 Breakdown by season:")
    for season in seasons[:5]:  # Show first 5 seasons
        season_count = sum(1 for c in completed if c[1] == season)
        print(f"   {season}: {season_count}/30 files")
    
    # Show what's missing for current seasons
    print(f"\n🔴 Missing files (showing first 10):")
    missing_sorted = sorted(list(missing_set), key=lambda x: (x[1], x[2], x[0]))
    for i, (cat, season, stype) in enumerate(missing_sorted[:10]):
        print(f"   {i+1}. {season} - {stype} - {cat}")
    
    if len(missing_sorted) > 10:
        print(f"   ... and {len(missing_sorted) - 10} more")
    
    print(f"\n📝 Progress saved to: {progress_file}")
    print(f"✅ Now run: python3 ctg_robust_scraper.py")
    print(f"   It will skip the {len(completed)} files you already have!")
    
    return completed, list(missing_set)

if __name__ == "__main__":
    rebuild_progress()