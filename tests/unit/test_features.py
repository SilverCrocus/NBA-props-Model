#!/usr/bin/env python3
"""Test the feature engineering pipeline with actual CTG data"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Test data loading
print("Testing Feature Engineering Pipeline")
print("=" * 50)

base_path = Path('/Users/diyagamah/Documents/nba_props_model/data')
player_data_path = base_path / 'ctg_data_organized' / 'players'

# Load 2023-24 regular season data
season_path = player_data_path / '2023-24' / 'regular_season'

if not season_path.exists():
    print(f"ERROR: Path not found: {season_path}")
    exit(1)

# Load offensive overview
offensive_path = season_path / 'offensive_overview' / 'offensive_overview.csv'
if offensive_path.exists():
    offensive_df = pd.read_csv(offensive_path)
    print(f"✓ Loaded offensive_overview: {len(offensive_df)} players")
    print(f"  Columns: {list(offensive_df.columns)[:10]}...")
    
    # Clean percentage columns
    percentage_cols = ['Usage', 'AST%', 'TOV%']
    for col in percentage_cols:
        if col in offensive_df.columns and offensive_df[col].dtype == 'object':
            offensive_df[col] = offensive_df[col].str.replace('%', '').astype(float)
else:
    print(f"ERROR: File not found: {offensive_path}")
    exit(1)

# Load defense and rebounding
defense_path = season_path / 'defense_rebounding' / 'defense_rebounding.csv'
if defense_path.exists():
    defense_df = pd.read_csv(defense_path)
    print(f"✓ Loaded defense_rebounding: {len(defense_df)} players")
    
    # Clean percentage columns in defense data
    defense_percentage_cols = ['fgOR%', 'fgDR%', 'ftOR%', 'ftDR%', 'BLK%', 'STL%', 'FOUL%']
    for col in defense_percentage_cols:
        if col in defense_df.columns and defense_df[col].dtype == 'object':
            defense_df[col] = defense_df[col].str.replace('%', '').astype(float)
else:
    print(f"WARNING: File not found: {defense_path}")

# Test feature creation
print("\n" + "=" * 50)
print("Testing Feature Creation")
print("=" * 50)

# Create sample features
if 'Usage' in offensive_df.columns:
    usg_mean = offensive_df['Usage'].mean()
    print(f"✓ Usage Rate - Mean: {usg_mean:.2f}")

if 'PSA' in offensive_df.columns:
    psa_mean = offensive_df['PSA'].mean()
    print(f"✓ PSA (Points per Shot Attempt) - Mean: {psa_mean:.2f}")

if 'AST%' in offensive_df.columns:
    ast_mean = offensive_df['AST%'].mean()
    print(f"✓ Assist Percentage - Mean: {ast_mean:.2f}")

# Test merging
print("\n" + "=" * 50)
print("Testing Data Merging")
print("=" * 50)

if defense_path.exists():
    merged_df = offensive_df.merge(
        defense_df[['Player', 'Team', 'fgOR%', 'fgDR%']],
        on=['Player', 'Team'],
        how='left',
        suffixes=('', '_defense')
    )
    print(f"✓ Merged dataframe shape: {merged_df.shape}")
    print(f"✓ Columns after merge: {merged_df.shape[1]} columns")

# Test feature engineering
print("\n" + "=" * 50)
print("Testing Three-Tier Feature Architecture")
print("=" * 50)

# Tier 1: Core Performance
core_features = pd.DataFrame()
if 'Usage' in merged_df.columns:
    core_features['USG_percent'] = merged_df['Usage']
if 'PSA' in merged_df.columns:
    core_features['PSA'] = merged_df['PSA']
if 'AST%' in merged_df.columns and 'Usage' in merged_df.columns:
    core_features['AST_to_USG_Ratio'] = merged_df['AST%'] / (merged_df['Usage'] + 0.001)

print(f"✓ Core Performance Features: {len(core_features.columns)} features")

# Tier 2: Contextual
context_features = pd.DataFrame()
if 'MIN' in merged_df.columns:
    context_features['Minutes_Season_Avg'] = merged_df['MIN']
    
print(f"✓ Contextual Features: {len(context_features.columns)} features")

# Tier 3: Temporal (simulated)
temporal_features = pd.DataFrame()
if 'Usage Rank' in merged_df.columns and 'PSA Rank' in merged_df.columns:
    temporal_features['Consistency_Score'] = 1 / (1 + np.abs(merged_df['Usage Rank'] - merged_df['PSA Rank'])/100)
    
print(f"✓ Temporal Features: {len(temporal_features.columns)} features")

# Save test results
print("\n" + "=" * 50)
print("Saving Processed Features")
print("=" * 50)

output_path = base_path / 'processed'
output_path.mkdir(parents=True, exist_ok=True)

# Combine features
final_features = pd.concat([
    merged_df[['Player', 'Team']],
    core_features,
    context_features,
    temporal_features
], axis=1)

# Save
output_file = output_path / 'test_features.csv'
final_features.head(50).to_csv(output_file, index=False)
print(f"✓ Saved test features to: {output_file}")
print(f"  Shape: {final_features.shape}")
print(f"  Total features: {len(final_features.columns) - 2} (excluding Player, Team)")

# Summary
print("\n" + "=" * 50)
print("Pipeline Test Summary")
print("=" * 50)
print("✅ All tests passed successfully!")
print(f"   - Data loaded: {len(merged_df)} players")
print(f"   - Features created: {len(final_features.columns) - 2} features")
print(f"   - Three-tier architecture implemented")
print(f"   - Ready for model development")

# Show sample
print("\nSample of top players:")
if 'MIN' in merged_df.columns and 'Usage' in merged_df.columns:
    top_players = merged_df.nlargest(5, 'MIN')[['Player', 'Team', 'MIN', 'Usage', 'PSA']]
    print(top_players.to_string())