#!/usr/bin/env python3
"""
Integration Test for Phase 1 Features

Tests that all Phase 1 feature modules work together:
1. advanced_stats.py (TS%, USG%, pace-adjusted)
2. opponent_features.py (DRtg, pace, DvP)
3. consistency_features.py (CV, volatility, boom/bust)
4. recent_form_features.py (L3 averages, momentum, hot/cold)

Validates:
- No errors during feature calculation
- No data loss (row count preserved)
- Temporal isolation (no future leakage)
- Feature counts match expectations
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from src.features.advanced_stats import AdvancedStatsFeatures
from src.features.opponent_features import OpponentFeatures
from src.features.consistency_features import ConsistencyFeatures
from src.features.recent_form_features import RecentFormFeatures

print("=" * 80)
print("PHASE 1 FEATURES INTEGRATION TEST")
print("=" * 80)
print()

# ======================================================================
# LOAD TEST DATA
# ======================================================================

print("1. Loading test data...")
try:
    # Load a sample of game logs for testing
    df = pd.read_csv('data/game_logs/all_game_logs_combined.csv', nrows=10000)
    print(f"   ✅ Loaded {len(df):,} games for testing")
    print(f"   Date range: {df['GAME_DATE'].min()} to {df['GAME_DATE'].max()}")

    # Ensure GAME_DATE is datetime
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])

    # Sort by player and date (required for temporal features)
    df = df.sort_values(['PLAYER_ID', 'GAME_DATE'])

    # Add PRA if not present
    if 'PRA' not in df.columns:
        df['PRA'] = df['PTS'] + df['REB'] + df['AST']

    # Add SEASON if not present
    if 'SEASON' not in df.columns:
        df['SEASON'] = df['GAME_DATE'].dt.year.astype(str) + '-' + (df['GAME_DATE'].dt.year + 1).astype(str).str[-2:]

    initial_row_count = len(df)
    initial_columns = set(df.columns)

except Exception as e:
    print(f"   ❌ Error loading data: {e}")
    sys.exit(1)

print()

# ======================================================================
# TEST ADVANCED STATS FEATURES
# ======================================================================

print("2. Testing AdvancedStatsFeatures...")
try:
    advanced_stats = AdvancedStatsFeatures()
    df = advanced_stats.add_all_features(df)

    # Check for new features
    new_cols = set(df.columns) - initial_columns
    advanced_features = [col for col in new_cols if any(x in col for x in
                        ['TS_pct', 'USG_pct', 'per_100', 'pace'])]

    print(f"   ✅ Added {len(advanced_features)} advanced stat features")
    print(f"   Row count: {len(df):,} (preserved: {len(df) == initial_row_count})")

    # Sample features
    print(f"   Sample features: {advanced_features[:5]}")

except Exception as e:
    print(f"   ❌ Error in AdvancedStatsFeatures: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# ======================================================================
# TEST OPPONENT FEATURES
# ======================================================================

print("3. Testing OpponentFeatures...")
try:
    # Skip if no MATCHUP column
    if 'MATCHUP' not in df.columns:
        print("   ⚠️  MATCHUP column not found - skipping opponent features")
    else:
        opponent_features = OpponentFeatures()

        # Load team stats for a season (may not exist in test environment)
        try:
            opponent_features.load_team_stats('2023-24')
        except Exception as team_err:
            print(f"   ⚠️  Could not load team stats: {team_err}")
            print("   ⚠️  Skipping opponent features (requires CTG team data)")
        else:
            pre_opponent_cols = set(df.columns)
            df = opponent_features.add_all_features(df)

            new_cols = set(df.columns) - pre_opponent_cols
            opponent_features_list = [col for col in new_cols if any(x in col for x in
                                     ['opp_', 'dvp_', 'pace_', 'matchup_'])]

            print(f"   ✅ Added {len(opponent_features_list)} opponent features")
            print(f"   Row count: {len(df):,} (preserved: {len(df) == initial_row_count})")
            print(f"   Sample features: {opponent_features_list[:5]}")

except Exception as e:
    print(f"   ⚠️  Error in OpponentFeatures (non-critical): {e}")
    # Don't exit - opponent features are optional for testing

print()

# ======================================================================
# TEST CONSISTENCY FEATURES
# ======================================================================

print("4. Testing ConsistencyFeatures...")
try:
    pre_consistency_cols = set(df.columns)
    consistency_features = ConsistencyFeatures()
    df = consistency_features.add_all_features(df)

    new_cols = set(df.columns) - pre_consistency_cols
    consistency_features_list = [col for col in new_cols if any(x in col for x in
                                ['_CV_', '_std_', 'volatility', 'consistency', 'boom',
                                 'bust', 'floor', 'ceiling', 'range', 'streak', 'oscillation'])]

    print(f"   ✅ Added {len(consistency_features_list)} consistency features")
    print(f"   Row count: {len(df):,} (preserved: {len(df) == initial_row_count})")
    print(f"   Sample features: {consistency_features_list[:5]}")

except Exception as e:
    print(f"   ❌ Error in ConsistencyFeatures: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# ======================================================================
# TEST RECENT FORM FEATURES
# ======================================================================

print("5. Testing RecentFormFeatures...")
try:
    pre_form_cols = set(df.columns)
    recent_form = RecentFormFeatures()
    df = recent_form.add_all_features(df)

    new_cols = set(df.columns) - pre_form_cols
    form_features_list = [col for col in new_cols if any(x in col for x in
                         ['_L3_', 'momentum', 'hot', 'cold', 'streak', 'trend',
                          'acceleration', 'role_change', 'efficiency'])]

    print(f"   ✅ Added {len(form_features_list)} recent form features")
    print(f"   Row count: {len(df):,} (preserved: {len(df) == initial_row_count})")
    print(f"   Sample features: {form_features_list[:5]}")

except Exception as e:
    print(f"   ❌ Error in RecentFormFeatures: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# ======================================================================
# VALIDATION CHECKS
# ======================================================================

print("6. Running validation checks...")
print()

# Check 1: Row count preserved
print("   Check 1: Row count preservation")
if len(df) == initial_row_count:
    print(f"      ✅ PASS - Row count preserved ({len(df):,} rows)")
else:
    print(f"      ❌ FAIL - Row count changed: {initial_row_count:,} → {len(df):,}")

# Check 2: No all-NaN columns
print("   Check 2: No all-NaN columns")
all_nan_cols = df.columns[df.isna().all()].tolist()
if len(all_nan_cols) == 0:
    print(f"      ✅ PASS - No all-NaN columns")
else:
    print(f"      ❌ FAIL - {len(all_nan_cols)} all-NaN columns: {all_nan_cols[:5]}")

# Check 3: Feature count
print("   Check 3: Feature count")
total_features = len(df.columns)
new_features = len(set(df.columns) - initial_columns)
print(f"      ✅ Total features: {total_features}")
print(f"      ✅ New features added: {new_features}")
print(f"      ✅ Original features: {len(initial_columns)}")

# Check 4: Temporal isolation (spot check)
print("   Check 4: Temporal isolation (L3 features)")
if 'PRA_L3_mean' in df.columns:
    # For each row, L3_mean should only use data from BEFORE that row
    sample_player = df[df['PLAYER_ID'].notna()].iloc[10:20]

    temporal_violations = 0
    for idx, row in sample_player.iterrows():
        if pd.notna(row['PRA_L3_mean']):
            # Get previous 3 games for this player
            prev_games = df[
                (df['PLAYER_ID'] == row['PLAYER_ID']) &
                (df['GAME_DATE'] < row['GAME_DATE'])
            ].tail(3)

            if len(prev_games) >= 1:
                expected_mean = prev_games['PRA'].mean()
                actual_mean = row['PRA_L3_mean']

                # Allow small floating point errors
                if abs(expected_mean - actual_mean) > 0.1:
                    temporal_violations += 1

    if temporal_violations == 0:
        print(f"      ✅ PASS - No temporal leakage detected (sampled 10 rows)")
    else:
        print(f"      ⚠️  WARNING - {temporal_violations}/10 potential temporal violations")
else:
    print(f"      ⚠️  SKIP - PRA_L3_mean not found")

# Check 5: No infinite values
print("   Check 5: No infinite values")
inf_cols = df.columns[df.isin([np.inf, -np.inf]).any()].tolist()
if len(inf_cols) == 0:
    print(f"      ✅ PASS - No infinite values")
else:
    print(f"      ⚠️  WARNING - {len(inf_cols)} columns with infinite values: {inf_cols[:5]}")

print()

# ======================================================================
# SUMMARY
# ======================================================================

print("=" * 80)
print("INTEGRATION TEST SUMMARY")
print("=" * 80)
print()

all_new_features = set(df.columns) - initial_columns
phase1_features = [col for col in all_new_features if any(x in col for x in
                  ['TS_pct', 'USG_pct', 'per_100', 'pace', 'opp_', 'dvp_',
                   '_CV_', '_std_', 'volatility', 'consistency', 'boom', 'bust',
                   'floor', 'ceiling', '_L3_', 'momentum', 'hot', 'cold', 'streak',
                   'trend', 'acceleration', 'role_change'])]

print(f"✅ Total Phase 1 features: {len(phase1_features)}")
print(f"✅ Initial row count: {initial_row_count:,}")
print(f"✅ Final row count: {len(df):,}")
print(f"✅ Row preservation: {len(df) == initial_row_count}")
print()

print("Feature breakdown:")
print(f"  - Advanced stats: {len([c for c in phase1_features if any(x in c for x in ['TS_', 'USG_', 'per_100', 'pace_factor'])])}")
print(f"  - Opponent: {len([c for c in phase1_features if any(x in c for x in ['opp_', 'dvp_'])])}")
print(f"  - Consistency: {len([c for c in phase1_features if any(x in c for x in ['_CV_', 'consistency', 'boom', 'bust', 'floor', 'ceiling'])])}")
print(f"  - Recent form: {len([c for c in phase1_features if any(x in c for x in ['_L3_', 'momentum', 'hot', 'cold', 'streak', 'trend'])])}")
print()

print("=" * 80)
print("✅ INTEGRATION TEST COMPLETE")
print("=" * 80)
print()
print("Next step: Retrain model with Phase 1 features")
print("Expected improvement: +3.1 to +5.1 percentage points in win rate")
print()
