# Phase 1 Implementation Guide - Detailed Technical Specifications

**Target:** 51.40% → 53.5-54.5% win rate
**Timeline:** 3 days
**Features:** 5 features (Opponent DRtg, L3 Form, Minutes EWMA, True Shooting, Usage x Pace)

---

## Day 1: Opponent Defense + L3 Features

### Feature 1: Opponent Defensive Rating (Position-Specific)

#### Step 1.1: Load CTG Team Data

```python
# File: src/features/opponent_defense.py

import pandas as pd
from pathlib import Path
from typing import Dict, Optional

class OpponentDefenseFeatures:
    """Calculate opponent defensive features from CTG team data"""

    def __init__(self, data_path: str = "/Users/diyagamah/Documents/nba_props_model/data"):
        self.data_path = Path(data_path)
        self.ctg_team_path = self.data_path / "ctg_team_data"
        self.team_data_cache = {}  # Cache team data by season

    def _load_team_data(self, season: str) -> pd.DataFrame:
        """
        Load CTG team defensive data for a season

        CTG team data files:
        - data/ctg_team_data/2024-25/defense_overview.csv
        - data/ctg_team_data/2024-25/pace_stats.csv

        Returns:
            DataFrame with team defensive stats
        """
        cache_key = season
        if cache_key in self.team_data_cache:
            return self.team_data_cache[cache_key]

        season_path = self.ctg_team_path / season

        # Load defense overview
        defense_file = season_path / "defense_overview.csv"
        if defense_file.exists():
            defense_df = pd.read_csv(defense_file)
        else:
            print(f"Warning: {defense_file} not found, using league averages")
            return pd.DataFrame()

        # Load pace stats
        pace_file = season_path / "pace_stats.csv"
        if pace_file.exists():
            pace_df = pd.read_csv(pace_file)
            # Merge defense + pace
            team_df = defense_df.merge(pace_df, on='Team', how='left')
        else:
            team_df = defense_df

        # Cache for future use
        self.team_data_cache[cache_key] = team_df

        return team_df

    def get_opponent_features(
        self,
        opponent: str,
        player_position: str,
        season: str = "2024-25"
    ) -> Dict[str, float]:
        """
        Get opponent defensive features adjusted for player position

        Args:
            opponent: Team abbreviation (e.g., 'LAL', 'BOS')
            player_position: 'Guard', 'Wing', or 'Big'
            season: Season string (e.g., '2024-25')

        Returns:
            Dictionary with opponent defensive features:
            - OPP_DRtg: Overall defensive rating
            - OPP_DRtg_Adjusted: Position-adjusted defensive rating
            - OPP_Pace: Team pace (possessions per 48 min)
            - OPP_ORB_PCT_Allowed: Offensive rebounding % allowed
            - OPP_DRB_PCT: Defensive rebounding %
        """
        # Load team data
        team_df = self._load_team_data(season)

        if team_df.empty:
            # Return league averages if data not available
            return self._get_league_averages()

        # Find opponent team
        opp_data = team_df[team_df['Team'] == opponent]

        if opp_data.empty:
            # Team not found, use league averages
            return self._get_league_averages()

        opp_data = opp_data.iloc[0]

        # Base features
        features = {
            'OPP_DRtg': opp_data.get('DRtg', 112.0),
            'OPP_Pace': opp_data.get('Pace', 100.0),
            'OPP_ORB_PCT_Allowed': opp_data.get('ORB_PCT_Allowed', 25.0),
            'OPP_DRB_PCT': opp_data.get('DRB_PCT', 75.0)
        }

        # Position-specific defensive rating
        # If CTG has position-specific data, use it
        if 'DRtg_vs_Guards' in opp_data:
            features['OPP_DRtg_vs_Guards'] = opp_data['DRtg_vs_Guards']
            features['OPP_DRtg_vs_Wings'] = opp_data.get('DRtg_vs_Wings', features['OPP_DRtg'])
            features['OPP_DRtg_vs_Bigs'] = opp_data.get('DRtg_vs_Bigs', features['OPP_DRtg'])
        else:
            # Use overall DRtg for all positions
            features['OPP_DRtg_vs_Guards'] = features['OPP_DRtg']
            features['OPP_DRtg_vs_Wings'] = features['OPP_DRtg']
            features['OPP_DRtg_vs_Bigs'] = features['OPP_DRtg']

        # Select position-adjusted DRtg
        position_map = {
            'Guard': 'OPP_DRtg_vs_Guards',
            'Wing': 'OPP_DRtg_vs_Wings',
            'Big': 'OPP_DRtg_vs_Bigs'
        }

        drtg_key = position_map.get(player_position, 'OPP_DRtg')
        features['OPP_DRtg_Adjusted'] = features[drtg_key]

        # Normalize DRtg (league average = 112.0)
        # Negative = good defense, positive = bad defense
        features['OPP_DRtg_vs_Avg'] = features['OPP_DRtg_Adjusted'] - 112.0

        return features

    def _get_league_averages(self) -> Dict[str, float]:
        """Return league average defensive stats"""
        return {
            'OPP_DRtg': 112.0,
            'OPP_DRtg_Adjusted': 112.0,
            'OPP_DRtg_vs_Avg': 0.0,
            'OPP_Pace': 100.0,
            'OPP_ORB_PCT_Allowed': 25.0,
            'OPP_DRB_PCT': 75.0,
            'OPP_DRtg_vs_Guards': 112.0,
            'OPP_DRtg_vs_Wings': 112.0,
            'OPP_DRtg_vs_Bigs': 112.0
        }
```

#### Step 1.2: Integration into Walk-Forward Validation

```python
# File: scripts/validation/walk_forward_validation_phase1.py

from src.features.opponent_defense import OpponentDefenseFeatures

# Initialize
opp_defense = OpponentDefenseFeatures()

# In walk-forward loop:
for pred_date in unique_dates:
    games_today = raw_gamelogs[raw_gamelogs['GAME_DATE'] == pred_date]

    for idx, row in games_today.iterrows():
        player_id = row['PLAYER_ID']
        opponent = row['OPPONENT']
        player_position = row.get('POSITION', 'Wing')  # Default to Wing if missing

        # Get opponent defensive features
        opp_def_features = opp_defense.get_opponent_features(
            opponent=opponent,
            player_position=player_position,
            season="2024-25"
        )

        # Add to feature vector
        all_features.update(opp_def_features)
```

---

### Feature 2: L3 Recent Form Features

#### Step 2.1: Calculate L3 Features

```python
# File: src/features/recent_form.py

import pandas as pd
import numpy as np
from typing import Dict

class RecentFormFeatures:
    """Calculate L3 (last 3 games) recent form features"""

    def calculate_l3_features(self, player_history: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate last 3 games features (strongest temporal signal)

        Args:
            player_history: DataFrame with player's games sorted by date (most recent first)
                           Must have columns: PRA, MIN, Usage, GAME_DATE

        Returns:
            Dictionary with L3 features:
            - PRA_L3_mean: Average PRA last 3 games
            - PRA_L3_std: Volatility in last 3 games
            - PRA_L3_trend: Trend (is player hot or cold?)
            - MIN_L3_mean: Average minutes last 3 games
            - USG_L3_mean: Average usage last 3 games
            - PRA_L3_vs_L10_ratio: Hot/cold indicator
        """
        if len(player_history) < 3:
            # Not enough history
            return {
                'PRA_L3_mean': 0,
                'PRA_L3_std': 0,
                'PRA_L3_trend': 0,
                'MIN_L3_mean': 0,
                'USG_L3_mean': 0,
                'PRA_L3_vs_L10_ratio': 1.0,
                'PRA_L3_max': 0,
                'PRA_L3_min': 0
            }

        # Get last 3 games (already sorted by date, most recent first)
        last_3 = player_history.iloc[:3]

        # Basic stats
        pra_l3_mean = last_3['PRA'].mean()
        pra_l3_std = last_3['PRA'].std() if len(last_3) > 1 else 0
        min_l3_mean = last_3['MIN'].mean()

        # Trend: (most recent - oldest) / 2
        # Positive = trending up, negative = trending down
        pra_l3_trend = (last_3.iloc[0]['PRA'] - last_3.iloc[2]['PRA']) / 2

        # Usage (if available)
        if 'Usage' in last_3.columns:
            usg_l3_mean = last_3['Usage'].mean()
        else:
            usg_l3_mean = 0

        # Hot/cold indicator: L3 vs L10 ratio
        # >1.1 = player is hot, <0.9 = player is cold
        if len(player_history) >= 10:
            last_10 = player_history.iloc[:10]
            pra_l10_mean = last_10['PRA'].mean()
            if pra_l10_mean > 0:
                pra_l3_vs_l10_ratio = pra_l3_mean / pra_l10_mean
            else:
                pra_l3_vs_l10_ratio = 1.0
        else:
            pra_l3_vs_l10_ratio = 1.0

        # Max and min in L3 (volatility indicators)
        pra_l3_max = last_3['PRA'].max()
        pra_l3_min = last_3['PRA'].min()

        return {
            'PRA_L3_mean': pra_l3_mean,
            'PRA_L3_std': pra_l3_std,
            'PRA_L3_trend': pra_l3_trend,
            'MIN_L3_mean': min_l3_mean,
            'USG_L3_mean': usg_l3_mean,
            'PRA_L3_vs_L10_ratio': pra_l3_vs_l10_ratio,
            'PRA_L3_max': pra_l3_max,
            'PRA_L3_min': pra_l3_min
        }

    def calculate_l3_per_component(self, player_history: pd.DataFrame) -> Dict[str, float]:
        """Calculate L3 for each component (PTS, REB, AST separately)"""
        if len(player_history) < 3:
            return {
                'PTS_L3_mean': 0,
                'REB_L3_mean': 0,
                'AST_L3_mean': 0
            }

        last_3 = player_history.iloc[:3]

        return {
            'PTS_L3_mean': last_3['PTS'].mean() if 'PTS' in last_3.columns else 0,
            'REB_L3_mean': last_3['REB'].mean() if 'REB' in last_3.columns else 0,
            'AST_L3_mean': last_3['AST'].mean() if 'AST' in last_3.columns else 0
        }
```

#### Step 2.2: Integration

```python
# In walk-forward loop:
recent_form = RecentFormFeatures()

for pred_date in unique_dates:
    for idx, row in games_today.iterrows():
        # Get player's history (sorted by date, most recent first)
        player_history = past_games[past_games['PLAYER_ID'] == player_id]
        player_history = player_history.sort_values('GAME_DATE', ascending=False)

        # Calculate L3 features
        l3_features = recent_form.calculate_l3_features(player_history)
        l3_component_features = recent_form.calculate_l3_per_component(player_history)

        # Add to feature vector
        all_features.update(l3_features)
        all_features.update(l3_component_features)
```

---

### Day 1 Testing

```python
# File: tests/unit/test_day1_features.py

import pytest
import pandas as pd
from src.features.opponent_defense import OpponentDefenseFeatures
from src.features.recent_form import RecentFormFeatures

def test_opponent_defense_features():
    """Test opponent defense feature calculation"""
    opp_defense = OpponentDefenseFeatures()

    # Test with real opponent
    features = opp_defense.get_opponent_features(
        opponent='LAL',
        player_position='Guard',
        season='2024-25'
    )

    # Should return all required features
    assert 'OPP_DRtg' in features
    assert 'OPP_DRtg_Adjusted' in features
    assert 'OPP_Pace' in features
    assert 'OPP_DRtg_vs_Avg' in features

    # DRtg should be reasonable (100-120)
    assert 100 <= features['OPP_DRtg'] <= 120

def test_l3_recent_form():
    """Test L3 recent form calculation"""
    recent_form = RecentFormFeatures()

    # Create sample player history
    player_history = pd.DataFrame({
        'GAME_DATE': pd.date_range('2024-01-01', periods=10, freq='2D'),
        'PRA': [20, 25, 30, 22, 28, 26, 24, 23, 27, 29],
        'MIN': [30, 32, 35, 28, 33, 31, 30, 29, 34, 36],
        'Usage': [25, 26, 27, 24, 26, 25, 25, 24, 27, 28]
    }).sort_values('GAME_DATE', ascending=False)

    features = recent_form.calculate_l3_features(player_history)

    # Should calculate L3 mean correctly
    expected_l3_mean = (29 + 27 + 23) / 3  # Last 3 games
    assert abs(features['PRA_L3_mean'] - expected_l3_mean) < 0.01

    # Trend should be positive (29 - 23) / 2 = 3
    expected_trend = (29 - 23) / 2
    assert abs(features['PRA_L3_trend'] - expected_trend) < 0.01

    # Hot/cold ratio should be >1 (L3 > L10)
    l3_mean = (29 + 27 + 23) / 3
    l10_mean = player_history.iloc[:10]['PRA'].mean()
    expected_ratio = l3_mean / l10_mean
    assert abs(features['PRA_L3_vs_L10_ratio'] - expected_ratio) < 0.01
```

---

## Day 2: Minutes Projection + True Shooting %

### Feature 3: Improved Minutes Projection (EWMA)

```python
# File: src/features/minutes_projection.py

import pandas as pd
import numpy as np
from typing import Dict
from scipy import stats as scipy_stats

class MinutesProjection:
    """Advanced minutes projection using EWMA + trend detection"""

    def calculate_minutes_features(
        self,
        player_history: pd.DataFrame,
        ewma_span: int = 5
    ) -> Dict[str, float]:
        """
        Calculate minutes projection with trend adjustment

        Args:
            player_history: DataFrame sorted by date (most recent first)
            ewma_span: Span for EWMA (default 5 games)

        Returns:
            Dictionary with minutes features:
            - MIN_projected: Projected minutes for next game
            - MIN_ewma5: EWMA of last 5 games
            - MIN_trend: Linear trend (positive = increasing role)
            - MIN_volatility: Standard deviation of minutes
            - MIN_L3_mean: Last 3 games average
        """
        if len(player_history) < 3:
            return {
                'MIN_projected': 0,
                'MIN_ewma5': 0,
                'MIN_trend': 0,
                'MIN_volatility': 0,
                'MIN_L3_mean': 0,
                'MIN_L5_mean': 0
            }

        # Ensure sorted by date (most recent first)
        history = player_history.sort_values('GAME_DATE', ascending=False)

        # EWMA (exponentially weighted moving average)
        # Recent games weighted higher
        min_ewma = history['MIN'].ewm(span=ewma_span, min_periods=1).mean().iloc[0]

        # L3 and L5 averages
        min_l3_mean = history.iloc[:3]['MIN'].mean() if len(history) >= 3 else 0
        min_l5_mean = history.iloc[:5]['MIN'].mean() if len(history) >= 5 else 0

        # Trend detection: Linear regression on last 5-10 games
        # Positive slope = increasing role, negative = decreasing role
        if len(history) >= 5:
            last_n = min(10, len(history))
            minutes_recent = history.iloc[:last_n]['MIN'].values
            x = np.arange(last_n)

            # Fit linear regression (most recent = x[0])
            slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(x, minutes_recent[::-1])
            min_trend = slope
        else:
            min_trend = 0

        # Volatility (standard deviation)
        if len(history) >= 10:
            min_volatility = history.iloc[:10]['MIN'].std()
        else:
            min_volatility = history['MIN'].std() if len(history) > 1 else 0

        # Projected minutes: EWMA + trend adjustment
        # If trend is positive (increasing role), project higher
        # If trend is negative (decreasing role), project lower
        min_projected = min_ewma + (min_trend * 0.3)  # 30% weight on trend

        # Cap at reasonable bounds (0-48 minutes)
        min_projected = np.clip(min_projected, 0, 48)

        # Also create a "confidence" score based on volatility
        # Low volatility = high confidence, high volatility = low confidence
        min_confidence = 1.0 / (1.0 + min_volatility)

        return {
            'MIN_projected': min_projected,
            'MIN_ewma5': min_ewma,
            'MIN_trend': min_trend,
            'MIN_volatility': min_volatility,
            'MIN_L3_mean': min_l3_mean,
            'MIN_L5_mean': min_l5_mean,
            'MIN_confidence': min_confidence
        }

    def detect_role_change(self, player_history: pd.DataFrame) -> Dict[str, float]:
        """
        Detect if player's role has changed recently

        Returns:
            - role_change_indicator: 1 if role increased, -1 if decreased, 0 if stable
            - role_change_magnitude: How much role changed (in minutes)
        """
        if len(player_history) < 10:
            return {
                'role_change_indicator': 0,
                'role_change_magnitude': 0
            }

        history = player_history.sort_values('GAME_DATE', ascending=False)

        # Compare L3 vs L7-10 (recent vs medium-term)
        l3_mean = history.iloc[:3]['MIN'].mean()
        l7_10_mean = history.iloc[6:10]['MIN'].mean()

        role_change_magnitude = l3_mean - l7_10_mean

        # Threshold for "significant" role change: 5 minutes
        if role_change_magnitude > 5:
            role_change_indicator = 1  # Role increased
        elif role_change_magnitude < -5:
            role_change_indicator = -1  # Role decreased
        else:
            role_change_indicator = 0  # Stable

        return {
            'role_change_indicator': role_change_indicator,
            'role_change_magnitude': role_change_magnitude
        }
```

---

### Feature 4: True Shooting % (Game-Level)

```python
# File: src/features/scoring_efficiency.py

import pandas as pd
import numpy as np
from typing import Dict

class ScoringEfficiency:
    """Calculate True Shooting % and other efficiency metrics"""

    def calculate_true_shooting(self, player_history: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate True Shooting % (TS%)

        TS% = PTS / (2 * (FGA + 0.44 * FTA))

        Better than FG% because it accounts for:
        - 3-pointers (worth more than 2s)
        - Free throws

        Args:
            player_history: DataFrame sorted by date (most recent first)

        Returns:
            Dictionary with TS% features
        """
        if len(player_history) == 0:
            return self._get_default_ts_features()

        history = player_history.sort_values('GAME_DATE', ascending=False)

        # Calculate TS% for each game
        history = history.copy()
        history['TS_PCT'] = history['PTS'] / (2 * (history['FGA'] + 0.44 * history['FTA']))

        # Handle division by zero
        history['TS_PCT'] = history['TS_PCT'].replace([np.inf, -np.inf], np.nan)
        history['TS_PCT'] = history['TS_PCT'].fillna(0)

        # Cap at reasonable bounds (0% to 100%)
        history['TS_PCT'] = history['TS_PCT'].clip(0, 1.0)

        # Rolling averages
        ts_pct_l3 = history.iloc[:3]['TS_PCT'].mean() if len(history) >= 3 else 0
        ts_pct_l5 = history.iloc[:5]['TS_PCT'].mean() if len(history) >= 5 else 0
        ts_pct_l10 = history.iloc[:10]['TS_PCT'].mean() if len(history) >= 10 else 0

        # EWMA
        if len(history) >= 5:
            ts_pct_ewma5 = history['TS_PCT'].ewm(span=5, min_periods=1).mean().iloc[0]
        else:
            ts_pct_ewma5 = 0

        # Trend: Is efficiency improving or declining?
        if len(history) >= 5:
            ts_trend = ts_pct_l3 - ts_pct_l10
        else:
            ts_trend = 0

        return {
            'TS_PCT_L3': ts_pct_l3,
            'TS_PCT_L5': ts_pct_l5,
            'TS_PCT_L10': ts_pct_l10,
            'TS_PCT_ewma5': ts_pct_ewma5,
            'TS_PCT_trend': ts_trend
        }

    def calculate_scoring_breakdown(self, player_history: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate scoring breakdown (2PT, 3PT, FT percentages)

        Returns:
            - PTS_from_2PT_pct: % of points from 2-pointers
            - PTS_from_3PT_pct: % of points from 3-pointers
            - PTS_from_FT_pct: % of points from free throws
        """
        if len(player_history) == 0:
            return {
                'PTS_from_2PT_pct': 0,
                'PTS_from_3PT_pct': 0,
                'PTS_from_FT_pct': 0
            }

        history = player_history.sort_values('GAME_DATE', ascending=False)

        # Get last 5 games
        last_5 = history.iloc[:5] if len(history) >= 5 else history

        # Calculate points from each source
        pts_from_2pt = (last_5['FGM'] - last_5['FG3M']) * 2
        pts_from_3pt = last_5['FG3M'] * 3
        pts_from_ft = last_5['FTM']

        total_pts = last_5['PTS'].sum()

        if total_pts > 0:
            return {
                'PTS_from_2PT_pct': pts_from_2pt.sum() / total_pts,
                'PTS_from_3PT_pct': pts_from_3pt.sum() / total_pts,
                'PTS_from_FT_pct': pts_from_ft.sum() / total_pts
            }
        else:
            return {
                'PTS_from_2PT_pct': 0,
                'PTS_from_3PT_pct': 0,
                'PTS_from_FT_pct': 0
            }

    def _get_default_ts_features(self) -> Dict[str, float]:
        """Return default values when no history"""
        return {
            'TS_PCT_L3': 0,
            'TS_PCT_L5': 0,
            'TS_PCT_L10': 0,
            'TS_PCT_ewma5': 0,
            'TS_PCT_trend': 0
        }
```

---

## Day 3: Usage x Pace + Integration

### Feature 5: Usage x Pace Interaction

```python
# File: src/features/interactions.py

import pandas as pd
import numpy as np
from typing import Dict

class FeatureInteractions:
    """Calculate interaction features"""

    def calculate_usage_pace_interaction(
        self,
        player_usage: float,
        team_pace: float,
        opp_pace: float
    ) -> Dict[str, float]:
        """
        Calculate Usage x Pace interaction

        High usage players benefit MORE from fast pace
        Example:
        - 30% usage in 95 pace game → ~28.5 PRA
        - 30% usage in 105 pace game → ~31.5 PRA

        Args:
            player_usage: Player's usage rate (0-40)
            team_pace: Team's pace (possessions per 48 min)
            opp_pace: Opponent's pace

        Returns:
            Dictionary with interaction features
        """
        # Average game pace
        avg_pace = (team_pace + opp_pace) / 2

        # Normalize to league average (100)
        pace_factor = avg_pace / 100.0

        # Interaction: Usage * Pace adjustment
        # Higher usage + faster pace = multiplicative effect
        usage_pace_interaction = player_usage * pace_factor

        # Pace differential (team vs opponent)
        # Positive = team plays faster than opponent
        # Negative = team plays slower than opponent
        pace_differential = team_pace - opp_pace

        return {
            'Usage_x_Pace': usage_pace_interaction,
            'Avg_Game_Pace': avg_pace,
            'Pace_Factor': pace_factor,
            'Pace_Differential': pace_differential
        }

    def calculate_minutes_usage_interaction(
        self,
        minutes: float,
        usage: float
    ) -> Dict[str, float]:
        """
        Calculate Minutes x Usage interaction

        More minutes + higher usage = more opportunities
        """
        # Normalize minutes (0-48) to 0-1 scale
        minutes_normalized = minutes / 48.0

        # Normalize usage (0-40) to 0-1 scale
        usage_normalized = usage / 40.0

        # Interaction
        min_usage_interaction = minutes_normalized * usage_normalized * 100

        return {
            'MIN_x_Usage': min_usage_interaction
        }

    def calculate_efficiency_volume_interaction(
        self,
        ts_pct: float,
        usage: float
    ) -> Dict[str, float]:
        """
        Calculate TS% x Usage interaction

        Efficient high-usage players are most valuable
        """
        # Normalize TS% (0-1)
        ts_normalized = ts_pct

        # Normalize usage (0-40)
        usage_normalized = usage / 40.0

        # Interaction: Efficiency * Volume
        efficiency_volume = ts_normalized * usage_normalized * 100

        return {
            'TS_x_Usage': efficiency_volume
        }
```

---

### Integration: Full Phase 1 Pipeline

```python
# File: scripts/validation/walk_forward_validation_phase1.py

import pandas as pd
import numpy as np
from pathlib import Path
import xgboost as xgb
from tqdm import tqdm

# Import Phase 1 features
from src.features.opponent_defense import OpponentDefenseFeatures
from src.features.recent_form import RecentFormFeatures
from src.features.minutes_projection import MinutesProjection
from src.features.scoring_efficiency import ScoringEfficiency
from src.features.interactions import FeatureInteractions

print("="*80)
print("PHASE 1 WALK-FORWARD VALIDATION")
print("="*80)

# Initialize feature engines
opp_defense = OpponentDefenseFeatures()
recent_form = RecentFormFeatures()
minutes_proj = MinutesProjection()
scoring_eff = ScoringEfficiency()
interactions = FeatureInteractions()

# Load data
print("\n1. Loading data...")
train_df = pd.read_parquet('data/processed/train.parquet')
raw_gamelogs = pd.read_csv('data/game_logs/game_logs_2024_25_preprocessed.csv')
raw_gamelogs['GAME_DATE'] = pd.to_datetime(raw_gamelogs['GAME_DATE'])
raw_gamelogs = raw_gamelogs.sort_values('GAME_DATE')

# Define feature columns
base_feature_cols = [col for col in train_df.columns if col not in ['PRA', 'PTS', 'REB', 'AST', 'PLAYER_NAME', 'GAME_DATE']]

# Phase 1 feature columns
phase1_feature_cols = base_feature_cols + [
    # Opponent defense
    'OPP_DRtg', 'OPP_DRtg_Adjusted', 'OPP_DRtg_vs_Avg', 'OPP_Pace',

    # L3 recent form
    'PRA_L3_mean', 'PRA_L3_std', 'PRA_L3_trend', 'MIN_L3_mean',
    'USG_L3_mean', 'PRA_L3_vs_L10_ratio',

    # Minutes projection
    'MIN_projected', 'MIN_ewma5', 'MIN_trend', 'MIN_volatility',

    # True Shooting
    'TS_PCT_L3', 'TS_PCT_L5', 'TS_PCT_ewma5', 'TS_PCT_trend',

    # Interactions
    'Usage_x_Pace', 'Pace_Factor', 'MIN_x_Usage', 'TS_x_Usage'
]

# Train baseline model (for comparison)
print("\n2. Training baseline model...")
X_train_base = train_df[base_feature_cols].fillna(0)
y_train = train_df['PRA']

model_baseline = xgb.XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
model_baseline.fit(X_train_base, y_train)

# Walk-forward validation
print("\n3. Running Phase 1 walk-forward validation...")
unique_dates = sorted(raw_gamelogs['GAME_DATE'].unique())
all_predictions = []

for pred_date in tqdm(unique_dates, desc="Phase 1 walk-forward"):
    games_today = raw_gamelogs[raw_gamelogs['GAME_DATE'] == pred_date]
    past_games = raw_gamelogs[raw_gamelogs['GAME_DATE'] < pred_date]

    for idx, row in games_today.iterrows():
        player_id = row['PLAYER_ID']
        player_name = row['PLAYER_NAME']
        opponent = row.get('OPPONENT', 'UNK')
        player_position = row.get('POSITION', 'Wing')

        # Get player history
        player_history = past_games[past_games['PLAYER_ID'] == player_id]
        player_history = player_history.sort_values('GAME_DATE', ascending=False)

        if len(player_history) < 5:
            continue

        # Calculate Phase 1 features
        # 1. Opponent defense
        opp_def_features = opp_defense.get_opponent_features(opponent, player_position, "2024-25")

        # 2. L3 recent form
        l3_features = recent_form.calculate_l3_features(player_history)

        # 3. Minutes projection
        min_features = minutes_proj.calculate_minutes_features(player_history)

        # 4. True Shooting
        ts_features = scoring_eff.calculate_true_shooting(player_history)

        # 5. Interactions
        player_usage = player_history.iloc[0].get('Usage', 25.0)
        team_pace = 100.0  # TODO: Get from team data
        opp_pace = opp_def_features['OPP_Pace']

        usage_pace_features = interactions.calculate_usage_pace_interaction(
            player_usage, team_pace, opp_pace
        )
        min_usage_features = interactions.calculate_minutes_usage_interaction(
            min_features['MIN_projected'], player_usage
        )
        ts_usage_features = interactions.calculate_efficiency_volume_interaction(
            ts_features['TS_PCT_L5'], player_usage
        )

        # Combine all features
        all_features = {
            **opp_def_features,
            **l3_features,
            **min_features,
            **ts_features,
            **usage_pace_features,
            **min_usage_features,
            **ts_usage_features
        }

        # Create feature vector (baseline features + Phase 1 features)
        feature_vector = []
        for col in phase1_feature_cols:
            feature_vector.append(all_features.get(col, 0))

        # Make prediction (using baseline model for now)
        # TODO: Retrain model with Phase 1 features
        pred_pra_baseline = model_baseline.predict([X_train_base.iloc[0].values])[0]

        # Store prediction
        all_predictions.append({
            'PLAYER_NAME': player_name,
            'GAME_DATE': pred_date,
            'PRA': row['PRA'],
            'predicted_PRA_baseline': pred_pra_baseline,
            # Phase 1 features (for analysis)
            **all_features
        })

# Create predictions DataFrame
predictions_df = pd.DataFrame(all_predictions)

# Calculate metrics
mae_baseline = np.abs(predictions_df['PRA'] - predictions_df['predicted_PRA_baseline']).mean()

print(f"\n4. Results:")
print(f"   Baseline MAE: {mae_baseline:.2f} pts")
print(f"   Total predictions: {len(predictions_df):,}")

# Save
predictions_df.to_csv('data/results/phase1_predictions.csv', index=False)
print(f"\n✅ Saved predictions to data/results/phase1_predictions.csv")
```

---

## Phase 1 Complete Checklist

**Day 1:**
- [ ] Implement `OpponentDefenseFeatures` class
- [ ] Load CTG team data (defense + pace)
- [ ] Test opponent defense features
- [ ] Implement `RecentFormFeatures` class
- [ ] Calculate L3 features
- [ ] Test L3 features
- [ ] Run walk-forward with Day 1 features
- [ ] Compare: Baseline MAE vs Day 1 MAE

**Day 2:**
- [ ] Implement `MinutesProjection` class
- [ ] Calculate EWMA, trend, volatility
- [ ] Test minutes projection
- [ ] Implement `ScoringEfficiency` class
- [ ] Calculate True Shooting %
- [ ] Test TS% features
- [ ] Run walk-forward with Day 1+2 features
- [ ] Compare: Day 1 MAE vs Day 2 MAE

**Day 3:**
- [ ] Implement `FeatureInteractions` class
- [ ] Calculate Usage x Pace
- [ ] Calculate MIN x Usage
- [ ] Calculate TS x Usage
- [ ] Test interaction features
- [ ] Integrate ALL Phase 1 features
- [ ] Run FULL walk-forward validation
- [ ] Run backtest with betting simulation
- [ ] Calculate win rate and ROI
- [ ] **Decision point:** If win rate > 53.5%, proceed to Phase 2

---

**Document End**
