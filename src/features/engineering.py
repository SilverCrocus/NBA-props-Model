"""
NBA Player Props (PRA) Feature Engineering Pipeline
Three-Tier Architecture for Points + Rebounds + Assists Prediction
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
from datetime import datetime, timedelta
from pathlib import Path
import logging

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NBAFeatureEngineer:
    """
    Comprehensive feature engineering pipeline for NBA player props prediction.
    Implements three-tier architecture:
    1. Core Performance Engine (Player's Baseline)
    2. Contextual Modulators (Game-Specific)
    3. Temporal Dynamics (Recent Form)
    """
    
    def __init__(self, data_path: str = "/Users/diyagamah/Documents/nba_props_model/data"):
        self.data_path = Path(data_path)
        self.ctg_player_path = self.data_path / "ctg_data_organized" / "players"
        self.ctg_team_path = self.data_path / "ctg_team_data"
        
        # Feature configuration
        self.rolling_windows = [5, 10, 15]
        self.ewma_spans = [5, 10, 15]
        self.min_games_threshold = 5
        
    # ============================================================================
    # TIER 1: CORE PERFORMANCE ENGINE
    # ============================================================================
    
    def calculate_usage_rate_ewma(self, df: pd.DataFrame, span: int = 15) -> pd.Series:
        """
        Calculate Exponentially Weighted Moving Average of Usage Rate
        Formula: USG% = ((FGA + 0.44 * FTA + TOV) * (Tm MP / 5)) / (MP * (Tm FGA + 0.44 * Tm FTA + Tm TOV))
        """
        if 'Usage' in df.columns:
            # Direct usage from CTG data
            return df.groupby('Player')['Usage'].transform(
                lambda x: x.ewm(span=span, min_periods=1).mean()
            )
        else:
            # Calculate from raw stats if needed
            logger.warning("Usage not found in data, calculating from raw stats")
            return None
    
    def calculate_scoring_efficiency(self, df: pd.DataFrame, span: int = 15) -> pd.Series:
        """
        Calculate Points per 100 Shot Attempts (PSA) with EWMA
        PSA = Points / (FGA + 0.44 * FTA) * 100
        """
        if 'PSA' in df.columns:
            return df.groupby('Player')['PSA'].transform(
                lambda x: x.ewm(span=span, min_periods=1).mean()
            )
        else:
            logger.warning("PSA not found in data")
            return None
    
    def calculate_playmaking_ratio(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate AST% to USG% ratio (season-long)
        Indicates player's playmaking efficiency relative to their usage
        """
        if 'AST%' in df.columns and 'Usage' in df.columns:
            # Avoid division by zero
            usage_safe = df['Usage'].replace(0, np.nan)
            return (df['AST%'] / usage_safe).fillna(0)
        return None
    
    def calculate_rebounding_metrics(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Calculate rebounding percentages (offensive and defensive)
        """
        rebound_features = {}
        
        if 'fgOR%' in df.columns:
            rebound_features['offensive_rebound_pct'] = df['fgOR%']
        
        if 'fgDR%' in df.columns:
            rebound_features['defensive_rebound_pct'] = df['fgDR%']
            
        if 'ftOR%' in df.columns:
            rebound_features['ft_offensive_rebound_pct'] = df['ftOR%']
            
        if 'ftDR%' in df.columns:
            rebound_features['ft_defensive_rebound_pct'] = df['ftDR%']
            
        return rebound_features
    
    def calculate_advanced_metrics(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Calculate advanced metrics like PER approximation, efficiency ratings
        """
        advanced_features = {}
        
        # Approximate PER using available CTG stats
        if all(col in df.columns for col in ['Usage', 'PSA', 'AST%', 'TOV%']):
            # Simplified PER approximation
            advanced_features['per_approximation'] = (
                df['Usage'] * 0.3 +
                df['PSA'] / 100 * 0.3 +
                df['AST%'] * 0.2 -
                df['TOV%'] * 0.2
            )
        
        # Offensive efficiency composite
        if 'PSA' in df.columns and 'AST%' in df.columns and 'TOV%' in df.columns:
            advanced_features['offensive_efficiency'] = (
                df['PSA'] / 100 * 0.5 +
                df['AST%'] * 0.3 -
                df['TOV%'] * 0.2
            )
        
        return advanced_features
    
    # ============================================================================
    # TIER 2: CONTEXTUAL MODULATORS
    # ============================================================================
    
    def calculate_minutes_average(self, df: pd.DataFrame, games: int = 5) -> pd.Series:
        """
        Calculate rolling average of minutes played
        """
        if 'MIN' in df.columns:
            return df.groupby('Player')['MIN'].transform(
                lambda x: x.rolling(window=games, min_periods=1).mean()
            )
        return None
    
    def calculate_rest_days(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate days of rest between games
        """
        if 'game_date' in df.columns:
            df['game_date'] = pd.to_datetime(df['game_date'])
            df['days_rest'] = df.groupby('Player')['game_date'].diff().dt.days.fillna(0)
            return df['days_rest']
        return pd.Series(1, index=df.index)  # Default to 1 day rest
    
    def identify_back_to_back(self, df: pd.DataFrame) -> pd.Series:
        """
        Identify back-to-back games (0 or 1 day rest)
        """
        if 'days_rest' not in df.columns:
            df['days_rest'] = self.calculate_rest_days(df)
        
        return (df['days_rest'] <= 1).astype(int)
    
    def calculate_usage_delta_teammates_out(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate usage rate change when key teammates are out
        Using on/off data if available
        """
        # This would require on/off data analysis
        # Placeholder for now - would need game-by-game teammate availability
        return pd.Series(0, index=df.index)
    
    def get_opponent_metrics(self, df: pd.DataFrame, team_data: pd.DataFrame) -> pd.DataFrame:
        """
        Get opponent-specific metrics (pace, defensive rating, etc.)
        """
        opponent_features = pd.DataFrame(index=df.index)
        
        if 'opponent' in df.columns and not team_data.empty:
            # Merge opponent team stats
            opponent_features = df.merge(
                team_data[['Team', 'Pace', 'DefRtg', 'ORB%_allowed']],
                left_on='opponent',
                right_on='Team',
                how='left'
            )
            
            opponent_features = opponent_features.rename(columns={
                'Pace': 'opp_pace',
                'DefRtg': 'opp_def_rating',
                'ORB%_allowed': 'opp_orb_allowed'
            })
        
        return opponent_features
    
    def calculate_position_vs_defense(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate how opponent defends against player's position
        """
        # This would require position-specific defensive data
        # Placeholder implementation
        position_defense_modifier = {
            'Guard': 1.0,
            'Wing': 1.0,
            'Big': 1.0
        }
        
        if 'Pos' in df.columns:
            return df['Pos'].map(position_defense_modifier).fillna(1.0)
        return pd.Series(1.0, index=df.index)
    
    # ============================================================================
    # TIER 3: TEMPORAL DYNAMICS
    # ============================================================================
    
    def calculate_rolling_averages(self, df: pd.DataFrame, 
                                  stats: List[str],
                                  windows: List[int] = None) -> pd.DataFrame:
        """
        Calculate rolling averages for specified statistics
        """
        if windows is None:
            windows = self.rolling_windows
        
        rolling_features = pd.DataFrame(index=df.index)
        
        for stat in stats:
            if stat in df.columns:
                for window in windows:
                    col_name = f"{stat}_L{window}_mean"
                    rolling_features[col_name] = df.groupby('Player')[stat].transform(
                        lambda x: x.rolling(window=window, min_periods=1).mean()
                    )
        
        return rolling_features
    
    def calculate_ewma_features(self, df: pd.DataFrame,
                               stats: List[str],
                               spans: List[int] = None) -> pd.DataFrame:
        """
        Calculate exponentially weighted moving averages
        """
        if spans is None:
            spans = self.ewma_spans
        
        ewma_features = pd.DataFrame(index=df.index)
        
        for stat in stats:
            if stat in df.columns:
                for span in spans:
                    col_name = f"{stat}_ewma_{span}"
                    ewma_features[col_name] = df.groupby('Player')[stat].transform(
                        lambda x: x.ewm(span=span, min_periods=1).mean()
                    )
        
        return ewma_features
    
    def calculate_volatility(self, df: pd.DataFrame,
                           stats: List[str],
                           window: int = 15) -> pd.DataFrame:
        """
        Calculate rolling standard deviation (volatility) of key stats
        """
        volatility_features = pd.DataFrame(index=df.index)
        
        for stat in stats:
            if stat in df.columns:
                col_name = f"{stat}_volatility_L{window}"
                volatility_features[col_name] = df.groupby('Player')[stat].transform(
                    lambda x: x.rolling(window=window, min_periods=2).std()
                ).fillna(0)
        
        return volatility_features
    
    def calculate_trend_features(self, df: pd.DataFrame,
                                stats: List[str],
                                window: int = 10) -> pd.DataFrame:
        """
        Calculate trend features (linear regression slope over window)
        """
        from scipy import stats as scipy_stats
        
        trend_features = pd.DataFrame(index=df.index)
        
        def calculate_trend(series):
            if len(series) < 2:
                return 0
            x = np.arange(len(series))
            try:
                slope, _, _, _, _ = scipy_stats.linregress(x, series)
                return slope
            except:
                return 0
        
        for stat in stats:
            if stat in df.columns:
                col_name = f"{stat}_trend_L{window}"
                trend_features[col_name] = df.groupby('Player')[stat].transform(
                    lambda x: x.rolling(window=window, min_periods=2).apply(calculate_trend)
                ).fillna(0)
        
        return trend_features
    
    # ============================================================================
    # DATA PREPROCESSING
    # ============================================================================
    
    def handle_missing_values(self, df: pd.DataFrame,
                            strategy: str = 'smart') -> pd.DataFrame:
        """
        Handle missing values with different strategies
        """
        df = df.copy()
        
        if strategy == 'smart':
            # Forward fill for time series data within player groups
            time_series_cols = [col for col in df.columns if 'ewma' in col or 'L' in col]
            if time_series_cols:
                df[time_series_cols] = df.groupby('Player')[time_series_cols].ffill()
            
            # Fill percentage columns with 0
            pct_cols = [col for col in df.columns if '%' in col or 'pct' in col]
            df[pct_cols] = df[pct_cols].fillna(0)
            
            # Fill rate/ratio columns with median
            rate_cols = [col for col in df.columns if 'rate' in col.lower() or 'ratio' in col.lower()]
            for col in rate_cols:
                df[col] = df[col].fillna(df[col].median())
            
            # Fill remaining numeric columns with 0
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(0)
            
        elif strategy == 'drop':
            df = df.dropna()
            
        elif strategy == 'zero':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(0)
        
        return df
    
    def remove_outliers(self, df: pd.DataFrame,
                       cols: List[str],
                       method: str = 'iqr',
                       threshold: float = 3.0) -> pd.DataFrame:
        """
        Remove outliers using IQR or Z-score method
        """
        df = df.copy()
        
        for col in cols:
            if col not in df.columns:
                continue
                
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                df[col] = df[col].clip(lower_bound, upper_bound)
                
            elif method == 'zscore':
                mean = df[col].mean()
                std = df[col].std()
                lower_bound = mean - threshold * std
                upper_bound = mean + threshold * std
                df[col] = df[col].clip(lower_bound, upper_bound)
        
        return df
    
    def normalize_features(self, df: pd.DataFrame,
                         method: str = 'standard',
                         exclude_cols: List[str] = None) -> pd.DataFrame:
        """
        Normalize features using various methods
        """
        from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
        
        df = df.copy()
        
        if exclude_cols is None:
            exclude_cols = ['Player', 'Team', 'Pos', 'game_date', 'opponent']
        
        numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                       if col not in exclude_cols]
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            return df
        
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        
        return df
    
    # ============================================================================
    # FEATURE SELECTION
    # ============================================================================
    
    def select_features_mutual_info(self, X: pd.DataFrame, y: pd.Series,
                                   n_features: int = 50) -> List[str]:
        """
        Select top features using mutual information
        """
        from sklearn.feature_selection import mutual_info_regression
        
        # Calculate mutual information scores
        mi_scores = mutual_info_regression(X, y, random_state=42)
        
        # Create feature importance dataframe
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'mi_score': mi_scores
        }).sort_values('mi_score', ascending=False)
        
        return feature_importance.head(n_features)['feature'].tolist()
    
    def select_features_lasso(self, X: pd.DataFrame, y: pd.Series,
                            alpha: float = 0.01) -> List[str]:
        """
        Select features using LASSO regularization
        """
        from sklearn.linear_model import Lasso
        
        lasso = Lasso(alpha=alpha, random_state=42)
        lasso.fit(X, y)
        
        # Get non-zero coefficients
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'coefficient': np.abs(lasso.coef_)
        })
        
        selected_features = feature_importance[
            feature_importance['coefficient'] > 0
        ]['feature'].tolist()
        
        return selected_features
    
    def select_features_recursive(self, X: pd.DataFrame, y: pd.Series,
                                 n_features: int = 50) -> List[str]:
        """
        Recursive Feature Elimination with Cross-Validation
        """
        from sklearn.feature_selection import RFECV
        from sklearn.ensemble import RandomForestRegressor
        
        estimator = RandomForestRegressor(n_estimators=100, random_state=42)
        selector = RFECV(estimator, min_features_to_select=n_features, cv=5)
        selector.fit(X, y)
        
        selected_features = X.columns[selector.support_].tolist()
        
        return selected_features
    
    def correlation_filter(self, df: pd.DataFrame,
                         threshold: float = 0.95) -> List[str]:
        """
        Remove highly correlated features
        """
        corr_matrix = df.corr().abs()
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find features to drop
        to_drop = [column for column in upper_tri.columns 
                  if any(upper_tri[column] > threshold)]
        
        return [col for col in df.columns if col not in to_drop]
    
    # ============================================================================
    # MAIN PIPELINE
    # ============================================================================
    
    def create_feature_set(self, player_data: pd.DataFrame,
                          team_data: pd.DataFrame = None) -> pd.DataFrame:
        """
        Main method to create complete feature set
        """
        logger.info("Starting feature engineering pipeline...")
        
        features = pd.DataFrame(index=player_data.index)
        
        # Copy identifier columns
        id_cols = ['Player', 'Team', 'Pos', 'Age', 'MIN']
        for col in id_cols:
            if col in player_data.columns:
                features[col] = player_data[col]
        
        # ========== TIER 1: Core Performance Engine ==========
        logger.info("Building Tier 1: Core Performance Engine features...")
        
        # Usage Rate EWMA
        usage_ewma = self.calculate_usage_rate_ewma(player_data, span=15)
        if usage_ewma is not None:
            features['USG_L15_EWMA'] = usage_ewma
        
        # Scoring Efficiency
        psa_ewma = self.calculate_scoring_efficiency(player_data, span=15)
        if psa_ewma is not None:
            features['PSA_L15_EWMA'] = psa_ewma
        
        # Playmaking Ratio
        playmaking = self.calculate_playmaking_ratio(player_data)
        if playmaking is not None:
            features['AST_to_USG_Ratio_S'] = playmaking
        
        # Rebounding Metrics
        rebounding = self.calculate_rebounding_metrics(player_data)
        for key, value in rebounding.items():
            features[key] = value
        
        # Advanced Metrics
        advanced = self.calculate_advanced_metrics(player_data)
        for key, value in advanced.items():
            features[key] = value
        
        # ========== TIER 2: Contextual Modulators ==========
        logger.info("Building Tier 2: Contextual Modulators...")
        
        # Minutes Average
        min_avg = self.calculate_minutes_average(player_data, games=5)
        if min_avg is not None:
            features['Minutes_L5_Mean'] = min_avg
        
        # Rest Days and B2B
        features['Days_Rest'] = self.calculate_rest_days(player_data)
        features['Is_B2B'] = self.identify_back_to_back(player_data)
        
        # Opponent Metrics (if team data available)
        if team_data is not None:
            opp_features = self.get_opponent_metrics(player_data, team_data)
            features = pd.concat([features, opp_features], axis=1)
        else:
            # Use new opponent features if available
            try:
                from features.opponent_features import OpponentFeatures
                opp_feat_engine = OpponentFeatures(str(self.data_path))
                if 'Opponent' in player_data.columns and 'Season' in player_data.columns:
                    player_with_opp = opp_feat_engine.get_opponent_features(player_data)
                    matchup_feats = opp_feat_engine.create_matchup_features(player_data, player_with_opp)
                    features = pd.concat([features, matchup_feats], axis=1)
                    logger.info("Added opponent-adjusted features")
            except Exception as e:
                logger.warning(f"Could not add opponent features: {e}")
        
        # Position vs Defense
        features['Pos_vs_Def'] = self.calculate_position_vs_defense(player_data)
        
        # ========== TIER 3: Temporal Dynamics ==========
        logger.info("Building Tier 3: Temporal Dynamics features...")
        
        # Key stats for temporal features
        temporal_stats = ['Usage', 'PSA', 'AST%', 'TOV%', 'fgOR%', 'fgDR%']
        available_stats = [s for s in temporal_stats if s in player_data.columns]
        
        # Rolling Averages
        rolling_features = self.calculate_rolling_averages(
            player_data, available_stats, windows=[5, 10, 15]
        )
        features = pd.concat([features, rolling_features], axis=1)
        
        # EWMA Features
        ewma_features = self.calculate_ewma_features(
            player_data, available_stats, spans=[5, 10, 15]
        )
        features = pd.concat([features, ewma_features], axis=1)
        
        # Volatility Features
        volatility_features = self.calculate_volatility(
            player_data, available_stats[:3], window=15  # Focus on offensive stats
        )
        features = pd.concat([features, volatility_features], axis=1)
        
        # Trend Features
        trend_features = self.calculate_trend_features(
            player_data, available_stats[:3], window=10
        )
        features = pd.concat([features, trend_features], axis=1)
        
        logger.info(f"Feature engineering complete. Created {len(features.columns)} features.")
        
        return features
    
    def validate_features(self, features: pd.DataFrame) -> Dict[str, any]:
        """
        Validate feature quality and provide diagnostics
        """
        validation_report = {
            'n_samples': len(features),
            'n_features': len(features.columns),
            'missing_pct': {},
            'zero_variance': [],
            'high_correlation_pairs': [],
            'outlier_features': []
        }
        
        # Check missing values
        for col in features.columns:
            missing_pct = features[col].isna().mean() * 100
            if missing_pct > 0:
                validation_report['missing_pct'][col] = round(missing_pct, 2)
        
        # Check zero variance features
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if features[col].std() < 1e-7:
                validation_report['zero_variance'].append(col)
        
        # Check high correlations
        corr_matrix = features[numeric_cols].corr().abs()
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        high_corr = np.where(upper_tri > 0.95)
        for i, j in zip(high_corr[0], high_corr[1]):
            pair = (numeric_cols[i], numeric_cols[j], round(upper_tri.iloc[i, j], 3))
            validation_report['high_correlation_pairs'].append(pair)
        
        # Check for potential outliers (values > 3 std from mean)
        for col in numeric_cols:
            z_scores = np.abs((features[col] - features[col].mean()) / features[col].std())
            if (z_scores > 3).any():
                outlier_pct = (z_scores > 3).mean() * 100
                validation_report['outlier_features'].append((col, round(outlier_pct, 2)))
        
        return validation_report


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

def main():
    """
    Example usage of the feature engineering pipeline
    """
    # Initialize feature engineer
    engineer = NBAFeatureEngineer()
    
    # Load sample data (you would load your actual data here)
    # Example: Load offensive overview data
    sample_path = "/Users/diyagamah/Documents/nba_props_model/data/ctg_data_organized/players/2023-24/regular_season/offensive_overview/offensive_overview.csv"
    
    try:
        player_data = pd.read_csv(sample_path)
        
        # Create features
        features = engineer.create_feature_set(player_data)
        
        # Handle missing values
        features = engineer.handle_missing_values(features, strategy='smart')
        
        # Validate features
        validation_report = engineer.validate_features(features)
        
        print("\n=== Feature Engineering Report ===")
        print(f"Total features created: {validation_report['n_features']}")
        print(f"Total samples: {validation_report['n_samples']}")
        
        if validation_report['missing_pct']:
            print("\nFeatures with missing values:")
            for feat, pct in validation_report['missing_pct'].items():
                print(f"  - {feat}: {pct}%")
        
        if validation_report['zero_variance']:
            print(f"\nZero variance features: {validation_report['zero_variance']}")
        
        if validation_report['high_correlation_pairs']:
            print("\nHighly correlated feature pairs:")
            for pair in validation_report['high_correlation_pairs'][:5]:
                print(f"  - {pair[0]} & {pair[1]}: {pair[2]}")
        
        # Save features
        output_path = "/Users/diyagamah/Documents/nba_props_model/engineered_features.csv"
        features.to_csv(output_path, index=False)
        print(f"\nFeatures saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Error in feature engineering pipeline: {e}")
        raise


if __name__ == "__main__":
    main()