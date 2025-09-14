"""
Complete Pipeline Integration for NBA Player Props Model
Combines data preprocessing, feature engineering, and feature selection
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple
import pickle
import json
from datetime import datetime

# Import our custom modules
from preprocessing.preprocessor import NBADataPreprocessor
from features.engineering import NBAFeatureEngineer
from features.selection import FeatureSelector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NBAPropsPipeline:
    """
    End-to-end pipeline for NBA player props prediction
    """
    
    def __init__(self, data_path: str = "/Users/diyagamah/Documents/nba_props_model/data"):
        self.data_path = Path(data_path)
        self.preprocessor = NBADataPreprocessor(data_path)
        self.feature_engineer = NBAFeatureEngineer(data_path)
        self.feature_selector = FeatureSelector()
        
        # Pipeline configuration
        self.config = {
            'target': 'PRA',  # Points + Rebounds + Assists
            'features': {
                'tier1_core': [],
                'tier2_contextual': [],
                'tier3_temporal': []
            },
            'selected_features': [],
            'model_metrics': {}
        }
        
    def run_full_pipeline(self, seasons: List[str], 
                         target_season: str = None,
                         save_artifacts: bool = True) -> Dict:
        """
        Run the complete pipeline from raw data to selected features
        
        Args:
            seasons: List of seasons to use for training
            target_season: Season to use for testing (if None, uses train/test split)
            save_artifacts: Whether to save intermediate results
        
        Returns:
            Dictionary containing processed data and results
        """
        logger.info("Starting NBA Props Pipeline...")
        results = {}
        
        # ========== Step 1: Data Loading and Preprocessing ==========
        logger.info("Step 1: Loading and preprocessing data...")
        
        all_data = []
        for season in seasons:
            season_data = self.preprocessor.prepare_modeling_data(season)
            season_data['season'] = season
            all_data.append(season_data)
        
        combined_data = pd.concat(all_data, ignore_index=True)
        logger.info(f"Loaded {len(combined_data)} samples from {len(seasons)} seasons")
        
        # Handle multi-team players
        combined_data = self.preprocessor.handle_multi_team_players(combined_data)
        
        # Data quality check
        quality_report = self.preprocessor.validate_data_quality(combined_data)
        results['data_quality'] = quality_report
        
        # ========== Step 2: Feature Engineering ==========
        logger.info("Step 2: Engineering features...")
        
        # Create features using the three-tier architecture
        engineered_features = self.feature_engineer.create_feature_set(combined_data)
        
        # Handle missing values
        engineered_features = self.feature_engineer.handle_missing_values(
            engineered_features, strategy='smart'
        )
        
        # Remove outliers for key features
        key_features = ['USG_L15_EWMA', 'PSA_L15_EWMA', 'Minutes_L5_Mean']
        available_features = [f for f in key_features if f in engineered_features.columns]
        engineered_features = self.feature_engineer.remove_outliers(
            engineered_features, available_features, method='iqr'
        )
        
        # Validate features
        feature_validation = self.feature_engineer.validate_features(engineered_features)
        results['feature_validation'] = feature_validation
        
        logger.info(f"Created {len(engineered_features.columns)} features")
        
        # ========== Step 3: Create Target Variable ==========
        logger.info("Step 3: Creating target variable...")
        
        # For now, create estimated PRA from available stats
        if 'PRA' not in engineered_features.columns:
            if all(col in combined_data.columns for col in ['Usage', 'PSA', 'AST%']):
                # Create estimated PRA
                engineered_features['PRA_estimate'] = (
                    combined_data['Usage'] * 1.5 +
                    combined_data['PSA'] * 0.3 +
                    combined_data.get('AST%', 0) * 0.5 +
                    combined_data.get('fgDR%', 0) * 0.4 +
                    combined_data.get('fgOR%', 0) * 0.3
                )
                target_col = 'PRA_estimate'
            else:
                logger.error("Cannot create target variable from available data")
                return results
        else:
            target_col = 'PRA'
        
        # ========== Step 4: Train/Test Split ==========
        logger.info("Step 4: Creating train/test split...")
        
        # Prepare features and target
        feature_cols = [col for col in engineered_features.columns 
                       if col not in ['Player', 'Team', 'Pos', 'Age', 'season', 
                                     target_col, 'PRA_estimate', 'PRA']]
        
        X = engineered_features[feature_cols]
        y = engineered_features[target_col]
        
        # Remove any remaining non-numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X = X[numeric_cols]
        
        if target_season:
            # Use specific season for testing
            train_mask = combined_data['season'] != target_season
            test_mask = combined_data['season'] == target_season
            
            X_train = X[train_mask]
            y_train = y[train_mask]
            X_test = X[test_mask]
            y_test = y[test_mask]
        else:
            # Random split
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        
        logger.info(f"Train set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        
        # ========== Step 5: Feature Selection ==========
        logger.info("Step 5: Selecting features...")
        
        # Apply correlation filter first
        low_corr_features = self.feature_selector.correlation_filter(X_train, threshold=0.95)
        X_train_filtered = X_train[low_corr_features]
        X_test_filtered = X_test[low_corr_features]
        
        # Apply variance threshold
        high_var_features = self.feature_selector.variance_threshold(X_train_filtered, threshold=0.01)
        X_train_filtered = X_train_filtered[high_var_features]
        X_test_filtered = X_test_filtered[high_var_features]
        
        logger.info(f"After filtering: {len(X_train_filtered.columns)} features remain")
        
        # Run multiple selection methods
        selection_results = {}
        
        # Mutual Information
        mi_result = self.feature_selector.mutual_information_selection(
            X_train_filtered, y_train, threshold=0.01
        )
        selection_results['mutual_info'] = mi_result
        
        # LASSO
        lasso_result = self.feature_selector.lasso_selection(
            X_train_filtered, y_train, alpha=0.01
        )
        selection_results['lasso'] = lasso_result
        
        # Tree-based
        tree_result = self.feature_selector.tree_based_selection(
            X_train_filtered, y_train, model_type='random_forest'
        )
        selection_results['tree'] = tree_result
        
        # Ensemble selection
        ensemble_features = self.feature_selector.ensemble_selection(
            X_train_filtered, y_train, voting='majority'
        )
        
        # Limit to top features
        max_features = min(50, len(ensemble_features))
        final_features = ensemble_features[:max_features]
        
        logger.info(f"Final feature set: {len(final_features)} features")
        
        # ========== Step 6: Evaluate Final Feature Set ==========
        logger.info("Step 6: Evaluating final feature set...")
        
        evaluation = self.feature_selector.evaluate_feature_set(
            X_train_filtered, y_train, final_features, cv_folds=5
        )
        
        results['feature_evaluation'] = evaluation
        results['selected_features'] = final_features
        
        # ========== Step 7: Categorize Features by Tier ==========
        self.categorize_features(final_features)
        
        # ========== Step 8: Save Results ==========
        if save_artifacts:
            self.save_pipeline_artifacts(
                engineered_features,
                final_features,
                results
            )
        
        # Compile final results
        results['data'] = {
            'X_train': X_train_filtered[final_features],
            'X_test': X_test_filtered[final_features],
            'y_train': y_train,
            'y_test': y_test,
            'full_features': engineered_features,
            'feature_columns': final_features
        }
        
        logger.info("Pipeline complete!")
        
        return results
    
    def categorize_features(self, features: List[str]):
        """
        Categorize selected features by tier
        """
        tier1_keywords = ['USG', 'PSA', 'AST', 'PER', 'efficiency', 'OR%', 'DR%']
        tier2_keywords = ['Minutes', 'Rest', 'B2B', 'opp_', 'Pos_vs']
        tier3_keywords = ['L5', 'L10', 'L15', 'ewma', 'volatility', 'trend']
        
        for feature in features:
            categorized = False
            
            # Check Tier 3 first (temporal features often contain tier 1 keywords)
            for keyword in tier3_keywords:
                if keyword in feature:
                    self.config['features']['tier3_temporal'].append(feature)
                    categorized = True
                    break
            
            if not categorized:
                # Check Tier 2
                for keyword in tier2_keywords:
                    if keyword in feature:
                        self.config['features']['tier2_contextual'].append(feature)
                        categorized = True
                        break
            
            if not categorized:
                # Check Tier 1
                for keyword in tier1_keywords:
                    if keyword in feature:
                        self.config['features']['tier1_core'].append(feature)
                        categorized = True
                        break
            
            # Default to Tier 1 if uncategorized
            if not categorized:
                self.config['features']['tier1_core'].append(feature)
        
        logger.info(f"Tier 1 (Core): {len(self.config['features']['tier1_core'])} features")
        logger.info(f"Tier 2 (Contextual): {len(self.config['features']['tier2_contextual'])} features")
        logger.info(f"Tier 3 (Temporal): {len(self.config['features']['tier3_temporal'])} features")
    
    def save_pipeline_artifacts(self, features_df: pd.DataFrame,
                               selected_features: List[str],
                               results: Dict):
        """
        Save pipeline artifacts for later use
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save engineered features
        features_path = self.data_path.parent / f"engineered_features_{timestamp}.csv"
        features_df.to_csv(features_path, index=False)
        logger.info(f"Saved engineered features to {features_path}")
        
        # Save selected features list
        features_list_path = self.data_path.parent / f"selected_features_{timestamp}.json"
        with open(features_list_path, 'w') as f:
            json.dump({
                'features': selected_features,
                'tiers': self.config['features'],
                'timestamp': timestamp
            }, f, indent=2)
        logger.info(f"Saved selected features to {features_list_path}")
        
        # Save pipeline configuration
        config_path = self.data_path.parent / f"pipeline_config_{timestamp}.json"
        
        # Convert numpy types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Clean results for JSON
        json_results = {}
        for key, value in results.items():
            if key != 'data':  # Skip data frames
                if isinstance(value, dict):
                    json_results[key] = {k: convert_numpy(v) for k, v in value.items()}
                else:
                    json_results[key] = convert_numpy(value)
        
        with open(config_path, 'w') as f:
            json.dump({
                'config': self.config,
                'results': json_results,
                'timestamp': timestamp
            }, f, indent=2, default=str)
        logger.info(f"Saved pipeline configuration to {config_path}")
    
    def generate_feature_report(self, results: Dict) -> str:
        """
        Generate a comprehensive feature engineering report
        """
        report = []
        report.append("=" * 80)
        report.append("NBA PLAYER PROPS FEATURE ENGINEERING REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Data Quality
        if 'data_quality' in results:
            dq = results['data_quality']
            report.append("DATA QUALITY SUMMARY")
            report.append("-" * 40)
            report.append(f"Total Samples: {dq['total_rows']}")
            report.append(f"Total Features: {dq['total_columns']}")
            report.append(f"Unique Players: {dq['unique_players']}")
            report.append(f"Unique Teams: {dq['unique_teams']}")
            
            if dq['potential_issues']:
                report.append("\nPotential Issues:")
                for issue in dq['potential_issues']:
                    report.append(f"  - {issue}")
            report.append("")
        
        # Feature Validation
        if 'feature_validation' in results:
            fv = results['feature_validation']
            report.append("FEATURE VALIDATION")
            report.append("-" * 40)
            report.append(f"Features Created: {fv['n_features']}")
            report.append(f"Zero Variance Features: {len(fv['zero_variance'])}")
            report.append(f"High Correlation Pairs: {len(fv['high_correlation_pairs'])}")
            
            if fv['outlier_features']:
                report.append("\nFeatures with Outliers:")
                for feat, pct in fv['outlier_features'][:5]:
                    report.append(f"  - {feat}: {pct}% outliers")
            report.append("")
        
        # Selected Features
        if 'selected_features' in results:
            sf = results['selected_features']
            report.append("SELECTED FEATURES")
            report.append("-" * 40)
            report.append(f"Total Selected: {len(sf)}")
            report.append("")
            
            # By Tier
            report.append("Features by Tier:")
            for tier_name, tier_features in self.config['features'].items():
                if tier_features:
                    report.append(f"\n{tier_name.upper()} ({len(tier_features)} features):")
                    for feat in tier_features[:5]:  # Show first 5
                        report.append(f"  - {feat}")
                    if len(tier_features) > 5:
                        report.append(f"  ... and {len(tier_features) - 5} more")
            report.append("")
        
        # Model Evaluation
        if 'feature_evaluation' in results:
            fe = results['feature_evaluation']
            report.append("CROSS-VALIDATION RESULTS")
            report.append("-" * 40)
            report.append(f"Number of Features: {fe['n_features']}")
            report.append(f"R² Score: {fe['r2_mean']:.4f} ± {fe['r2_std']:.4f}")
            report.append(f"MAE: {fe['mae_mean']:.4f} ± {fe['mae_std']:.4f}")
            report.append(f"MSE: {fe['mse_mean']:.4f} ± {fe['mse_std']:.4f}")
            report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

def main():
    """
    Example usage of the integrated pipeline
    """
    # Initialize pipeline
    pipeline = NBAPropsPipeline()
    
    # Define seasons to use
    training_seasons = ['2021-22', '2022-23', '2023-24']
    
    print("=" * 80)
    print("NBA PLAYER PROPS PREDICTION - FEATURE ENGINEERING PIPELINE")
    print("=" * 80)
    print()
    print(f"Training Seasons: {', '.join(training_seasons)}")
    print()
    
    # Run full pipeline
    results = pipeline.run_full_pipeline(
        seasons=training_seasons,
        target_season=None,  # Use random split instead of specific season
        save_artifacts=True
    )
    
    # Generate and print report
    report = pipeline.generate_feature_report(results)
    print(report)
    
    # Save report
    report_path = Path("/Users/diyagamah/Documents/nba_props_model/feature_engineering_report.txt")
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nReport saved to: {report_path}")
    
    # Display feature importance summary
    if 'selected_features' in results:
        print("\n" + "=" * 80)
        print("TOP 20 SELECTED FEATURES")
        print("=" * 80)
        for i, feature in enumerate(results['selected_features'][:20], 1):
            print(f"{i:2d}. {feature}")
    
    # Save final dataset for modeling
    if 'data' in results:
        X_train = results['data']['X_train']
        X_test = results['data']['X_test']
        y_train = results['data']['y_train']
        y_test = results['data']['y_test']
        
        # Save as pickle for easy loading
        import pickle
        
        model_data = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': results['selected_features'],
            'feature_tiers': pipeline.config['features']
        }
        
        pickle_path = Path("/Users/diyagamah/Documents/nba_props_model/model_data.pkl")
        with open(pickle_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"\nModel data saved to: {pickle_path}")
        print(f"  - Training samples: {len(X_train)}")
        print(f"  - Test samples: {len(X_test)}")
        print(f"  - Features: {len(X_train.columns)}")


if __name__ == "__main__":
    main()