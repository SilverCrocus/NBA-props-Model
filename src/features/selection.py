"""
Feature Selection and Validation Module for NBA Player Props Model
Implements multiple feature selection techniques and validation approaches
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.feature_selection import (
    mutual_info_regression, 
    SelectKBest, 
    f_regression,
    RFECV,
    SelectFromModel
)
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureSelector:
    """
    Comprehensive feature selection for NBA player props prediction
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.feature_scores = {}
        self.selected_features = {}
        
    def univariate_selection(self, X: pd.DataFrame, y: pd.Series, 
                            k: int = 50, score_func=f_regression) -> Dict:
        """
        Univariate feature selection using statistical tests
        """
        selector = SelectKBest(score_func=score_func, k=min(k, X.shape[1]))
        selector.fit(X, y)
        
        # Get scores and selected features
        scores = pd.DataFrame({
            'feature': X.columns,
            'score': selector.scores_
        }).sort_values('score', ascending=False)
        
        selected_features = X.columns[selector.get_support()].tolist()
        
        result = {
            'selected_features': selected_features,
            'scores': scores,
            'selector': selector
        }
        
        self.feature_scores['univariate'] = scores
        self.selected_features['univariate'] = selected_features
        
        logger.info(f"Univariate selection: {len(selected_features)} features selected")
        
        return result
    
    def mutual_information_selection(self, X: pd.DataFrame, y: pd.Series,
                                    threshold: float = 0.01) -> Dict:
        """
        Select features based on mutual information
        """
        # Calculate mutual information
        mi_scores = mutual_info_regression(X, y, random_state=self.random_state)
        
        # Create scores dataframe
        scores = pd.DataFrame({
            'feature': X.columns,
            'mi_score': mi_scores
        }).sort_values('mi_score', ascending=False)
        
        # Select features above threshold
        selected_features = scores[scores['mi_score'] > threshold]['feature'].tolist()
        
        result = {
            'selected_features': selected_features,
            'scores': scores,
            'threshold': threshold
        }
        
        self.feature_scores['mutual_info'] = scores
        self.selected_features['mutual_info'] = selected_features
        
        logger.info(f"Mutual information selection: {len(selected_features)} features selected")
        
        return result
    
    def lasso_selection(self, X: pd.DataFrame, y: pd.Series,
                       alpha: float = 0.01, max_features: int = None) -> Dict:
        """
        Feature selection using LASSO regularization
        """
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Fit LASSO
        lasso = Lasso(alpha=alpha, random_state=self.random_state)
        lasso.fit(X_scaled, y)
        
        # Get feature importance
        importance = np.abs(lasso.coef_)
        scores = pd.DataFrame({
            'feature': X.columns,
            'coefficient': importance
        }).sort_values('coefficient', ascending=False)
        
        # Select non-zero coefficients
        selected_features = scores[scores['coefficient'] > 0]['feature'].tolist()
        
        # Limit features if specified
        if max_features and len(selected_features) > max_features:
            selected_features = selected_features[:max_features]
        
        result = {
            'selected_features': selected_features,
            'scores': scores,
            'model': lasso,
            'alpha': alpha
        }
        
        self.feature_scores['lasso'] = scores
        self.selected_features['lasso'] = selected_features
        
        logger.info(f"LASSO selection: {len(selected_features)} features selected (alpha={alpha})")
        
        return result
    
    def elastic_net_selection(self, X: pd.DataFrame, y: pd.Series,
                            alpha: float = 0.01, l1_ratio: float = 0.5) -> Dict:
        """
        Feature selection using Elastic Net (combination of L1 and L2)
        """
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Fit Elastic Net
        elastic = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=self.random_state)
        elastic.fit(X_scaled, y)
        
        # Get feature importance
        importance = np.abs(elastic.coef_)
        scores = pd.DataFrame({
            'feature': X.columns,
            'coefficient': importance
        }).sort_values('coefficient', ascending=False)
        
        # Select non-zero coefficients
        selected_features = scores[scores['coefficient'] > 0]['feature'].tolist()
        
        result = {
            'selected_features': selected_features,
            'scores': scores,
            'model': elastic,
            'alpha': alpha,
            'l1_ratio': l1_ratio
        }
        
        self.feature_scores['elastic_net'] = scores
        self.selected_features['elastic_net'] = selected_features
        
        logger.info(f"Elastic Net selection: {len(selected_features)} features selected")
        
        return result
    
    def recursive_feature_elimination(self, X: pd.DataFrame, y: pd.Series,
                                     min_features: int = 20,
                                     cv_folds: int = 5) -> Dict:
        """
        Recursive Feature Elimination with Cross-Validation
        """
        # Use Random Forest as the estimator
        estimator = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=self.random_state
        )
        
        # RFECV
        selector = RFECV(
            estimator=estimator,
            min_features_to_select=min_features,
            cv=cv_folds,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        
        selector.fit(X, y)
        
        # Get selected features
        selected_features = X.columns[selector.support_].tolist()
        
        # Create ranking dataframe
        ranking = pd.DataFrame({
            'feature': X.columns,
            'ranking': selector.ranking_
        }).sort_values('ranking')
        
        result = {
            'selected_features': selected_features,
            'ranking': ranking,
            'optimal_features': selector.n_features_,
            'cv_scores': selector.cv_results_,
            'selector': selector
        }
        
        self.selected_features['rfecv'] = selected_features
        
        logger.info(f"RFECV selection: {len(selected_features)} features selected")
        
        return result
    
    def tree_based_selection(self, X: pd.DataFrame, y: pd.Series,
                           model_type: str = 'random_forest',
                           threshold: str = 'median') -> Dict:
        """
        Feature selection using tree-based feature importance
        """
        # Choose model
        if model_type == 'random_forest':
            model = RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                random_state=self.random_state
            )
        elif model_type == 'gradient_boosting':
            model = GradientBoostingRegressor(
                n_estimators=200,
                max_depth=5,
                random_state=self.random_state
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Fit model
        model.fit(X, y)
        
        # Get feature importance
        importance = model.feature_importances_
        scores = pd.DataFrame({
            'feature': X.columns,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        # Select features using threshold
        selector = SelectFromModel(model, threshold=threshold, prefit=True)
        selected_features = X.columns[selector.get_support()].tolist()
        
        result = {
            'selected_features': selected_features,
            'scores': scores,
            'model': model,
            'model_type': model_type,
            'threshold': threshold
        }
        
        self.feature_scores[f'tree_{model_type}'] = scores
        self.selected_features[f'tree_{model_type}'] = selected_features
        
        logger.info(f"Tree-based selection ({model_type}): {len(selected_features)} features selected")
        
        return result
    
    def correlation_filter(self, X: pd.DataFrame, threshold: float = 0.95) -> List[str]:
        """
        Remove highly correlated features
        """
        # Calculate correlation matrix
        corr_matrix = X.corr().abs()
        
        # Select upper triangle
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find features to drop
        to_drop = set()
        for column in upper_tri.columns:
            if column in to_drop:
                continue
            # Find correlated features
            correlated_features = list(upper_tri.index[upper_tri[column] > threshold])
            to_drop.update(correlated_features)
        
        # Keep features
        selected_features = [col for col in X.columns if col not in to_drop]
        
        logger.info(f"Correlation filter: Removed {len(to_drop)} highly correlated features")
        
        return selected_features
    
    def variance_threshold(self, X: pd.DataFrame, threshold: float = 0.01) -> List[str]:
        """
        Remove low variance features
        """
        # Calculate variance
        variances = X.var()
        
        # Select features above threshold
        selected_features = variances[variances > threshold].index.tolist()
        
        removed_features = [col for col in X.columns if col not in selected_features]
        
        logger.info(f"Variance filter: Removed {len(removed_features)} low variance features")
        
        return selected_features
    
    def ensemble_selection(self, X: pd.DataFrame, y: pd.Series,
                          methods: List[str] = None,
                          voting: str = 'majority') -> List[str]:
        """
        Ensemble feature selection combining multiple methods
        """
        if methods is None:
            methods = ['mutual_info', 'lasso', 'tree_random_forest']
        
        # Run each method if not already done
        for method in methods:
            if method not in self.selected_features:
                if method == 'mutual_info':
                    self.mutual_information_selection(X, y)
                elif method == 'lasso':
                    self.lasso_selection(X, y)
                elif method == 'tree_random_forest':
                    self.tree_based_selection(X, y, 'random_forest')
        
        # Combine selections
        all_features = set()
        feature_votes = {}
        
        for method in methods:
            if method in self.selected_features:
                features = self.selected_features[method]
                all_features.update(features)
                
                for feature in features:
                    feature_votes[feature] = feature_votes.get(feature, 0) + 1
        
        if voting == 'majority':
            # Select features that appear in majority of methods
            min_votes = len(methods) // 2 + 1
            selected_features = [f for f, v in feature_votes.items() if v >= min_votes]
        elif voting == 'unanimous':
            # Select features that appear in all methods
            selected_features = [f for f, v in feature_votes.items() if v == len(methods)]
        elif voting == 'any':
            # Select features that appear in any method
            selected_features = list(all_features)
        else:
            raise ValueError(f"Unknown voting method: {voting}")
        
        logger.info(f"Ensemble selection ({voting}): {len(selected_features)} features selected")
        
        return selected_features
    
    def evaluate_feature_set(self, X: pd.DataFrame, y: pd.Series,
                           features: List[str],
                           cv_folds: int = 5) -> Dict:
        """
        Evaluate a feature set using cross-validation
        """
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        # Select features
        X_selected = X[features]
        
        # Initialize model
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=self.random_state
        )
        
        # Cross-validation
        kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        # Calculate different metrics
        mse_scores = -cross_val_score(model, X_selected, y, cv=kfold, 
                                     scoring='neg_mean_squared_error')
        mae_scores = -cross_val_score(model, X_selected, y, cv=kfold,
                                     scoring='neg_mean_absolute_error')
        r2_scores = cross_val_score(model, X_selected, y, cv=kfold,
                                   scoring='r2')
        
        result = {
            'n_features': len(features),
            'mse_mean': np.mean(mse_scores),
            'mse_std': np.std(mse_scores),
            'mae_mean': np.mean(mae_scores),
            'mae_std': np.std(mae_scores),
            'r2_mean': np.mean(r2_scores),
            'r2_std': np.std(r2_scores),
            'features': features
        }
        
        return result
    
    def compare_selection_methods(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Compare different feature selection methods
        """
        comparison_results = []
        
        # Run different selection methods
        methods = {
            'mutual_info': self.mutual_information_selection(X, y)['selected_features'],
            'lasso': self.lasso_selection(X, y)['selected_features'],
            'elastic_net': self.elastic_net_selection(X, y)['selected_features'],
            'tree_rf': self.tree_based_selection(X, y, 'random_forest')['selected_features'],
            'tree_gb': self.tree_based_selection(X, y, 'gradient_boosting')['selected_features']
        }
        
        # Evaluate each method
        for method_name, features in methods.items():
            if features:  # Only evaluate if features were selected
                eval_result = self.evaluate_feature_set(X, y, features[:50])  # Limit to top 50
                eval_result['method'] = method_name
                comparison_results.append(eval_result)
        
        # Create comparison dataframe
        comparison_df = pd.DataFrame(comparison_results)
        comparison_df = comparison_df.sort_values('r2_mean', ascending=False)
        
        return comparison_df
    
    def plot_feature_importance(self, top_n: int = 20, method: str = 'tree_random_forest'):
        """
        Plot feature importance scores
        """
        if method not in self.feature_scores:
            logger.warning(f"Method {method} not found in feature scores")
            return
        
        scores = self.feature_scores[method].head(top_n)
        
        plt.figure(figsize=(10, 8))
        
        if 'score' in scores.columns:
            plt.barh(scores['feature'], scores['score'])
            plt.xlabel('Score')
        elif 'mi_score' in scores.columns:
            plt.barh(scores['feature'], scores['mi_score'])
            plt.xlabel('Mutual Information Score')
        elif 'importance' in scores.columns:
            plt.barh(scores['feature'], scores['importance'])
            plt.xlabel('Feature Importance')
        elif 'coefficient' in scores.columns:
            plt.barh(scores['feature'], scores['coefficient'])
            plt.xlabel('Coefficient (Absolute Value)')
        
        plt.ylabel('Feature')
        plt.title(f'Top {top_n} Features - {method}')
        plt.tight_layout()
        plt.gca().invert_yaxis()
        
        return plt.gcf()
    
    def create_feature_summary(self) -> pd.DataFrame:
        """
        Create summary of selected features across all methods
        """
        all_features = set()
        feature_presence = {}
        
        # Collect all unique features
        for method, features in self.selected_features.items():
            all_features.update(features)
        
        # Check presence in each method
        for feature in all_features:
            feature_presence[feature] = {}
            for method, features in self.selected_features.items():
                feature_presence[feature][method] = 1 if feature in features else 0
        
        # Create dataframe
        summary_df = pd.DataFrame(feature_presence).T
        summary_df['total_selections'] = summary_df.sum(axis=1)
        summary_df = summary_df.sort_values('total_selections', ascending=False)
        
        return summary_df


# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def validate_feature_stability(X: pd.DataFrame, y: pd.Series,
                             selector: FeatureSelector,
                             n_iterations: int = 10,
                             sample_size: float = 0.8) -> pd.DataFrame:
    """
    Validate feature selection stability through bootstrap sampling
    """
    feature_counts = {}
    
    for i in range(n_iterations):
        # Sample data
        sample_idx = np.random.choice(len(X), size=int(len(X) * sample_size), replace=True)
        X_sample = X.iloc[sample_idx]
        y_sample = y.iloc[sample_idx]
        
        # Run feature selection
        selected = selector.lasso_selection(X_sample, y_sample)['selected_features']
        
        # Count features
        for feature in selected:
            feature_counts[feature] = feature_counts.get(feature, 0) + 1
    
    # Create stability report
    stability_df = pd.DataFrame({
        'feature': list(feature_counts.keys()),
        'selection_frequency': list(feature_counts.values())
    })
    stability_df['selection_rate'] = stability_df['selection_frequency'] / n_iterations
    stability_df = stability_df.sort_values('selection_rate', ascending=False)
    
    return stability_df


def cross_validate_features(X: pd.DataFrame, y: pd.Series,
                          features: List[str],
                          n_folds: int = 5) -> Dict:
    """
    Detailed cross-validation of selected features
    """
    from sklearn.model_selection import KFold
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    results = {
        'mse': [],
        'mae': [],
        'r2': [],
        'predictions': [],
        'actuals': []
    }
    
    X_selected = X[features]
    
    for train_idx, test_idx in kfold.split(X_selected):
        X_train, X_test = X_selected.iloc[train_idx], X_selected.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        results['mse'].append(mean_squared_error(y_test, y_pred))
        results['mae'].append(mean_absolute_error(y_test, y_pred))
        results['r2'].append(r2_score(y_test, y_pred))
        results['predictions'].extend(y_pred)
        results['actuals'].extend(y_test)
    
    # Summary statistics
    summary = {
        'mse_mean': np.mean(results['mse']),
        'mse_std': np.std(results['mse']),
        'mae_mean': np.mean(results['mae']),
        'mae_std': np.std(results['mae']),
        'r2_mean': np.mean(results['r2']),
        'r2_std': np.std(results['r2']),
        'raw_results': results
    }
    
    return summary


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

def main():
    """
    Example usage of feature selection module
    """
    # Generate sample data for demonstration
    np.random.seed(42)
    n_samples = 1000
    n_features = 100
    
    # Create sample features
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    # Create target with some correlation to features
    y = (
        X['feature_0'] * 2 + 
        X['feature_1'] * 1.5 + 
        X['feature_2'] * 0.5 +
        np.random.randn(n_samples) * 0.5
    )
    
    # Initialize selector
    selector = FeatureSelector()
    
    print("=== Running Feature Selection Methods ===\n")
    
    # 1. Mutual Information
    mi_result = selector.mutual_information_selection(X, y)
    print(f"Mutual Information: Selected {len(mi_result['selected_features'])} features")
    print(f"Top 5: {mi_result['selected_features'][:5]}\n")
    
    # 2. LASSO
    lasso_result = selector.lasso_selection(X, y, alpha=0.01)
    print(f"LASSO: Selected {len(lasso_result['selected_features'])} features")
    print(f"Top 5: {lasso_result['selected_features'][:5]}\n")
    
    # 3. Tree-based
    tree_result = selector.tree_based_selection(X, y, model_type='random_forest')
    print(f"Random Forest: Selected {len(tree_result['selected_features'])} features")
    print(f"Top 5: {tree_result['selected_features'][:5]}\n")
    
    # 4. Ensemble selection
    ensemble_features = selector.ensemble_selection(X, y, voting='majority')
    print(f"Ensemble (majority): Selected {len(ensemble_features)} features")
    print(f"Features: {ensemble_features[:10]}\n")
    
    # 5. Compare methods
    print("=== Comparing Selection Methods ===")
    comparison = selector.compare_selection_methods(X, y)
    print(comparison[['method', 'n_features', 'r2_mean', 'mae_mean']])
    
    # 6. Feature stability validation
    print("\n=== Feature Stability Analysis ===")
    stability = validate_feature_stability(X, y, selector, n_iterations=10)
    print(stability.head(10))
    
    # 7. Create feature summary
    print("\n=== Feature Selection Summary ===")
    summary = selector.create_feature_summary()
    print(summary.head(10))
    
    # 8. Cross-validate best features
    best_features = ensemble_features[:20]  # Use top 20 ensemble features
    cv_results = cross_validate_features(X, y, best_features)
    
    print("\n=== Cross-Validation Results ===")
    print(f"R² Score: {cv_results['r2_mean']:.3f} ± {cv_results['r2_std']:.3f}")
    print(f"MAE: {cv_results['mae_mean']:.3f} ± {cv_results['mae_std']:.3f}")
    print(f"MSE: {cv_results['mse_mean']:.3f} ± {cv_results['mse_std']:.3f}")


if __name__ == "__main__":
    main()