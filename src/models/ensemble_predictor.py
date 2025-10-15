"""
Tree Ensemble Predictor - Stacking XGBoost + LightGBM + CatBoost

This module implements a two-level stacking ensemble for NBA props prediction:
- Level 0: XGBoost, LightGBM, CatBoost (base models)
- Level 1: Ridge Regression (meta-learner)

Expected improvement: 5-10% MAE reduction through model diversity

Author: NBA Props Model - Phase 2 Week 2
Date: October 15, 2025
"""

import logging
from typing import Dict, List, Optional, Tuple

import catboost as cat
import lightgbm as lgb
import numpy as np
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TreeEnsemblePredictor:
    """
    Stacking ensemble of tree-based models for PRA prediction.

    Architecture:
        Level 0 (Base Models):
            - XGBoost: Level-wise tree growth, proven baseline
            - LightGBM: Leaf-wise growth, histogram-based, fast
            - CatBoost: Ordered boosting, symmetric trees, robust

        Level 1 (Meta-Learner):
            - Ridge Regression: Combines base model predictions with L2 regularization

    Diversity Sources:
        - Different tree growth strategies (level-wise vs leaf-wise vs ordered)
        - Different regularization approaches
        - Different handling of features (histogram, symmetric trees, etc.)
    """

    def __init__(
        self,
        xgb_params: Optional[Dict] = None,
        lgb_params: Optional[Dict] = None,
        cat_params: Optional[Dict] = None,
        meta_params: Optional[Dict] = None,
        use_meta_learner: bool = True,
    ):
        """
        Initialize the tree ensemble.

        Args:
            xgb_params: XGBoost hyperparameters
            lgb_params: LightGBM hyperparameters
            cat_params: CatBoost hyperparameters
            meta_params: Ridge regression hyperparameters
            use_meta_learner: If False, use simple average instead of Ridge
        """
        # Default hyperparameters
        self.xgb_params = xgb_params or {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
            "gamma": 0,
            "random_state": 42,
            "objective": "reg:squarederror",
            "n_jobs": -1,
        }

        self.lgb_params = lgb_params or {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_samples": 20,
            "num_leaves": 63,  # 2^6 - 1 for depth=6
            "boosting_type": "gbdt",
            "objective": "regression",
            "metric": "mae",
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }

        self.cat_params = cat_params or {
            "iterations": 300,
            "depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bylevel": 0.8,
            "min_data_in_leaf": 20,
            "loss_function": "RMSE",
            "eval_metric": "MAE",
            "random_state": 42,
            "thread_count": -1,
            "verbose": False,
        }

        self.meta_params = meta_params or {
            "alpha": 1.0,  # L2 regularization
            "fit_intercept": True,
            "random_state": 42,
        }

        self.use_meta_learner = use_meta_learner

        # Initialize models
        self.xgb_model = xgb.XGBRegressor(**self.xgb_params)
        self.lgb_model = lgb.LGBMRegressor(**self.lgb_params)
        self.cat_model = cat.CatBoostRegressor(**self.cat_params)
        self.meta_model = Ridge(**self.meta_params) if use_meta_learner else None

        # Store base model names for tracking
        self.base_model_names = ["XGBoost", "LightGBM", "CatBoost"]

        # Track if models are fitted
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = True) -> "TreeEnsemblePredictor":
        """
        Train the ensemble (base models + meta-learner).

        Args:
            X: Training features (n_samples, n_features)
            y: Training targets (n_samples,)
            verbose: Whether to log training progress

        Returns:
            self (fitted ensemble)
        """
        if verbose:
            logger.info(f"\n{'='*80}")
            logger.info("TRAINING TREE ENSEMBLE")
            logger.info(f"{'='*80}")
            logger.info(f"Training samples: {len(X):,}")
            logger.info(f"Features: {X.shape[1]}")

        # Train base models
        if verbose:
            logger.info("\n1. Training base models...")

        # XGBoost
        if verbose:
            logger.info("   Training XGBoost...")
        self.xgb_model.fit(X, y, verbose=False)
        xgb_train_pred = self.xgb_model.predict(X)
        xgb_mae = mean_absolute_error(y, xgb_train_pred)
        if verbose:
            logger.info(f"      ✅ XGBoost trained - Train MAE: {xgb_mae:.2f}")

        # LightGBM
        if verbose:
            logger.info("   Training LightGBM...")
        self.lgb_model.fit(X, y)
        lgb_train_pred = self.lgb_model.predict(X)
        lgb_mae = mean_absolute_error(y, lgb_train_pred)
        if verbose:
            logger.info(f"      ✅ LightGBM trained - Train MAE: {lgb_mae:.2f}")

        # CatBoost
        if verbose:
            logger.info("   Training CatBoost...")
        self.cat_model.fit(X, y, verbose=False)
        cat_train_pred = self.cat_model.predict(X)
        cat_mae = mean_absolute_error(y, cat_train_pred)
        if verbose:
            logger.info(f"      ✅ CatBoost trained - Train MAE: {cat_mae:.2f}")

        # Train meta-learner
        if verbose:
            logger.info("\n2. Training meta-learner...")

        meta_X = self._get_meta_features(X)

        if self.use_meta_learner:
            self.meta_model.fit(meta_X, y)
            meta_weights = self.meta_model.coef_
            meta_intercept = self.meta_model.intercept_

            if verbose:
                logger.info(f"      Meta-learner weights:")
                logger.info(f"         XGBoost:  {meta_weights[0]:.3f}")
                logger.info(f"         LightGBM: {meta_weights[1]:.3f}")
                logger.info(f"         CatBoost: {meta_weights[2]:.3f}")
                logger.info(f"         Intercept: {meta_intercept:.3f}")

            ensemble_train_pred = self.meta_model.predict(meta_X)
        else:
            # Simple average
            ensemble_train_pred = meta_X.mean(axis=1)
            if verbose:
                logger.info(f"      Using simple average (equal weights)")

        ensemble_mae = mean_absolute_error(y, ensemble_train_pred)

        if verbose:
            logger.info(f"\n3. Training complete!")
            logger.info(f"      Ensemble Train MAE: {ensemble_mae:.2f}")
            logger.info(f"      Best base model MAE: {min(xgb_mae, lgb_mae, cat_mae):.2f}")
            logger.info(f"      Improvement: {min(xgb_mae, lgb_mae, cat_mae) - ensemble_mae:+.2f}")

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the ensemble.

        Args:
            X: Features to predict on (n_samples, n_features)

        Returns:
            Predictions (n_samples,)
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")

        meta_X = self._get_meta_features(X)

        if self.use_meta_learner:
            return self.meta_model.predict(meta_X)
        else:
            return meta_X.mean(axis=1)

    def predict_base_models(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Get predictions from each base model individually.

        Args:
            X: Features to predict on

        Returns:
            Dictionary mapping model name to predictions
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")

        return {
            "XGBoost": self.xgb_model.predict(X),
            "LightGBM": self.lgb_model.predict(X),
            "CatBoost": self.cat_model.predict(X),
        }

    def _get_meta_features(self, X: np.ndarray) -> np.ndarray:
        """
        Generate meta-features from base model predictions.

        Args:
            X: Input features

        Returns:
            Meta-features [pred_xgb, pred_lgb, pred_cat] (n_samples, 3)
        """
        return np.column_stack(
            [
                self.xgb_model.predict(X),
                self.lgb_model.predict(X),
                self.cat_model.predict(X),
            ]
        )

    def evaluate(self, X: np.ndarray, y: np.ndarray, verbose: bool = True) -> Dict[str, float]:
        """
        Evaluate ensemble and individual base models.

        Args:
            X: Evaluation features
            y: True targets
            verbose: Whether to print results

        Returns:
            Dictionary of metrics
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before evaluation")

        # Ensemble predictions
        ensemble_pred = self.predict(X)
        ensemble_mae = mean_absolute_error(y, ensemble_pred)
        ensemble_rmse = np.sqrt(mean_squared_error(y, ensemble_pred))
        ensemble_r2 = r2_score(y, ensemble_pred)

        # Base model predictions
        base_predictions = self.predict_base_models(X)

        results = {
            "ensemble_mae": ensemble_mae,
            "ensemble_rmse": ensemble_rmse,
            "ensemble_r2": ensemble_r2,
        }

        if verbose:
            logger.info(f"\n{'='*80}")
            logger.info("EVALUATION RESULTS")
            logger.info(f"{'='*80}")
            logger.info(f"\nSamples: {len(y):,}")

        # Evaluate each base model
        for model_name, preds in base_predictions.items():
            mae = mean_absolute_error(y, preds)
            rmse = np.sqrt(mean_squared_error(y, preds))
            r2 = r2_score(y, preds)

            results[f"{model_name.lower()}_mae"] = mae
            results[f"{model_name.lower()}_rmse"] = rmse
            results[f"{model_name.lower()}_r2"] = r2

            if verbose:
                logger.info(f"\n{model_name}:")
                logger.info(f"  MAE:  {mae:.2f}")
                logger.info(f"  RMSE: {rmse:.2f}")
                logger.info(f"  R²:   {r2:.3f}")

        if verbose:
            logger.info(f"\nENSEMBLE:")
            logger.info(f"  MAE:  {ensemble_mae:.2f}")
            logger.info(f"  RMSE: {ensemble_rmse:.2f}")
            logger.info(f"  R²:   {ensemble_r2:.3f}")

            best_base_mae = min(results[f"{m.lower()}_mae"] for m in self.base_model_names)
            improvement = best_base_mae - ensemble_mae

            logger.info(f"\nImprovement over best base model:")
            logger.info(f"  {improvement:+.2f} points ({improvement/best_base_mae*100:+.1f}%)")

        return results

    def get_feature_importance(self, feature_names: List[str]) -> Dict[str, np.ndarray]:
        """
        Get feature importance from each base model.

        Args:
            feature_names: List of feature names

        Returns:
            Dictionary mapping model name to feature importance arrays
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before getting feature importance")

        return {
            "XGBoost": self.xgb_model.feature_importances_,
            "LightGBM": self.lgb_model.feature_importances_,
            "CatBoost": self.cat_model.feature_importances_,
        }

    def analyze_diversity(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Analyze diversity between base models.

        Args:
            X: Features
            y: True targets

        Returns:
            Dictionary with correlation coefficients and diversity metrics
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before analyzing diversity")

        # Get predictions
        preds = self.predict_base_models(X)
        pred_xgb = preds["XGBoost"]
        pred_lgb = preds["LightGBM"]
        pred_cat = preds["CatBoost"]

        # Calculate correlations
        corr_xgb_lgb = np.corrcoef(pred_xgb, pred_lgb)[0, 1]
        corr_xgb_cat = np.corrcoef(pred_xgb, pred_cat)[0, 1]
        corr_lgb_cat = np.corrcoef(pred_lgb, pred_cat)[0, 1]

        avg_correlation = (corr_xgb_lgb + corr_xgb_cat + corr_lgb_cat) / 3

        # Calculate error diversity (disagreement)
        errors_xgb = pred_xgb - y
        errors_lgb = pred_lgb - y
        errors_cat = pred_cat - y

        # Average absolute difference in errors
        error_div_xgb_lgb = np.mean(np.abs(errors_xgb - errors_lgb))
        error_div_xgb_cat = np.mean(np.abs(errors_xgb - errors_cat))
        error_div_lgb_cat = np.mean(np.abs(errors_lgb - errors_cat))

        avg_error_diversity = (error_div_xgb_lgb + error_div_xgb_cat + error_div_lgb_cat) / 3

        logger.info(f"\n{'='*80}")
        logger.info("BASE MODEL DIVERSITY ANALYSIS")
        logger.info(f"{'='*80}")
        logger.info(f"\nPrediction Correlations:")
        logger.info(f"  XGBoost <-> LightGBM: {corr_xgb_lgb:.3f}")
        logger.info(f"  XGBoost <-> CatBoost: {corr_xgb_cat:.3f}")
        logger.info(f"  LightGBM <-> CatBoost: {corr_lgb_cat:.3f}")
        logger.info(f"  Average: {avg_correlation:.3f}")
        logger.info(f"\nError Diversity (lower = more diverse):")
        logger.info(f"  XGBoost <-> LightGBM: {error_div_xgb_lgb:.2f}")
        logger.info(f"  XGBoost <-> CatBoost: {error_div_xgb_cat:.2f}")
        logger.info(f"  LightGBM <-> CatBoost: {error_div_lgb_cat:.2f}")
        logger.info(f"  Average: {avg_error_diversity:.2f}")

        if avg_correlation > 0.95:
            logger.warning(
                f"\n⚠️  High correlation ({avg_correlation:.3f}) - models may be too similar"
            )
        elif avg_correlation < 0.75:
            logger.info(
                f"\n✅ Good diversity ({avg_correlation:.3f}) - models are sufficiently different"
            )
        else:
            logger.info(f"\n✓  Moderate diversity ({avg_correlation:.3f})")

        return {
            "corr_xgb_lgb": corr_xgb_lgb,
            "corr_xgb_cat": corr_xgb_cat,
            "corr_lgb_cat": corr_lgb_cat,
            "avg_correlation": avg_correlation,
            "error_div_xgb_lgb": error_div_xgb_lgb,
            "error_div_xgb_cat": error_div_xgb_cat,
            "error_div_lgb_cat": error_div_lgb_cat,
            "avg_error_diversity": avg_error_diversity,
        }
