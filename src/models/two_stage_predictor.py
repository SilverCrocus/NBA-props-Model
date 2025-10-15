"""
Two-Stage Predictor for NBA Props

This is the KEY INNOVATION identified by all 4 specialized agents:
Minutes variance accounts for 40-50% of current MAE 6.10.

Stage 1: Predict minutes played
Stage 2: Predict PRA given predicted minutes

Expected Impact: MAE 6.10 → 5.60-5.80 (7-8% improvement)

Why this works:
- Player plays 40 min → 50 PRA
- Same player plays 20 min → 25 PRA
- Current model predicts 37.5 PRA (wrong by 12.5 either way)
- Two-stage model predicts minutes first, then adjusts PRA accordingly

Author: NBA Props Model - Phase 1 Weeks 2-3
Date: October 14, 2025
"""

import logging
from typing import Dict, Tuple

import catboost as cat
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TwoStagePredictor:
    """
    Two-stage predictor: Minutes → PRA

    Stage 1: Predicts minutes played based on:
        - Recent minutes trends (MIN_L5_mean, MIN_lag1, etc.)
        - Starter status
        - Rest days, back-to-backs
        - Team/opponent pace
        - Season (early/late season patterns)

    Stage 2: Predicts PRA given predicted minutes:
        - Predicted minutes (from Stage 1)
        - Per-minute efficiency (PRA_per_36, CTG_USG, etc.)
        - Opponent defense
        - Recent performance trends

    Usage:
        predictor = TwoStagePredictor()
        predictor.fit(X_train, y_train_pra, y_train_minutes)
        predictions = predictor.predict(X_test)
    """

    def __init__(self, minutes_model_params: Dict = None, pra_model_params: Dict = None):
        """
        Initialize two-stage predictor.

        Args:
            minutes_model_params: Hyperparameters for Stage 1 (minutes) model
            pra_model_params: Hyperparameters for Stage 2 (PRA) model
        """
        # Default hyperparameters for minutes model (CatBoost - best from Phase 2)
        if minutes_model_params is None:
            minutes_model_params = {
                "iterations": 300,
                "depth": 5,  # Shallower than PRA model
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

        # Default hyperparameters for PRA model (CatBoost - best from Phase 2)
        if pra_model_params is None:
            pra_model_params = {
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

        self.minutes_model = cat.CatBoostRegressor(**minutes_model_params)
        self.pra_model = cat.CatBoostRegressor(**pra_model_params)

        self.minutes_features = None
        self.pra_features = None
        self.is_fitted = False

    def _select_minutes_features(self, X: pd.DataFrame) -> list:
        """
        Select features relevant for predicting minutes.

        Minutes are primarily determined by:
        - Recent minutes patterns
        - Rest/fatigue
        - Starter status (if available)
        - Team pace
        """
        minutes_features = []

        # Recent minutes features (critical)
        for col in X.columns:
            if "MIN" in col and col != "MIN":  # Exclude actual MIN if present
                minutes_features.append(col)

        # Rest and fatigue features
        rest_features = ["days_rest", "is_b2b", "games_last_7d"]
        for feat in rest_features:
            if feat in X.columns:
                minutes_features.append(feat)

        # Pace features (if available)
        pace_features = ["opp_pace", "CTG_USG"]  # Usage rate correlates with minutes
        for feat in pace_features:
            if feat in X.columns:
                minutes_features.append(feat)

        # Age-related (older players get load managed)
        # Note: We don't have age in current features, but would add if available

        if len(minutes_features) == 0:
            raise ValueError("No suitable features found for minutes prediction!")

        logger.info(f"   Selected {len(minutes_features)} features for minutes model")
        logger.info(f"   Key features: {minutes_features[:10]}")

        return minutes_features

    def _select_pra_features(self, X: pd.DataFrame, include_predicted_minutes: bool = True) -> list:
        """
        Select features relevant for predicting PRA given minutes.

        PRA given minutes depends on:
        - Per-minute efficiency
        - Recent performance trends
        - Opponent defense
        - CTG advanced stats
        """
        pra_features = []

        # ALWAYS include predicted minutes (this is the key innovation)
        if include_predicted_minutes:
            pra_features.append("predicted_MIN")

        # Per-minute efficiency features
        efficiency_features = ["PRA_per_36", "PTS_per_36", "TS_pct", "eFG_pct", "PTS_per_shot"]
        for feat in efficiency_features:
            if feat in X.columns:
                pra_features.append(feat)

        # CTG advanced stats (usage, shooting, playmaking)
        ctg_features = ["CTG_USG", "CTG_PSA", "CTG_AST_PCT", "CTG_eFG", "CTG_TOV_PCT"]
        for feat in ctg_features:
            if feat in X.columns:
                pra_features.append(feat)

        # Recent performance (EWMA and rolling averages)
        for col in X.columns:
            if any(pattern in col for pattern in ["ewma", "L5_mean", "L10_mean", "L20_mean"]):
                pra_features.append(col)

        # Opponent features
        opponent_features = ["opp_DRtg", "opp_pace", "opp_PRA_allowed"]
        for feat in opponent_features:
            if feat in X.columns:
                pra_features.append(feat)

        # Rest features (fatigue affects per-minute efficiency)
        rest_features = ["days_rest", "is_b2b"]
        for feat in rest_features:
            if feat in X.columns:
                pra_features.append(feat)

        # Remove duplicates while preserving order
        pra_features = list(dict.fromkeys(pra_features))

        logger.info(f"   Selected {len(pra_features)} features for PRA model")
        logger.info(f"   Key features: {pra_features[:10]}")

        return pra_features

    def fit(
        self, X: pd.DataFrame, y_pra: pd.Series, y_minutes: pd.Series = None
    ) -> Dict[str, float]:
        """
        Fit both stages of the predictor.

        Args:
            X: Feature matrix
            y_pra: Target PRA values
            y_minutes: Target minutes values (if None, extracted from X['MIN'])

        Returns:
            Dictionary of training metrics
        """
        logger.info("\n" + "=" * 80)
        logger.info("FITTING TWO-STAGE PREDICTOR")
        logger.info("=" * 80)

        # Extract minutes if not provided
        if y_minutes is None:
            if "MIN" not in X.columns:
                raise ValueError("Either provide y_minutes or include 'MIN' column in X")
            y_minutes = X["MIN"].copy()
            logger.info("   Using 'MIN' column from X as target for Stage 1")

        # Select features
        logger.info("\n1. Selecting features...")
        self.minutes_features = self._select_minutes_features(X)
        self.pra_features = self._select_pra_features(
            X, include_predicted_minutes=False
        )  # Will add predicted_MIN during prediction

        # Stage 1: Train minutes model
        logger.info("\n2. Training Stage 1 (Minutes Prediction)...")

        X_minutes = X[self.minutes_features].fillna(0)
        self.minutes_model.fit(X_minutes, y_minutes)

        # Predict minutes on training set
        predicted_minutes_train = self.minutes_model.predict(X_minutes)

        # Calculate minutes metrics
        minutes_mae = mean_absolute_error(y_minutes, predicted_minutes_train)
        minutes_r2 = r2_score(y_minutes, predicted_minutes_train)

        logger.info(f"   ✅ Minutes Model Trained")
        logger.info(f"      MAE: {minutes_mae:.2f} minutes")
        logger.info(f"      R²: {minutes_r2:.3f}")

        # Stage 2: Train PRA model with predicted minutes
        logger.info("\n3. Training Stage 2 (PRA given Minutes)...")

        # Create feature matrix with predicted minutes
        X_pra = X[self.pra_features].copy()
        X_pra["predicted_MIN"] = predicted_minutes_train
        X_pra = X_pra.fillna(0)

        self.pra_model.fit(X_pra, y_pra)

        # Predict PRA on training set
        predicted_pra_train = self.pra_model.predict(X_pra)

        # Calculate PRA metrics
        pra_mae = mean_absolute_error(y_pra, predicted_pra_train)
        pra_r2 = r2_score(y_pra, predicted_pra_train)

        logger.info(f"   ✅ PRA Model Trained")
        logger.info(f"      MAE: {pra_mae:.2f} points")
        logger.info(f"      R²: {pra_r2:.3f}")

        self.is_fitted = True

        # Return metrics
        metrics = {
            "minutes_mae": minutes_mae,
            "minutes_r2": minutes_r2,
            "pra_mae": pra_mae,
            "pra_r2": pra_r2,
            "n_samples": len(X),
            "n_minutes_features": len(self.minutes_features),
            "n_pra_features": len(self.pra_features) + 1,  # +1 for predicted_MIN
        }

        logger.info("\n" + "=" * 80)
        logger.info("TWO-STAGE TRAINING COMPLETE")
        logger.info("=" * 80)

        return metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using two-stage approach.

        Args:
            X: Feature matrix

        Returns:
            Predicted PRA values
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction. Call fit() first.")

        # Stage 1: Predict minutes
        X_minutes = X[self.minutes_features].fillna(0)
        predicted_minutes = self.minutes_model.predict(X_minutes)

        # Stage 2: Predict PRA given predicted minutes
        X_pra = X[self.pra_features].copy()
        X_pra["predicted_MIN"] = predicted_minutes
        X_pra = X_pra.fillna(0)

        predicted_pra = self.pra_model.predict(X_pra)

        return predicted_pra

    def predict_with_minutes(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions and return both minutes and PRA predictions.

        Args:
            X: Feature matrix

        Returns:
            Tuple of (predicted_pra, predicted_minutes)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction. Call fit() first.")

        # Stage 1: Predict minutes
        X_minutes = X[self.minutes_features].fillna(0)
        predicted_minutes = self.minutes_model.predict(X_minutes)

        # Stage 2: Predict PRA given predicted minutes
        X_pra = X[self.pra_features].copy()
        X_pra["predicted_MIN"] = predicted_minutes
        X_pra = X_pra.fillna(0)

        predicted_pra = self.pra_model.predict(X_pra)

        return predicted_pra, predicted_minutes

    def get_feature_importance(self) -> Dict[str, pd.DataFrame]:
        """
        Get feature importance from both stages.

        Returns:
            Dictionary with 'minutes' and 'pra' DataFrames
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted to get feature importance")

        # Minutes model importance
        minutes_importance = pd.DataFrame(
            {
                "feature": self.minutes_features,
                "importance": self.minutes_model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)

        # PRA model importance (including predicted_MIN)
        pra_feature_names = self.pra_features + ["predicted_MIN"]
        pra_importance = pd.DataFrame(
            {"feature": pra_feature_names, "importance": self.pra_model.feature_importances_}
        ).sort_values("importance", ascending=False)

        return {"minutes": minutes_importance, "pra": pra_importance}

    def save(self, path_prefix: str):
        """
        Save both models to disk.

        Args:
            path_prefix: Prefix for model files (e.g., 'models/two_stage')
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted models")

        import pickle

        # Save models (CatBoost format)
        self.minutes_model.save_model(f"{path_prefix}_minutes.cbm")
        self.pra_model.save_model(f"{path_prefix}_pra.cbm")

        # Save feature lists
        with open(f"{path_prefix}_features.pkl", "wb") as f:
            pickle.dump(
                {"minutes_features": self.minutes_features, "pra_features": self.pra_features}, f
            )

        logger.info(f"✅ Two-stage model saved to {path_prefix}_*.cbm/pkl")

    @classmethod
    def load(cls, path_prefix: str) -> "TwoStagePredictor":
        """
        Load both models from disk.

        Args:
            path_prefix: Prefix for model files

        Returns:
            Loaded TwoStagePredictor
        """
        import pickle

        # Create instance
        predictor = cls()

        # Load models (CatBoost format)
        predictor.minutes_model.load_model(f"{path_prefix}_minutes.cbm")
        predictor.pra_model.load_model(f"{path_prefix}_pra.cbm")

        # Load feature lists
        with open(f"{path_prefix}_features.pkl", "rb") as f:
            features = pickle.load(f)
            predictor.minutes_features = features["minutes_features"]
            predictor.pra_features = features["pra_features"]

        predictor.is_fitted = True

        logger.info(f"✅ Two-stage model loaded from {path_prefix}_*.cbm/pkl")

        return predictor
