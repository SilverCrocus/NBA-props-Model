#!/usr/bin/env python3
"""
Make Predictions with Ensemble Model
=====================================

Loads the 3-fold ensemble models and makes predictions by:
1. Getting predictions from all 3 models
2. Averaging the predictions (simple ensemble)
3. Applying averaged calibration from all 3 calibrators
4. Returning final calibrated prediction

Usage:
    from scripts.production.predict_ensemble import EnsemblePredictor

    predictor = EnsemblePredictor()
    prediction = predictor.predict(game_features_df)

Or standalone:
    uv run python scripts/production/predict_ensemble.py
"""

import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


class EnsemblePredictor:
    """Ensemble predictor using 3-fold time-series CV models."""

    def __init__(self, models_dir: str = "models"):
        """
        Initialize ensemble predictor.

        Args:
            models_dir: Directory containing ensemble model files
        """
        self.models_dir = Path(models_dir)
        self.models = []
        self.calibrators = []
        self.feature_cols = None
        self.meta = None

        self._load_models()

    def _load_models(self):
        """Load all fold models and calibrators."""
        print("Loading ensemble models...")

        # Load metadata
        meta_path = self.models_dir / "ensemble_meta.pkl"
        if meta_path.exists():
            with open(meta_path, "rb") as f:
                self.meta = pickle.load(f)
            print(f"   ✅ Loaded metadata")
        else:
            raise FileNotFoundError(f"Ensemble metadata not found at {meta_path}")

        # Load each fold model
        for i in range(1, 4):
            model_path = self.models_dir / f"ensemble_fold_{i}.pkl"

            if not model_path.exists():
                raise FileNotFoundError(f"Model not found: {model_path}")

            with open(model_path, "rb") as f:
                fold_dict = pickle.load(f)

            self.models.append(fold_dict["model"])
            self.calibrators.append(fold_dict["calibrator"])

            if self.feature_cols is None:
                self.feature_cols = fold_dict["feature_cols"]

            print(f"   ✅ Loaded Fold {i}")

        print(f"\nEnsemble ready: {len(self.models)} models")
        print(f"Features: {len(self.feature_cols)}")
        print()

    def predict(self, X: pd.DataFrame, return_components: bool = False) -> np.ndarray:
        """
        Make ensemble predictions.

        Args:
            X: DataFrame with features (must have all required feature columns)
            return_components: If True, return dict with individual predictions

        Returns:
            Array of final calibrated predictions (or dict if return_components=True)
        """
        # Ensure X has all required features
        X_prepared = X[self.feature_cols].fillna(0)

        # Get predictions from all 3 models
        raw_predictions = []
        for model in self.models:
            pred = model.predict(X_prepared)
            raw_predictions.append(pred)

        # Ensemble (simple average)
        ensemble_raw = np.mean(raw_predictions, axis=0)

        # Apply each calibrator and average
        calibrated_predictions = []
        for calibrator in self.calibrators:
            pred_calibrated = calibrator.transform(ensemble_raw)
            calibrated_predictions.append(pred_calibrated)

        # Final prediction (averaged calibration)
        final_prediction = np.mean(calibrated_predictions, axis=0)

        if return_components:
            return {
                "raw_predictions": raw_predictions,
                "ensemble_raw": ensemble_raw,
                "calibrated_predictions": calibrated_predictions,
                "final": final_prediction,
            }
        else:
            return final_prediction

    def predict_single_game(self, game_features: Dict) -> float:
        """
        Make prediction for a single game.

        Args:
            game_features: Dictionary of feature values

        Returns:
            Final calibrated PRA prediction
        """
        # Convert to DataFrame
        df = pd.DataFrame([game_features])

        # Make prediction
        prediction = self.predict(df)

        return prediction[0]

    def get_model_info(self) -> Dict:
        """Get information about the ensemble models."""
        if self.meta is None:
            return {}

        return {
            "num_models": len(self.models),
            "feature_count": len(self.feature_cols),
            "cv_mae": np.mean([r["mae_calibrated"] for r in self.meta["fold_results"]]),
            "test_mae": self.meta.get("test_mae"),
            "created_at": self.meta.get("created_at"),
            "folds": [f["name"] for f in self.meta["folds"]],
        }


# ======================================================================
# EXAMPLE USAGE
# ======================================================================


def main():
    """Example usage of ensemble predictor."""
    print("=" * 80)
    print("ENSEMBLE PREDICTOR - EXAMPLE USAGE")
    print("=" * 80)
    print()

    # Initialize predictor
    predictor = EnsemblePredictor()

    # Get model info
    info = predictor.get_model_info()
    print("Model Information:")
    print(f"   Models: {info['num_models']}")
    print(f"   Features: {info['feature_count']}")
    print(f"   CV MAE: {info['cv_mae']:.2f} pts")
    print(f"   Test MAE: {info['test_mae']:.2f} pts")
    print(f"   Created: {info['created_at']}")
    print()

    # Example: Load some test data and make predictions
    print("Loading test data...")
    try:
        # Try to load actual test data
        import sys

        sys.path.append("scripts/utils")
        from fast_feature_builder import FastFeatureBuilder

        # Load recent games
        df = pd.read_csv("data/game_logs/all_game_logs_through_2025.csv")
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
        df = df[df["GAME_DATE"] >= "2023-01-01"].copy()

        # Add PRA
        if "PRA" not in df.columns:
            df["PRA"] = df["PTS"] + df["REB"] + df["AST"]

        # Build features
        builder = FastFeatureBuilder()
        df_with_features = builder.build_features(df, pd.DataFrame(), verbose=False)

        # Take sample
        sample = df_with_features.sample(min(10, len(df_with_features)))

        print(f"   ✅ Loaded {len(sample)} sample games")
        print()

        # Make predictions
        print("Making predictions...")
        predictions = predictor.predict(sample)

        # Compare to actual
        actual = sample["PRA"].values

        results = pd.DataFrame(
            {
                "Player": sample["PLAYER_NAME"].values,
                "Date": sample["GAME_DATE"].dt.strftime("%Y-%m-%d").values,
                "Actual_PRA": actual,
                "Predicted_PRA": predictions,
                "Error": actual - predictions,
            }
        )

        print(results.to_string(index=False))
        print()

        mae = np.mean(np.abs(results["Error"]))
        print(f"Sample MAE: {mae:.2f} pts")
        print()

    except Exception as e:
        print(f"   Could not load test data: {e}")
        print("   (This is expected if running without game data)")
        print()

    print("=" * 80)
    print("✅ Ensemble predictor ready for use!")
    print("=" * 80)


if __name__ == "__main__":
    main()
