"""
Proper time-series cross-validation for NBA props model.
No fallback logic - direct implementation only.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class TimeSeriesValidator:
    """Implement proper time-series validation for NBA predictions."""

    def __init__(self, n_splits: int = 5, test_size: int = 1000):
        """Initialize validator with number of splits and test size."""
        self.n_splits = n_splits
        self.test_size = test_size

    def validate_model(self, model, X: pd.DataFrame, y: pd.Series, dates: pd.Series) -> dict:
        """Validate model using proper time-series splits."""
        # Ensure chronological order
        sort_idx = dates.argsort()
        X_sorted = X.iloc[sort_idx]
        y_sorted = y.iloc[sort_idx]
        dates_sorted = dates.iloc[sort_idx]

        # Use TimeSeriesSplit for validation
        tscv = TimeSeriesSplit(n_splits=self.n_splits, test_size=self.test_size)

        results = {
            'mae': [],
            'rmse': [],
            'r2': [],
            'fold_dates': []
        }

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X_sorted)):
            # Get train/test sets
            X_train = X_sorted.iloc[train_idx]
            y_train = y_sorted.iloc[train_idx]
            X_test = X_sorted.iloc[test_idx]
            y_test = y_sorted.iloc[test_idx]

            # Record date ranges
            train_dates = (dates_sorted.iloc[train_idx].min(), dates_sorted.iloc[train_idx].max())
            test_dates = (dates_sorted.iloc[test_idx].min(), dates_sorted.iloc[test_idx].max())

            # Train model
            model.fit(X_train, y_train)

            # Predict
            y_pred = model.predict(X_test)

            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            results['mae'].append(mae)
            results['rmse'].append(rmse)
            results['r2'].append(r2)
            results['fold_dates'].append({
                'fold': fold + 1,
                'train': train_dates,
                'test': test_dates
            })

        # Calculate summary statistics
        results['summary'] = {
            'mae_mean': np.mean(results['mae']),
            'mae_std': np.std(results['mae']),
            'rmse_mean': np.mean(results['rmse']),
            'rmse_std': np.std(results['rmse']),
            'r2_mean': np.mean(results['r2']),
            'r2_std': np.std(results['r2'])
        }

        return results

    def walk_forward_validation(self, model, X: pd.DataFrame, y: pd.Series,
                               dates: pd.Series, initial_train_size: int = 5000,
                               step_size: int = 500) -> dict:
        """Implement walk-forward validation for realistic backtesting."""
        # Sort by date
        sort_idx = dates.argsort()
        X_sorted = X.iloc[sort_idx]
        y_sorted = y.iloc[sort_idx]
        dates_sorted = dates.iloc[sort_idx]

        predictions = []
        actuals = []
        prediction_dates = []

        # Start with initial training set
        train_end = initial_train_size

        while train_end < len(X_sorted) - step_size:
            # Train set: all data up to train_end
            X_train = X_sorted.iloc[:train_end]
            y_train = y_sorted.iloc[:train_end]

            # Test set: next step_size samples
            test_start = train_end
            test_end = min(train_end + step_size, len(X_sorted))
            X_test = X_sorted.iloc[test_start:test_end]
            y_test = y_sorted.iloc[test_start:test_end]

            # Train and predict
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Store results
            predictions.extend(y_pred)
            actuals.extend(y_test)
            prediction_dates.extend(dates_sorted.iloc[test_start:test_end])

            # Move forward
            train_end += step_size

        # Calculate overall metrics
        mae = mean_absolute_error(actuals, predictions)
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        r2 = r2_score(actuals, predictions)

        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'n_predictions': len(predictions),
            'date_range': (min(prediction_dates), max(prediction_dates)),
            'predictions': predictions,
            'actuals': actuals,
            'dates': prediction_dates
        }

    def season_based_validation(self, model, df: pd.DataFrame,
                               feature_cols: list, target_col: str,
                               season_col: str = 'season') -> dict:
        """Validate by training on past seasons and testing on current."""
        if season_col not in df.columns:
            raise ValueError(f"Season column '{season_col}' not found")

        seasons = sorted(df[season_col].unique())

        if len(seasons) < 2:
            raise ValueError("Need at least 2 seasons for validation")

        results = []

        # Train on each season, test on next
        for i in range(len(seasons) - 1):
            train_seasons = seasons[:i+1]
            test_season = seasons[i+1]

            # Split data
            train_df = df[df[season_col].isin(train_seasons)]
            test_df = df[df[season_col] == test_season]

            X_train = train_df[feature_cols]
            y_train = train_df[target_col]
            X_test = test_df[feature_cols]
            y_test = test_df[target_col]

            # Train and predict
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            results.append({
                'train_seasons': train_seasons,
                'test_season': test_season,
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'n_train': len(train_df),
                'n_test': len(test_df)
            })

        return results