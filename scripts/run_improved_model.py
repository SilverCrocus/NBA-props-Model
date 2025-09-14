"""
Run improved NBA props model with opponent features and proper validation
Simple script to test the improvements
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from models.train_model import NBAPropsTrainer
from features.opponent_features import OpponentFeatures
from models.validation import NBATimeSeriesValidator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_sample_data():
    """
    Load and prepare sample data from CTG files
    You'll need to implement actual data loading based on your file structure
    """
    # Example structure - adjust based on your actual data
    data_path = Path("/Users/diyagamah/Documents/nba_props_model/data")

    # This is a placeholder - implement actual data loading
    # You need to:
    # 1. Load player stats from ctg_data_organized
    # 2. Merge different stat types (shooting, rebounding, etc.)
    # 3. Add game dates and opponent info

    logger.info("Load your CTG data here")
    # df = pd.read_csv(...)
    # return df


def main():
    """
    Main execution function
    """
    logger.info("Starting improved NBA props model training...")

    # Initialize trainer
    trainer = NBAPropsTrainer()

    # Load data (implement load_sample_data based on your file structure)
    # df = load_sample_data()

    # For testing, create sample data structure
    # This shows the expected format
    sample_df = pd.DataFrame({
        'Player': ['Player1'] * 50 + ['Player2'] * 50,
        'Date': pd.date_range('2024-01-01', periods=50).tolist() * 2,
        'Season': ['2023-24'] * 100,
        'Team': ['BOS'] * 50 + ['LAL'] * 50,
        'Opponent': ['MIA', 'NYK', 'PHI', 'BKN', 'TOR'] * 20,
        'Points': np.random.normal(20, 5, 100),
        'Rebounds': np.random.normal(5, 2, 100),
        'Assists': np.random.normal(5, 2, 100),
        'Minutes': np.random.normal(30, 5, 100),
        'Usage': np.random.normal(25, 5, 100),
        'PSA': np.random.normal(110, 10, 100),
        'AST%': np.random.normal(20, 5, 100),
        'fgDR%': np.random.normal(15, 3, 100),
        'fgOR%': np.random.normal(5, 2, 100),
    })

    logger.info(f"Data shape: {sample_df.shape}")

    # Prepare training data
    try:
        df_prepared = trainer.prepare_training_data(sample_df)
        logger.info(f"Prepared data shape: {df_prepared.shape}")

        # Train model
        model_results = trainer.train_model(df_prepared, model_type='xgboost')

        # Print results
        logger.info("\n=== Model Training Complete ===")
        logger.info(f"Training samples: {model_results['training_samples']}")
        logger.info(f"Number of features: {len(model_results['features'])}")

        # Print validation metrics
        val_results = model_results['validation_results']['overall']
        logger.info(f"\nOverall Validation Metrics:")
        logger.info(f"  MAE: {val_results['mae']:.2f} ± {val_results['mae_std']:.2f}")
        logger.info(f"  RMSE: {val_results['rmse']:.2f} ± {val_results['rmse_std']:.2f}")
        logger.info(f"  R²: {val_results['r2']:.3f}")

        # Print top features
        logger.info(f"\nTop 10 Features:")
        for _, row in model_results['feature_importance'].head(10).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.3f}")

        # Save model
        model_path = trainer.save_model(model_results)
        logger.info(f"\nModel saved to: {model_path}")

    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise


if __name__ == "__main__":
    main()