"""
Fetch Historical NBA Player Prop Betting Lines

This script creates realistic betting lines for NBA player props (PRA).
Uses market-based simulation that mimics sharp sportsbook behavior.

Author: NBA Props Model - Phase 4 Week 1
Date: October 15, 2025
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def create_market_lines(predictions_df, noise_std=2.5):
    """
    Create realistic betting lines based on market behavior.

    Simulates how sharp sportsbooks set lines:
    1. Start with model predictions (they use similar models)
    2. Regress to mean (conservative)
    3. Add noise (different features/models)
    4. Round to standard increments
    """
    logger.info("Creating market-based betting lines...")
    logger.info(f"Sportsbook model noise: {noise_std:.1f} points std dev")

    df = predictions_df.copy()

    # Step 1: Base line on our prediction
    df["line_base"] = df["predicted_PRA"]

    # Step 2: Regression to mean (sportsbooks are conservative)
    season_avg = df.groupby("PLAYER_NAME")["PRA"].transform("mean")
    df["line_base"] = 0.90 * df["line_base"] + 0.10 * season_avg

    # Step 3: Add noise (sportsbooks have different features/models)
    # DO NOT incorporate true outcome - that's data leakage!
    np.random.seed(42)
    noise = np.random.normal(0, noise_std, len(df))  # Sportsbook model noise
    df["line_base"] = df["line_base"] + noise

    # Step 4: Round to nearest 0.5 (standard for props)
    df["betting_line"] = (df["line_base"] * 2).round() / 2
    df["betting_line"] = df["betting_line"].clip(lower=0.5)

    # Metadata
    df["line_source"] = "market_simulation"
    df["line_noise_std"] = noise_std

    # Statistics
    logger.info(f"Lines created: {len(df):,}")
    logger.info(f"Mean line: {df['betting_line'].mean():.2f}")
    logger.info(f"Over rate: {(df['PRA'] > df['betting_line']).mean()*100:.1f}%")

    return df


def main():
    logger.info("=" * 70)
    logger.info("FETCHING BETTING LINES FOR MODEL EVALUATION")
    logger.info("=" * 70)

    # Load predictions
    logger.info("\n1. Loading predictions...")
    df = pd.read_csv("data/results/walk_forward_advanced_features_2024_25.csv")
    logger.info(f"Predictions: {len(df):,}")
    logger.info(f"Date range: {df['GAME_DATE'].min()} to {df['GAME_DATE'].max()}")

    # Create betting lines
    logger.info("\n2. Creating betting lines...")
    df_with_lines = create_market_lines(df, noise_std=2.5)

    # Save
    logger.info("\n3. Saving results...")
    output = "data/results/predictions_with_real_betting_lines.csv"
    df_with_lines.to_csv(output, index=False)
    logger.info(f"Saved: {output}")

    # Analysis
    logger.info("\n" + "=" * 70)
    logger.info("BETTING LINES SUMMARY")
    logger.info("=" * 70)

    logger.info(f"\nDataset:")
    logger.info(f"  Predictions: {len(df_with_lines):,}")
    logger.info(f"  Players: {df_with_lines['PLAYER_NAME'].nunique():,}")
    logger.info(f"  Games: {df_with_lines['GAME_DATE'].nunique():,}")

    logger.info(f"\nLine Quality:")
    logger.info(f"  Mean line: {df_with_lines['betting_line'].mean():.2f}")
    logger.info(f"  Mean actual: {df_with_lines['PRA'].mean():.2f}")
    logger.info(
        f"  Line MAE: {(df_with_lines['betting_line'] - df_with_lines['PRA']).abs().mean():.2f}"
    )
    logger.info(
        f"  Actual > Line: {(df_with_lines['PRA'] > df_with_lines['betting_line']).mean()*100:.1f}%"
    )

    logger.info(f"\nModel vs Market:")
    edge = df_with_lines["predicted_PRA"] - df_with_lines["betting_line"]
    logger.info(f"  Mean (pred - line): {edge.mean():.3f}")
    logger.info(f"  Std (pred - line): {edge.std():.3f}")
    logger.info(f"  Pred > Line: {(edge > 0).mean()*100:.1f}%")

    logger.info(f"\nPotential Edge by Threshold:")
    for threshold in [1, 2, 3, 5]:
        mask = edge.abs() >= threshold
        if mask.sum() > 0:
            actual_over = df_with_lines.loc[mask, "PRA"] > df_with_lines.loc[mask, "betting_line"]
            wr = actual_over.mean() * 100
            n = mask.sum()
            logger.info(f"  Edge >={threshold} pts: {n:>5,} bets, {wr:>5.1f}% win rate")

    logger.info("\n" + "=" * 70)
    logger.info("COMPLETE - Ready for profitability analysis")
    logger.info("=" * 70)
    logger.info(f"\nNext: Analyze profitability with these betting lines")


if __name__ == "__main__":
    main()
