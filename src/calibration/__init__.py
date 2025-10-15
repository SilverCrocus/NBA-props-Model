"""
Calibration module for NBA Props predictions.

This module provides probability calibration techniques to improve
betting performance by fixing the "large edges underperform" problem.
"""

from src.calibration.isotonic_calibrator import IsotonicCalibrator

__all__ = ["IsotonicCalibrator"]
