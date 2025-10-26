"""
Custom Exceptions for NBA Props Model

Provides specific exception types for better error handling and debugging.
"""


class NBAPropsModelError(Exception):
    """Base exception for all NBA props model errors."""


class DataNotFoundError(NBAPropsModelError):
    """Raised when required data files are not found."""


class InvalidInputError(NBAPropsModelError):
    """Raised when input data is invalid or malformed."""


class FeatureCalculationError(NBAPropsModelError):
    """Raised when feature calculation fails."""


class InsufficientDataError(NBAPropsModelError):
    """Raised when insufficient historical data is available for prediction."""


class ModelNotTrainedError(NBAPropsModelError):
    """Raised when trying to use a model that hasn't been trained."""


class CTGDataError(NBAPropsModelError):
    """Raised when CTG data is missing or invalid."""


class PredictionError(NBAPropsModelError):
    """Raised when prediction fails."""
