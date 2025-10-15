"""
Custom Exceptions for NBA Props Model

Provides specific exception types for better error handling and debugging.
"""


class NBAPropsModelError(Exception):
    """Base exception for all NBA props model errors."""

    pass


class DataNotFoundError(NBAPropsModelError):
    """Raised when required data files are not found."""

    pass


class InvalidInputError(NBAPropsModelError):
    """Raised when input data is invalid or malformed."""

    pass


class FeatureCalculationError(NBAPropsModelError):
    """Raised when feature calculation fails."""

    pass


class InsufficientDataError(NBAPropsModelError):
    """Raised when insufficient historical data is available for prediction."""

    pass


class ModelNotTrainedError(NBAPropsModelError):
    """Raised when trying to use a model that hasn't been trained."""

    pass


class CTGDataError(NBAPropsModelError):
    """Raised when CTG data is missing or invalid."""

    pass


class PredictionError(NBAPropsModelError):
    """Raised when prediction fails."""

    pass
