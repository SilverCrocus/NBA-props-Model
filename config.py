"""
Configuration Management for NBA Props Model

Centralized configuration for all hyperparameters, paths, and settings.
This eliminates hardcoded values scattered across scripts.

Usage:
    from config import data_config, model_config, validation_config

    df = pd.read_csv(data_config.GAME_LOGS_PATH)
    model = xgb.XGBRegressor(**model_config.XGBOOST_PARAMS)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

# Project root directory
PROJECT_ROOT = Path(__file__).parent


@dataclass
class DataConfig:
    """Data paths and filtering parameters."""

    # Data paths
    GAME_LOGS_PATH: Path = PROJECT_ROOT / "data" / "game_logs" / "all_game_logs_with_opponent.csv"
    CTG_DATA_PATH: Path = PROJECT_ROOT / "data" / "ctg_data_organized" / "players"
    CTG_TEAM_DATA_PATH: Path = PROJECT_ROOT / "data" / "ctg_team_data"
    PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"
    RESULTS_DIR: Path = PROJECT_ROOT / "data" / "results"
    MODELS_DIR: Path = PROJECT_ROOT / "models"

    # Data filtering
    MIN_MINUTES_PER_GAME: float = 15.0
    MIN_GAMES_PLAYED: int = 10

    # CTG data parameters
    CTG_SEASONS: List[str] = field(default_factory=lambda: ["2023-24", "2024-25"])
    CTG_SEASON_TYPE: str = "regular_season"


@dataclass
class ModelConfig:
    """Model hyperparameters for all model types."""

    # XGBoost parameters (single-stage baseline)
    XGBOOST_PARAMS: Dict[str, Any] = field(
        default_factory=lambda: {
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
    )

    # LightGBM parameters
    LIGHTGBM_PARAMS: Dict[str, Any] = field(
        default_factory=lambda: {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_samples": 20,
            "num_leaves": 63,  # 2^6 - 1
            "boosting_type": "gbdt",
            "objective": "regression",
            "metric": "mae",
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }
    )

    # CatBoost parameters (for two-stage predictor)
    CATBOOST_MINUTES_PARAMS: Dict[str, Any] = field(
        default_factory=lambda: {
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
    )

    CATBOOST_PRA_PARAMS: Dict[str, Any] = field(
        default_factory=lambda: {
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
    )

    # Ensemble parameters
    ENSEMBLE_META_PARAMS: Dict[str, Any] = field(
        default_factory=lambda: {
            "alpha": 1.0,  # L2 regularization for Ridge
            "fit_intercept": True,
            "random_state": 42,
        }
    )


@dataclass
class FeatureConfig:
    """Feature engineering parameters."""

    # Lag features
    LAG_STATS: List[str] = field(default_factory=lambda: ["PRA", "MIN", "PTS", "REB", "AST"])
    LAG_VALUES: List[int] = field(default_factory=lambda: [1, 3, 5, 7])

    # Rolling window features
    ROLLING_STATS: List[str] = field(default_factory=lambda: ["PRA", "MIN", "PTS", "REB", "AST"])
    ROLLING_WINDOWS: List[int] = field(default_factory=lambda: [5, 10, 20])
    ROLLING_AGGREGATIONS: List[str] = field(default_factory=lambda: ["mean", "std"])

    # EWMA features
    EWMA_STATS: List[str] = field(default_factory=lambda: ["PRA", "MIN"])
    EWMA_SPANS: List[int] = field(default_factory=lambda: [5, 10])

    # Trend features
    TREND_STATS: List[str] = field(default_factory=lambda: ["PRA", "MIN"])
    TREND_RECENT_WINDOW: int = 5
    TREND_COMPARISON_WINDOW: int = 20


@dataclass
class ValidationConfig:
    """Walk-forward validation and backtesting parameters."""

    # Walk-forward validation
    WALK_FORWARD_MIN_TRAIN_GAMES: int = 10
    WALK_FORWARD_MIN_HISTORY: int = 5

    # Train/validation split
    TRAIN_SEASON: str = "2023-24"
    VAL_SEASON: str = "2024-25"
    TRAIN_START_DATE: str = "2023-10-01"
    TRAIN_END_DATE: str = "2024-06-30"
    VAL_START_DATE: str = "2024-10-01"
    VAL_END_DATE: str = "2025-06-30"

    # Betting simulation
    BETTING_ODDS: int = -110  # American odds (standard -110 means 52.38% breakeven)
    CONFIDENCE_THRESHOLDS: List[float] = field(default_factory=lambda: [0.55, 0.60, 0.65])
    STARTING_BANKROLL: float = 1000.0
    KELLY_FRACTION: float = 0.25  # 1/4 Kelly
    MIN_BET: float = 10.0
    MAX_BET_PCT: float = 0.05  # Max 5% of bankroll per bet

    # Calibration
    CALIBRATION_METHOD: str = "isotonic"  # "isotonic" or "platt"
    CALIBRATION_MIN_SAMPLES: int = 100


@dataclass
class MLflowConfig:
    """MLflow experiment tracking configuration."""

    # Experiment names
    EXPERIMENT_PHASE1: str = "Phase1_Foundation"
    EXPERIMENT_PHASE2: str = "Phase2_PositionDefense"
    EXPERIMENT_PHASE3: str = "Phase3_TwoStage"
    EXPERIMENT_PRODUCTION: str = "Production"

    # Model registry
    MODEL_NAME: str = "NBAPropsModel"

    # Production criteria
    MAX_VAL_MAE: float = 5.0
    MIN_BETTING_ROI: float = 0.05
    MIN_BETTING_WIN_RATE: float = 0.55


@dataclass
class BettingConfig:
    """
    Optimal betting strategy configuration.

    Based on comprehensive backtesting (October 2025), this configuration defines
    the OPTIMAL betting filters that achieve 54.78% win rate and +5.95% ROI.

    See: OPTIMAL_BETTING_STRATEGY.md for full research findings.
    """

    # Star players to EXCLUDE from betting
    # These players have less efficient markets (44.59% win rate, -11.33% ROI)
    STAR_PLAYERS: List[str] = field(
        default_factory=lambda: [
            # Superstars
            "LeBron James",
            "Stephen Curry",
            "Kevin Durant",
            "Giannis Antetokounmpo",
            "Nikola Jokic",
            "Joel Embiid",
            "Luka Doncic",
            "Jayson Tatum",
            # All-Stars
            "Damian Lillard",
            "Anthony Davis",
            "Kawhi Leonard",
            "Jimmy Butler",
            "Devin Booker",
            "Donovan Mitchell",
            "Trae Young",
            "Ja Morant",
            "Kyrie Irving",
            "Paul George",
            "Anthony Edwards",
            "Shai Gilgeous-Alexander",
            "Karl-Anthony Towns",
            "Domantas Sabonis",
            "Jaylen Brown",
            "Bam Adebayo",
            # High-usage players
            "De'Aaron Fox",
            "Tyrese Haliburton",
            "Pascal Siakam",
            "Zion Williamson",
            "Brandon Ingram",
            "DeMar DeRozan",
            "Bradley Beal",
            "Jrue Holiday",
            "Kristaps Porzingis",
            "Jaren Jackson Jr.",
            "LaMelo Ball",
            "Dejounte Murray",
            "Fred VanVleet",
            "Draymond Green",
            "Klay Thompson",
            "Jalen Brunson",
            "Julius Randle",
            "Rudy Gobert",
            "CJ McCollum",
            "Tyler Herro",
            "Tobias Harris",
            "Khris Middleton",
            "Chris Paul",
            "Russell Westbrook",
            # Rising stars
            "Paolo Banchero",
            "Scottie Barnes",
            "Lauri Markkanen",
            "Jerami Grant",
            "Cade Cunningham",
            "Deni Avdija",
            "Cole Anthony",
            "Kyle Kuzma",
            "Derrick White",
            "Jalen Williams",
            "Anfernee Simons",
            "Jordan Poole",
        ]
    )

    # Optimal edge filters (validated through backtesting)
    EDGE_MEDIUM_MIN: float = 5.0  # Medium edge minimum (pts)
    EDGE_MEDIUM_MAX: float = 7.0  # Medium edge maximum (pts)
    EDGE_HUGE_MIN: float = 10.0  # Huge edge minimum (pts)

    # Edge ranges to AVOID (unprofitable)
    EDGE_SMALL_MIN: float = 3.0  # Small edge min (avoid: 50.9% win rate)
    EDGE_SMALL_MAX: float = 5.0  # Small edge max
    EDGE_LARGE_MIN: float = 7.0  # Large edge min (avoid: 50.0% win rate)
    EDGE_LARGE_MAX: float = 10.0  # Large edge max

    # Betting strategy flags
    USE_OPTIMAL_STRATEGY: bool = True  # Use optimal filters (non-stars + edge filters)
    USE_STAR_FILTER: bool = True  # Exclude star players
    USE_EDGE_FILTER: bool = True  # Only bet on medium/huge edges

    # Expected performance (based on 2024-25 backtest)
    EXPECTED_WIN_RATE: float = 0.5478  # 54.78%
    EXPECTED_ROI: float = 0.0595  # 5.95%
    EXPECTED_BETS_PER_SEASON: int = 230
    EXPECTED_BETS_PER_DAY: float = 1.4


# Singleton instances for easy import
data_config = DataConfig()
model_config = ModelConfig()
feature_config = FeatureConfig()
validation_config = ValidationConfig()
mlflow_config = MLflowConfig()


# Helper functions for common config operations
def get_model_params(model_type: str) -> Dict[str, Any]:
    """
    Get hyperparameters for a specific model type.

    Args:
        model_type: Model type ('xgboost', 'lightgbm', 'catboost_minutes', 'catboost_pra')

    Returns:
        Dictionary of hyperparameters

    Raises:
        ValueError: If model_type is unknown
    """
    model_params_map = {
        "xgboost": model_config.XGBOOST_PARAMS,
        "lightgbm": model_config.LIGHTGBM_PARAMS,
        "catboost_minutes": model_config.CATBOOST_MINUTES_PARAMS,
        "catboost_pra": model_config.CATBOOST_PRA_PARAMS,
    }

    if model_type not in model_params_map:
        raise ValueError(
            f"Unknown model type: {model_type}. " f"Valid types: {list(model_params_map.keys())}"
        )

    return model_params_map[model_type].copy()


def ensure_directories_exist():
    """Create necessary directories if they don't exist."""
    directories = [
        data_config.PROCESSED_DIR,
        data_config.RESULTS_DIR,
        data_config.MODELS_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def validate_data_paths():
    """
    Validate that critical data paths exist.

    Raises:
        FileNotFoundError: If required data files are missing
    """
    critical_paths = {
        "Game logs": data_config.GAME_LOGS_PATH,
        "CTG data directory": data_config.CTG_DATA_PATH,
    }

    missing_paths = []
    for name, path in critical_paths.items():
        if not path.exists():
            missing_paths.append(f"{name}: {path}")

    if missing_paths:
        raise FileNotFoundError(
            f"Critical data paths missing:\n  - " + "\n  - ".join(missing_paths)
        )


if __name__ == "__main__":
    # Test configuration loading
    print("=" * 80)
    print("NBA Props Model - Configuration Test")
    print("=" * 80)

    print("\n📁 Data Paths:")
    print(f"  Game Logs: {data_config.GAME_LOGS_PATH}")
    print(f"  Results: {data_config.RESULTS_DIR}")
    print(f"  Exists: {data_config.GAME_LOGS_PATH.exists()}")

    print("\n🤖 Model Hyperparameters:")
    print(f"  XGBoost n_estimators: {model_config.XGBOOST_PARAMS['n_estimators']}")
    print(f"  XGBoost learning_rate: {model_config.XGBOOST_PARAMS['learning_rate']}")

    print("\n🔍 Feature Config:")
    print(f"  Lag values: {feature_config.LAG_VALUES}")
    print(f"  Rolling windows: {feature_config.ROLLING_WINDOWS}")

    print("\n✅ Validation Config:")
    print(f"  Min train games: {validation_config.WALK_FORWARD_MIN_TRAIN_GAMES}")
    print(f"  Betting odds: {validation_config.BETTING_ODDS}")

    print("\n📊 MLflow Config:")
    print(f"  Experiment: {mlflow_config.EXPERIMENT_PHASE1}")
    print(f"  Model name: {mlflow_config.MODEL_NAME}")

    # Ensure directories exist
    ensure_directories_exist()
    print("\n✅ All directories verified/created")

    # Test get_model_params
    xgb_params = get_model_params("xgboost")
    print(f"\n✅ Retrieved XGBoost params: {len(xgb_params)} parameters")

    print("\n" + "=" * 80)
    print("Configuration loaded successfully!")
    print("=" * 80)
