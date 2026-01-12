"""Base configuration for PriceCheckTN

Contains default settings and environment variables.
"""

import os
from pathlib import Path
from typing import Dict, Any
from pydantic_settings import BaseSettings
from pydantic import Field, ConfigDict

class BaseConfig(BaseSettings):
    """Base configuration with default values"""

    model_config = ConfigDict(
        extra='ignore',
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )

    # Project metadata
    PROJECT_NAME: str = "PriceCheckTN"
    PROJECT_VERSION: str = "2.0.0"
    PROJECT_DESCRIPTION: str = "Price Comparison and Fake Review Detection System"

    # Paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    MODELS_DIR: Path = PROJECT_ROOT / "models"
    LOGS_DIR: Path = PROJECT_ROOT / "logs"
    CONFIG_DIR: Path = PROJECT_ROOT / "config"

    # Data paths
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
    FINAL_DATA_DIR: Path = DATA_DIR / "final"

    # Model paths
    BERT_MODEL_PATH: Path = MODELS_DIR / "saved_models" / "bert_fake_review" / "final_model"
    XGBOOST_MODEL_PATH: Path = MODELS_DIR / "fake_review_detector_xgboost.pkl"
    FEATURE_SCALER_PATH: Path = MODELS_DIR / "feature_scaler.pkl"
    FEATURE_NAMES_PATH: Path = MODELS_DIR / "feature_names.txt"

    # API settings (from .env file)
    APP_NAME: str = "PriceCheckTN API"
    VERSION: str = "2.0.0"
    DEBUG: bool = True
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    CORS_ORIGINS: str = "http://localhost:3000,http://localhost:5173"
    MODEL_PATH: str = "models/fake_review_detector_xgboost.pkl"

    # Additional API settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_DEBUG: bool = True
    API_RELOAD: bool = True

    # Scraping settings
    SCRAPING_RATE_LIMIT: float = 2.0
    SCRAPING_USER_AGENT: str = "PriceCheckTN-Bot/2.0 (MLOps Research Project)"
    SCRAPING_MAX_PAGES: int = 5

    # MLflow settings
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"
    MLFLOW_EXPERIMENT_NAME: str = "pricechecktn_fake_review_detection"
    MLFLOW_MODEL_REGISTRY: str = "bert_fake_review"

    # DVC settings
    DVC_REMOTE_STORAGE: str = "gdrive://1-zosSFa_v8f2mpTMTQTWy-dlnTjJZFA5"
    DVC_DATA_VERSIONING: bool = True

    # Monitoring settings
    MONITORING_ENABLED: bool = False
    PROMETHEUS_PORT: int = 9090

    # Database settings
    MONGO_URI: str = "mongodb://localhost:27017"
    MONGO_DB_NAME: str = "pricechecktn"

    # Currency conversion
    CURRENCY_API: str = "https://api.exchangerate.host/latest"
    DEFAULT_CURRENCY: str = "EUR"

def get_config() -> BaseConfig:
    """Get the current configuration"""
    return BaseConfig()

# Initialize configuration
config = get_config()

# Create directories if they don't exist
def initialize_directories():
    """Create necessary directories"""
    directories = [
        config.DATA_DIR,
        config.RAW_DATA_DIR,
        config.PROCESSED_DATA_DIR,
        config.FINAL_DATA_DIR,
        config.MODELS_DIR,
        config.LOGS_DIR
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

# Initialize directories on import
initialize_directories()# Initialize directories on import
