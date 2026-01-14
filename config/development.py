"""Development configuration for PriceCheckTN

Extends base configuration with development-specific settings.
"""

from .base import BaseConfig
try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings

class DevelopmentConfig(BaseConfig):
    """Development environment configuration"""

    # Debug settings
    API_DEBUG: bool = True
    API_RELOAD: bool = True

    # Scraping settings (more aggressive for development)
    SCRAPING_RATE_LIMIT: float = 1.0
    SCRAPING_MAX_PAGES: int = 3

    # MLflow settings (local for development)
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"
    MLFLOW_EXPERIMENT_NAME: str = "pricechecktn_dev"

    # Monitoring settings
    MONITORING_ENABLED: bool = True

    # Database settings (local MongoDB)
    MONGO_URI: str = "mongodb://localhost:27017"
    MONGO_DB_NAME: str = "pricechecktn_dev"

    class Config:
        env_file = ".env.development"
        env_file_encoding = "utf-8"

def get_development_config() -> DevelopmentConfig:
    """Get development configuration"""
    return DevelopmentConfig()

# Initialize development configuration (lazy loading)
dev_config = None