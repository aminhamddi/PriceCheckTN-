"""Production configuration for PriceCheckTN

Extends base configuration with production-specific settings.
"""

from .base import BaseConfig

class ProductionConfig(BaseConfig):
    """Production environment configuration"""

    # Debug settings (disabled for production)
    API_DEBUG: bool = False
    API_RELOAD: bool = False

    # Scraping settings (conservative for production)
    SCRAPING_RATE_LIMIT: float = 5.0
    SCRAPING_MAX_PAGES: int = 10

    # MLflow settings (remote server for production)
    MLFLOW_TRACKING_URI: str = "http://mlflow-server:5000"
    MLFLOW_EXPERIMENT_NAME: str = "pricechecktn_prod"

    # Monitoring settings (enabled for production)
    MONITORING_ENABLED: bool = True
    PROMETHEUS_PORT: int = 9090

    # Database settings (production MongoDB)
    MONGO_URI: str = "mongodb://mongo:27017"
    MONGO_DB_NAME: str = "pricechecktn_prod"

    # DVC settings (production storage)
    DVC_REMOTE_STORAGE: str = "s3://pricechecktn-data-bucket"
    DVC_DATA_VERSIONING: bool = True

    class Config:
        env_file = ".env.production"
        env_file_encoding = "utf-8"

def get_production_config() -> ProductionConfig:
    """Get production configuration"""
    return ProductionConfig()

# Initialize production configuration
prod_config = get_production_config()