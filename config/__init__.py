"""Configuration package for PriceCheckTN

Centralized configuration management for different environments.
"""

from .base import BaseConfig, get_config
from .development import DevelopmentConfig, get_development_config
from .production import ProductionConfig, get_production_config

# Lazy load config to avoid import issues
def __getattr__(name):
    if name == "config":
        return get_config()
    elif name == "dev_config":
        return get_development_config()
    elif name == "prod_config":
        return get_production_config()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    "BaseConfig", "config", "get_config",
    "DevelopmentConfig", "dev_config", "get_development_config",
    "ProductionConfig", "prod_config", "get_production_config"
]