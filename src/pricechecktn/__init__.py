"""PriceCheckTN - Price Comparison and Fake Review Detection System

A comprehensive MLOps-ready system for comparing tech product prices
between Tunisia and France, with advanced fake review detection.
"""

__version__ = "2.0.0"
__author__ = "PriceCheckTN Team"
__license__ = "MIT"
__description__ = "MLOps-ready price comparison and review analysis system"

# Package imports - import modules only (not specific classes that don't exist yet)
from . import data, features, models, pipelines, scraping, utils, api

# Main exports - only export what we actually have
from .utils.logging import setup_logging

# Setup package-level logging
setup_logging()

__all__ = [
    "setup_logging"
]

