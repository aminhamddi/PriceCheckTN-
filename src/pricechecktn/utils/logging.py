"""Logging utilities for PriceCheckTN

Centralized logging configuration and utilities.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from loguru import logger
from config.base import config

def setup_logging(
    log_file: Optional[str] = None,
    level: str = "INFO",
    rotation: str = "1 day",
    retention: str = "7 days"
) -> None:
    """Setup centralized logging configuration

    Args:
        log_file: Optional log file path
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        rotation: Log rotation frequency
        retention: Log retention period
    """
    # Remove default logger configuration
    logger.remove()

    # Add console handler
    logger.add(
        sys.stdout,
        level=level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
               "<level>{message}</level>",
        colorize=True
    )

    # Add file handler if log_file is provided
    if log_file:
        log_path = config.LOGS_DIR / log_file
        log_path.parent.mkdir(parents=True, exist_ok=True)

        logger.add(
            str(log_path),
            level=level,
            rotation=rotation,
            retention=retention,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            enqueue=True
        )

    logger.info("🚀 Logging configured successfully")
    logger.info(f"📁 Log directory: {config.LOGS_DIR}")
    if log_file:
        logger.info(f"📄 Log file: {log_file}")

def get_logger(name: str = __name__) -> logger:
    """Get a configured logger instance

    Args:
        name: Logger name (usually __name__)

    Returns:
        Configured logger instance
    """
    return logger.bind(name=name)

def setup_mlflow_logging() -> None:
    """Setup MLflow-specific logging"""
    import mlflow

    # Configure MLflow logging
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)
    logger.info(f"🔬 MLflow experiment: {config.MLFLOW_EXPERIMENT_NAME}")
    logger.info(f"🌐 MLflow tracking URI: {config.MLFLOW_TRACKING_URI}")

def log_system_info() -> None:
    """Log system and environment information"""
    import platform
    import psutil
    import torch

    logger.info("💻 System Information:")
    logger.info(f"  Platform: {platform.platform()}")
    logger.info(f"  Python: {platform.python_version()}")
    logger.info(f"  CPU Cores: {psutil.cpu_count()}")

    if torch.cuda.is_available():
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"  CUDA: {torch.version.cuda}")
    else:
        logger.info("  GPU: Not available")

# Initialize logging on import
setup_logging("pricechecktn.log")