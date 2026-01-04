"""
API Configuration
"""

from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    """API Settings"""

    # API Info
    app_name: str = "PriceCheck TN - Fake Review Detector API"
    version: str = "1.0.0"
    description: str = "ML-powered fake review detection for e-commerce"

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False

    # CORS
    cors_origins: list = [
        "http://localhost:3000",  # React dev
        "http://localhost:5173",  # Vite dev
        "http://localhost:4200",  # Angular dev
    ]

    # Model
    model_path: Path = Path(__file__).parent.parent / "models" / "fake_review_detector_xgboost.pkl"

    # Limits
    max_review_length: int = 5000
    rate_limit: int = 100  # requests per minute

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()