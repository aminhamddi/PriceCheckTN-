"""
Pydantic Models for Request/Response validation
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any
from datetime import datetime


class ReviewInput(BaseModel):
    """Input model for review analysis"""

    text: str = Field(
        ...,
        min_length=10,
        max_length=5000,
        description="Review text to analyze",
        example="Excellent produit, très satisfait de mon achat!"
    )

    language: Optional[str] = Field(
        None,
        description="Language code (fr, ar, en). Auto-detected if not provided",
        example="fr"
    )

    rating: Optional[int] = Field(
        None,
        ge=1,
        le=5,
        description="Review rating (1-5 stars)",
        example=5
    )

    product_id: Optional[str] = Field(
        None,
        description="Product identifier",
        example="PROD_12345"
    )

    @validator('text')
    def text_not_empty(cls, v):
        """Validate text is not just whitespace"""
        if not v.strip():
            raise ValueError("Review text cannot be empty")
        return v.strip()


class PredictionResponse(BaseModel):
    """Response model for prediction"""

    is_fake: bool = Field(
        ...,
        description="True if review is predicted as fake"
    )

    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model confidence score (0-1)"
    )

    fake_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probability of being fake (0-1)"
    )

    real_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probability of being real (0-1)"
    )

    language_detected: Optional[str] = Field(
        None,
        description="Detected language"
    )

    features: Optional[Dict[str, Any]] = Field(
        None,
        description="Extracted features (for debugging)"
    )

    model_version: str = Field(
        ...,
        description="Model version used"
    )

    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Prediction timestamp"
    )


class HealthResponse(BaseModel):
    """Health check response"""

    status: str = Field(..., example="healthy")
    model_loaded: bool
    version: str
    timestamp: datetime = Field(default_factory=datetime.now)


class ErrorResponse(BaseModel):
    """Error response"""

    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error info")
    timestamp: datetime = Field(default_factory=datetime.now)