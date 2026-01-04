"""
FastAPI Main Application
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from api.config import settings
from api.models import (
    ReviewInput,
    PredictionResponse,
    HealthResponse,
    ErrorResponse
)
from api.ml_model import get_model
from loguru import logger
import sys
from datetime import datetime

# Configure logging
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level:8}</level> | <level>{message}</level>",
    level="INFO"
)
logger.add(
    "logs/api.log",
    rotation="10 MB",
    retention="7 days",
    level="DEBUG"
)

# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description=settings.description,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    logger.info("🚀 Starting FastAPI application")
    logger.info(f"📋 Version: {settings.version}")

    try:
        model = get_model()
        logger.success("✅ Model loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("👋 Shutting down FastAPI application")


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "PriceCheck TN - Fake Review Detector API",
        "version": settings.version,
        "docs": "/docs",
        "health": "/health"
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"]
)
async def health_check():
    """
    Health check endpoint

    Returns API status and model availability
    """
    try:
        model = get_model()
        model_loaded = model.is_loaded()

        return HealthResponse(
            status="healthy" if model_loaded else "degraded",
            model_loaded=model_loaded,
            version=settings.version
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            version=settings.version
        )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        200: {"description": "Successful prediction"},
        400: {"model": ErrorResponse, "description": "Bad request"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    },
    tags=["Prediction"]
)
async def predict_fake_review(review: ReviewInput):
    """
    Predict if a review is fake

    Analyzes the review text using ML model and returns prediction with confidence score.

    - **text**: Review text to analyze (10-5000 characters)
    - **language**: Optional language code (fr, ar, en). Auto-detected if not provided
    - **rating**: Optional review rating (1-5 stars)
    - **product_id**: Optional product identifier for tracking
    """
    try:
        logger.info(f"📝 Received prediction request (length: {len(review.text)} chars)")

        # Get model
        model = get_model()

        # Make prediction
        is_fake, confidence, fake_prob, detected_lang, features = model.predict(
            text=review.text,
            language=review.language,
            rating=review.rating
        )

        # Log result
        result = "FAKE" if is_fake else "REAL"
        logger.info(f"🎯 Prediction: {result} (confidence: {confidence:.2%})")

        # Build response
        response = PredictionResponse(
            is_fake=is_fake,
            confidence=confidence,
            fake_probability=fake_prob,
            real_probability=1.0 - fake_prob,
            language_detected=detected_lang,
            features=features if settings.debug else None,
            model_version=model.model_version
        )

        return response

    except ValueError as e:
        logger.warning(f"⚠️  Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

    except Exception as e:
        logger.error(f"❌ Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during prediction"
        )


@app.post(
    "/predict/batch",
    response_model=list[PredictionResponse],
    tags=["Prediction"]
)
async def predict_batch(reviews: list[ReviewInput]):
    """
    Batch prediction for multiple reviews

    Analyzes multiple reviews in a single request.
    Limited to 100 reviews per request.
    """
    if len(reviews) > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum 100 reviews per batch request"
        )

    logger.info(f"📦 Batch prediction request ({len(reviews)} reviews)")

    results = []
    model = get_model()

    for idx, review in enumerate(reviews):
        try:
            is_fake, confidence, fake_prob, detected_lang, features = model.predict(
                text=review.text,
                language=review.language,
                rating=review.rating
            )

            results.append(PredictionResponse(
                is_fake=is_fake,
                confidence=confidence,
                fake_probability=fake_prob,
                real_probability=1.0 - fake_prob,
                language_detected=detected_lang,
                model_version=model.model_version
            ))

        except Exception as e:
            logger.error(f"Error processing review {idx}: {e}")
            # Continue with other reviews
            continue

    logger.success(f"✅ Batch complete: {len(results)}/{len(reviews)} successful")
    return results


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if settings.debug else "An error occurred",
            "timestamp": datetime.now().isoformat()
        }
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug
    )