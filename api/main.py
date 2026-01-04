"""
FastAPI with BERT Model
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from api.models.bert_predictor import BERTFakeReviewPredictor, EnsemblePredictor
from loguru import logger
import sys
from pathlib import Path

# Setup logging
logger.remove()
logger.add(sys.stdout, level="INFO")

# Create app
app = FastAPI(
    title="PriceCheck TN - Fake Review Detection API",
    description="Detect fake reviews using BERT + ML",
    version="2.0.0"
)

# Global predictor (loaded once at startup)
predictor = None


@app.on_event("startup")
async def load_model():
    """Load model on startup"""
    global predictor

    logger.info("🚀 Starting API...")

    # Check if BERT model exists
    bert_model_path = Path("models/saved_models/bert_fake_review/final_model")
    sklearn_model_path = Path("models/saved_models/sklearn_baseline/model.pkl")

    if bert_model_path.exists():
        logger.info("📦 Loading BERT model...")

        if sklearn_model_path.exists():
            # Load ensemble
            predictor = EnsemblePredictor(
                bert_path=str(bert_model_path),
                sklearn_path=str(sklearn_model_path)
            )
            logger.success("✅ Ensemble predictor loaded (BERT + Sklearn)")
        else:
            # BERT only
            predictor = BERTFakeReviewPredictor(model_path=str(bert_model_path))
            logger.success("✅ BERT predictor loaded")
    else:
        logger.error("❌ BERT model not found!")
        logger.info("💡 Place model in: models/saved_models/bert_fake_review/final_model")
        raise FileNotFoundError("BERT model not found")


# Request/Response models
class ReviewRequest(BaseModel):
    text: str
    method: str = "bert_only"  # 'bert_only', 'sklearn_only', or 'ensemble'

    class Config:
        schema_extra = {
            "example": {
                "text": "This product is amazing! Best purchase ever!!!",
                "method": "bert_only"
            }
        }


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: dict
    model: str
    details: dict = None


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "PriceCheck TN - Fake Review Detection API",
        "version": "2.0.0",
        "model": "BERT Fine-tuned",
        "docs": "/docs"
    }


@app.get("/health")
async def health():
    """Health check"""
    return {
        "status": "healthy",
        "model_loaded": predictor is not None,
        "model_type": type(predictor).__name__
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: ReviewRequest):
    """
    Predict if a review is fake or real

    - **text**: Review text to analyze
    - **method**: Prediction method ('bert_only', 'sklearn_only', 'ensemble')
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not request.text or len(request.text.strip()) < 10:
        raise HTTPException(status_code=400, detail="Review text too short (min 10 chars)")

    try:
        logger.info(f"🔍 Analyzing review (method: {request.method})")

        # Check if ensemble or single model
        if isinstance(predictor, EnsemblePredictor):
            result = predictor.predict(request.text, method=request.method)
        else:
            result = predictor.predict_single(request.text)

        logger.info(f"🎯 Prediction: {result['prediction']} (confidence: {result['confidence']:.2f}%)")

        return result

    except Exception as e:
        logger.error(f"❌ Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
async def predict_batch(texts: list[str]):
    """
    Predict on multiple reviews at once
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not texts:
        raise HTTPException(status_code=400, detail="No texts provided")

    try:
        logger.info(f"🔍 Analyzing {len(texts)} reviews")

        # Use BERT batch prediction
        if isinstance(predictor, EnsemblePredictor):
            results = predictor.bert_predictor.predict_batch(texts)
        else:
            results = predictor.predict_batch(texts)

        return {
            "count": len(results),
            "predictions": results
        }

    except Exception as e:
        logger.error(f"❌ Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/info")
async def model_info():
    """Get model information"""
    if predictor is None:
        return {"model_loaded": False}

    return {
        "model_loaded": True,
        "model_type": type(predictor).__name__,
        "device": str(predictor.bert_predictor.device) if hasattr(predictor, 'bert_predictor') else str(predictor.device),
        "model_name": "DistilBERT Multilingual",
        "num_labels": 2,
        "labels": {"0": "REAL", "1": "FAKE"}
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)