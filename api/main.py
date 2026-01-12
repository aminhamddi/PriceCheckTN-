"""\
FastAPI with BERT Model - Local Models Only (No MLflow)\
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import importlib.util
import sys
from pathlib import Path
import logging
from typing import Optional

# Dynamic import of api/models/bert_predictor.py to avoid name collision
# when an `api.models` module file exists alongside the `api/models/` package.
_bp = Path(__file__).parent / "models" / "bert_predictor.py"
_spec = importlib.util.spec_from_file_location("api_models_bert_predictor", _bp)
_bert_mod = importlib.util.module_from_spec(_spec)
sys.modules["api_models_bert_predictor"] = _bert_mod
_spec.loader.exec_module(_bert_mod)
BERTFakeReviewPredictor = _bert_mod.BERTFakeReviewPredictor
EnsemblePredictor = _bert_mod.EnsemblePredictor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



# Create app
app = FastAPI(
    title="PriceCheck TN - Fake Review Detection API",
    description="Detect fake reviews using BERT + ML with Model Registry",
    version="2.0.0"
)

# Global predictor (loaded once at startup)
predictor = None
model_metadata = None

@app.on_event("startup")
async def load_model():
    """Load model from local files on startup"""
    global predictor, model_metadata

    logger.info("🚀 Starting API with local models...")

    try:
        await load_local_model()
        model_metadata = None  # No registry metadata for local models
    except Exception as e:
        logger.error(f"❌ Failed to load local models: {e}")
        logger.error("💡 Make sure your models are in the correct directories:")
        logger.error("   - BERT: models/saved_models/bert_fake_review/final_model/")
        logger.error("   - XGBoost: models/fake_review_detector_xgboost.pkl")
        raise

async def load_local_model():
    """Load model from local files"""
    global predictor

    # Check if BERT model exists and contains model weights
    bert_model_path = Path("models/saved_models/bert_fake_review/final_model")
    xgboost_model_path = Path("models/fake_review_detector_xgboost.pkl")
    scaler_path = Path("models/feature_scaler.pkl")
    feature_names_path = Path("models/feature_names.txt")

    # Validate BERT model directory
    if bert_model_path.exists():
        # Look for common transformer weight files
        weight_files = [
            "pytorch_model.bin",
            "tf_model.h5",
            "model.safetensors",
            "flax_model.msgpack",
            "model.ckpt.index"
        ]
        has_weights = any((bert_model_path / wf).exists() for wf in weight_files)

        if not has_weights:
            logger.error("❌ BERT model directory found but no model weight files present")
            logger.info("💡 Expected one of: %s", weight_files)
            logger.info("💡 Place model weights in: models/saved_models/bert_fake_review/final_model")

            # Optional: download from HuggingFace if HF_MODEL_ID env var is provided
            import os
            hf_id = os.getenv("HF_MODEL_ID")
            if hf_id:
                try:
                    logger.info(f"⬇️ HF_MODEL_ID provided, attempting to download '{hf_id}' into {bert_model_path}")
                    from transformers import AutoModelForSequenceClassification, AutoTokenizer
                    bert_model_path.mkdir(parents=True, exist_ok=True)
                    model = AutoModelForSequenceClassification.from_pretrained(hf_id)
                    tokenizer = AutoTokenizer.from_pretrained(hf_id)
                    model.save_pretrained(bert_model_path)
                    tokenizer.save_pretrained(bert_model_path)
                    logger.success(f"✅ Downloaded and saved HF model '{hf_id}' to {bert_model_path}")
                    has_weights = True
                except Exception as e:
                    logger.error(f"❌ Failed to download HF model '{hf_id}': {e}")

            if not has_weights:
                raise FileNotFoundError("BERT model weights not found in models/saved_models/bert_fake_review/final_model")

        logger.info("📦 Loading models from local files...")

        # Check if XGBoost model exists for ensemble
        if xgboost_model_path.exists():
            # Load ensemble
            predictor = EnsemblePredictor(
                bert_path=str(bert_model_path),
                xgboost_path=str(xgboost_model_path),
                scaler_path=str(scaler_path) if scaler_path.exists() else None,
                feature_names_path=str(feature_names_path) if feature_names_path.exists() else None
            )
            logger.info("✅ Ensemble predictor loaded (BERT + XGBoost) from local files")
        else:
            # BERT only
            predictor = BERTFakeReviewPredictor(model_path=str(bert_model_path))
            logger.info("✅ BERT predictor loaded from local files")
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

class ModelInfoResponse(BaseModel):
    model_name: str
    version: str
    stage: str
    registry_enabled: bool
    fallback_mode: bool

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "PriceCheck TN - Fake Review Detection API",
        "version": "2.0.0",
        "model": "BERT Fine-tuned with MLflow Model Registry",
        "docs": "/docs"
    }

@app.get("/health")
async def health():
    """Health check"""
    return {
        "status": "healthy",
        "model_loaded": predictor is not None,
        "model_type": type(predictor).__name__ if predictor else None,
        "registry_integration": model_metadata is not None
    }

@app.get("/model/info")
async def model_info():
    """Get model information with registry details"""
    if predictor is None:
        return {"model_loaded": False}

    if model_metadata:
        return ModelInfoResponse(
            model_name=model_metadata["name"],
            version=str(model_metadata["version"]),
            stage=model_metadata["stage"],
            registry_enabled=True,
            fallback_mode=False
        )
    else:
        return ModelInfoResponse(
            model_name="bert_fake_review_detector",
            version="local",
            stage="local",
            registry_enabled=False,
            fallback_mode=True
        )

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

        # Add registry info to response
        if model_metadata:
            result["registry_info"] = {
                "model_version": model_metadata["version"],
                "stage": model_metadata["stage"],
                "run_id": model_metadata["run_id"]
            }

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
            "predictions": results,
            "model_info": {
                "registry_enabled": model_metadata is not None,
                "model_version": model_metadata["version"] if model_metadata else "local"
            }
        }

    except Exception as e:
        logger.error(f"❌ Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
