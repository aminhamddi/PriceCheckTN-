"""NLP module for PriceCheck TN"""

# Only import what exists
__all__ = ["preprocess_text", "extract_features", "BERTFakeReviewPredictor", "EnsemblePredictor"]

# Try to import safely
try:
    from .preprocessing import preprocess_text
except ImportError:
    preprocess_text = None

try:
    from .feature_engineering import extract_features
except ImportError:
    extract_features = None

try:
    from .models.bert_predictor import BERTFakeReviewPredictor, EnsemblePredictor
except ImportError:
    BERTFakeReviewPredictor = None
    EnsemblePredictor = None