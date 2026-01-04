"""
ML Model Loader and Predictor - Fixed for XGBoost
"""

import joblib
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
from loguru import logger
import re

# --- FEATURE COLUMNS USED BY THE MODEL ---
FEATURE_COLUMNS = [
    'text_length', 'word_count', 'avg_word_length',
    'exclamation_count', 'question_count', 'capital_ratio',
    'digit_count', 'unique_word_ratio', 'sentence_count',
    'sentiment_polarity', 'sentiment_subjectivity',
    'rating', 'is_extreme_rating', 'is_very_short', 'is_very_long'
]

# ---------------------------
class FakeReviewDetector:
    """Fake Review Detector Model Wrapper"""

    def __init__(self, model_path: Path, scaler_path: Path):
        """
        Load trained model and scaler

        Args:
            model_path: Path to pickled XGBoost model
            scaler_path: Path to pickled StandardScaler
        """
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = None
        self.model_version = "1.0.0"

        self.load_model()

    def load_model(self):
        """Load XGBoost model and scaler from disk"""
        try:
            logger.info(f"Loading model from: {self.model_path}")
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model not found at {self.model_path}")

            self.model = joblib.load(self.model_path)
            logger.success("✅ XGBoost model loaded successfully")

            logger.info(f"Loading scaler from: {self.scaler_path}")
            if not self.scaler_path.exists():
                raise FileNotFoundError(f"Scaler not found at {self.scaler_path}")

            self.scaler = joblib.load(self.scaler_path)
            logger.success("✅ Scaler loaded successfully")

        except Exception as e:
            logger.error(f"❌ Failed to load model/scaler: {e}")
            raise

    # ---------------------------
    def detect_language(self, text: str) -> str:
        """Simple language detection (fr vs en vs ar)"""
        arabic_chars = len(re.findall(r'[\u0600-\u06FF]', text))
        total_chars = len(text.replace(' ', ''))
        if total_chars == 0:
            return 'unknown'

        arabic_ratio = arabic_chars / total_chars
        if arabic_ratio > 0.3:
            return 'ar'

        # simple french detection
        french_words = ['le', 'la', 'les', 'de', 'du', 'des', 'un', 'une', 'très', 'est']
        text_lower = text.lower()
        french_count = sum(1 for word in french_words if word in text_lower)
        return 'fr' if french_count > 2 else 'en'

    # ---------------------------
    def extract_features(self, text: str, rating: Optional[int] = None) -> Dict:
        """Extract numeric features for ML model"""
        features = {}

        features['text_length'] = len(text)
        words = text.split()
        features['word_count'] = len(words)
        features['avg_word_length'] = features['text_length'] / max(features['word_count'], 1)

        # Uppercase ratio
        uppercase_count = sum(1 for c in text if c.isupper())
        features['capital_ratio'] = uppercase_count / max(features['text_length'], 1)

        # Punctuation
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')

        # Digits
        features['digit_count'] = sum(1 for c in text if c.isdigit())

        # Unique word ratio
        words_alpha = [w for w in words if w.isalpha()]
        features['unique_word_ratio'] = len(set(words_alpha)) / max(len(words_alpha), 1)

        # Sentence count (approx)
        features['sentence_count'] = max(text.count('.') + text.count('!') + text.count('?'), 1)

        # Sentiment (simple polarity using TextBlob)
        try:
            from textblob import TextBlob
            blob = TextBlob(text)
            features['sentiment_polarity'] = blob.sentiment.polarity
            features['sentiment_subjectivity'] = blob.sentiment.subjectivity
        except ImportError:
            features['sentiment_polarity'] = 0.0
            features['sentiment_subjectivity'] = 0.0

        # Rating features
        features['rating'] = rating if rating is not None else 3
        features['is_extreme_rating'] = 1 if rating in [1, 5] else 0
        features['is_very_short'] = 1 if features['word_count'] < 20 else 0
        features['is_very_long'] = 1 if features['word_count'] > 150 else 0

        return features

    # ---------------------------
    def predict(self, text: str, language: Optional[str] = None, rating: Optional[int] = None) -> Tuple[bool, float, float, str, Dict]:
        """Predict if review is fake"""
        try:
            if not language:
                language = self.detect_language(text)

            features = self.extract_features(text, rating)

            # Convert to array
            X = np.array([[features[col] for col in FEATURE_COLUMNS]])

            # Scale
            if self.scaler:
                X_scaled = self.scaler.transform(X)
            else:
                X_scaled = X

            # Predict
            pred = self.model.predict(X_scaled)[0]
            proba = self.model.predict_proba(X_scaled)[0]
            is_fake = bool(pred)
            fake_prob = float(proba[1])
            real_prob = float(proba[0])
            confidence = max(fake_prob, real_prob)

            return is_fake, confidence, fake_prob, language, features

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise

    # ---------------------------
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model is not None and self.scaler is not None


# ---------------------------
# Global model singleton
_model_instance: Optional[FakeReviewDetector] = None

def get_model() -> FakeReviewDetector:
    """Get or create model instance"""
    global _model_instance
    if _model_instance is None:
        from api.config import settings
        scaler_path = Path(settings.model_path.parent) / 'feature_scaler.pkl'
        _model_instance = FakeReviewDetector(settings.model_path, scaler_path)
    return _model_instance
