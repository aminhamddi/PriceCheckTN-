"""
BERT Predictor for FastAPI
Optimized for production inference
"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from pathlib import Path
from loguru import logger
import numpy as np
from typing import Dict, List


class BERTFakeReviewPredictor:
    """
    Production-ready BERT predictor
    """

    def __init__(self, model_path: str, device: str = None):
        """
        Args:
            model_path: Path to saved BERT model
            device: 'cuda', 'cpu', or None (auto-detect)
        """
        self.model_path = Path(model_path)

        # Auto-detect device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        logger.info(f"🤖 Loading BERT model from: {model_path}")
        logger.info(f"💻 Using device: {self.device}")

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)

        # Move to device
        self.model.to(self.device)

        # Set to evaluation mode
        self.model.eval()

        # Label mapping
        self.labels = {0: 'REAL', 1: 'FAKE'}

        logger.success("✅ BERT model loaded successfully")

    def predict_single(self, text: str) -> Dict:
        """
        Predict on a single review

        Args:
            text: Review text

        Returns:
            {
                'prediction': 'FAKE' or 'REAL',
                'confidence': float (0-100),
                'probabilities': {'FAKE': float, 'REAL': float}
            }
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            padding=True,
            max_length=512
        )

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)[0]
            predicted_class = torch.argmax(probabilities).item()

        # Get probabilities
        probs = probabilities.cpu().numpy()

        result = {
            'prediction': self.labels[predicted_class],
            'confidence': float(probs[predicted_class] * 100),
            'probabilities': {
                'REAL': float(probs[0] * 100),
                'FAKE': float(probs[1] * 100)
            },
            'model': 'BERT'
        }

        return result

    def predict_batch(self, texts: List[str], batch_size: int = 32) -> List[Dict]:
        """
        Predict on multiple reviews (batched for efficiency)

        Args:
            texts: List of review texts
            batch_size: Batch size for inference

        Returns:
            List of prediction dictionaries
        """
        results = []

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            # Tokenize batch
            inputs = self.tokenizer(
                batch_texts,
                return_tensors='pt',
                truncation=True,
                padding=True,
                max_length=512
            )

            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=1)
                predictions = torch.argmax(probabilities, dim=1)

            # Convert to results
            probs_np = probabilities.cpu().numpy()
            preds_np = predictions.cpu().numpy()

            for j in range(len(batch_texts)):
                result = {
                    'prediction': self.labels[preds_np[j]],
                    'confidence': float(probs_np[j][preds_np[j]] * 100),
                    'probabilities': {
                        'REAL': float(probs_np[j][0] * 100),
                        'FAKE': float(probs_np[j][1] * 100)
                    },
                    'model': 'BERT'
                }
                results.append(result)

        return results


class EnsemblePredictor:
    """
    Ensemble of BERT + Sklearn models
    For maximum accuracy
    """

    def __init__(self, bert_path: str, sklearn_path: str = None):
        """
        Args:
            bert_path: Path to BERT model
            sklearn_path: Path to sklearn model (optional)
        """
        # Load BERT
        self.bert_predictor = BERTFakeReviewPredictor(bert_path)

        # Load sklearn if provided
        self.sklearn_predictor = None
        if sklearn_path:
            import joblib
            self.sklearn_predictor = joblib.load(sklearn_path)
            logger.info("✓ Loaded sklearn model for ensemble")

    def predict(self, text: str, method: str = 'bert_only') -> Dict:
        """
        Predict with ensemble

        Args:
            text: Review text
            method: 'bert_only', 'sklearn_only', or 'ensemble'

        Returns:
            Prediction dictionary
        """
        if method == 'bert_only' or self.sklearn_predictor is None:
            return self.bert_predictor.predict_single(text)

        elif method == 'sklearn_only':
            # Use sklearn
            sklearn_pred = self.sklearn_predictor.predict([text])[0]
            sklearn_proba = self.sklearn_predictor.predict_proba([text])[0]

            return {
                'prediction': 'FAKE' if sklearn_pred == 1 else 'REAL',
                'confidence': float(max(sklearn_proba) * 100),
                'probabilities': {
                    'REAL': float(sklearn_proba[0] * 100),
                    'FAKE': float(sklearn_proba[1] * 100)
                },
                'model': 'Sklearn'
            }

        elif method == 'ensemble':
            # Get both predictions
            bert_result = self.bert_predictor.predict_single(text)

            sklearn_pred = self.sklearn_predictor.predict([text])[0]
            sklearn_proba = self.sklearn_predictor.predict_proba([text])[0]

            # Average probabilities
            avg_fake_prob = (bert_result['probabilities']['FAKE'] + sklearn_proba[1] * 100) / 2
            avg_real_prob = (bert_result['probabilities']['REAL'] + sklearn_proba[0] * 100) / 2

            final_pred = 'FAKE' if avg_fake_prob > avg_real_prob else 'REAL'

            return {
                'prediction': final_pred,
                'confidence': max(avg_fake_prob, avg_real_prob),
                'probabilities': {
                    'REAL': avg_real_prob,
                    'FAKE': avg_fake_prob
                },
                'model': 'Ensemble',
                'details': {
                    'bert': bert_result,
                    'sklearn': {
                        'prediction': 'FAKE' if sklearn_pred == 1 else 'REAL',
                        'probabilities': {
                            'REAL': float(sklearn_proba[0] * 100),
                            'FAKE': float(sklearn_proba[1] * 100)
                        }
                    }
                }
            }