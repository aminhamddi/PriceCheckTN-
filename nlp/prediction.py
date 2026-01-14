"""
Script 3: Prédictions Fake Reviews
----------------------------------
Objectif: Appliquer modèles ML (XGBoost + BERT) sur dataset reviews
         pour détecter les fake reviews

Author: PriceCheck TN Team
Date: 2026-01-04
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List
from loguru import logger
import sys
import joblib
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

# Add project root
sys.path.append(str(Path(__file__).parent.parent))

# Configure logger
logger.add(
    "logs/predictions_{time}.log",
    rotation="1 day",
    retention="7 days",
    level="INFO"
)


class FakeReviewPredictor:
    """Apply ML models to detect fake reviews"""

    def __init__(self):
        self.processed_dir = Path("data/processed")
        self.final_dir = Path("data/final")
        self.models_dir = Path("models")

        self.final_dir.mkdir(exist_ok=True, parents=True)

        # Models
        self.xgboost_model = None
        self.feature_scaler = None
        self.feature_names = None
        self.bert_model = None
        self.bert_tokenizer = None

        # Statistics
        self.stats = {
            'total_reviews': 0,
            'fake_detected': 0,
            'real_detected': 0,
            'high_confidence': 0,
            'medium_confidence': 0,
            'low_confidence': 0,
            'models_agree': 0,
            'models_disagree': 0
        }

        logger.info(" FakeReviewPredictor initialized")

    def load_models(self):
        """Load trained ML models"""
        logger.info("\n Loading models...")

        # Load XGBoost
        xgboost_path = self.models_dir / "fake_review_detector_xgboost.pkl"
        if xgboost_path.exists():
            self.xgboost_model = joblib.load(xgboost_path)
            logger.info(" Loaded XGBoost model")
        else:
            logger.warning(f"  XGBoost model not found: {xgboost_path}")

        # Load Scaler
        scaler_path = self.models_dir / "feature_scaler.pkl"
        if scaler_path.exists():
            self.feature_scaler = joblib.load(scaler_path)
            logger.info(" Loaded feature scaler")
        else:
            logger.warning(f"  Scaler not found: {scaler_path}")

        # Load Feature Names
        feature_names_path = self.models_dir / "feature_names.txt"
        if feature_names_path.exists():
            with open(feature_names_path, 'r') as f:
                self.feature_names = [line.strip() for line in f.readlines()]
            logger.info(f" Loaded {len(self.feature_names)} feature names")
        else:
            logger.warning(f"  Feature names not found: {feature_names_path}")

        # Load BERT - try multiple possible paths
        bert_paths = [
            self.models_dir / "bert",
            self.models_dir / "saved_models" / "bert_fake_review",
            Path("models/bert"),
            Path("models/saved_models/bert_fake_review")
        ]
        
        self.bert_model = None
        self.bert_tokenizer = None
        
        for bert_path in bert_paths:
            if bert_path.exists():
                try:
                    self.bert_tokenizer = AutoTokenizer.from_pretrained(bert_path)
                    self.bert_model = AutoModelForSequenceClassification.from_pretrained(bert_path)
                    self.bert_model.eval()

                    # Move to GPU if available
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    self.bert_model.to(device)

                    logger.info(f" Loaded BERT model from {bert_path} (device: {device})")
                    break
                except Exception as e:
                    logger.warning(f"  Failed to load BERT from {bert_path}: {e}")
                    continue
        
        if self.bert_model is None:
            logger.warning("  BERT model not found in any expected location")

    def load_reviews(self) -> pd.DataFrame:
        """Load processed reviews"""
        logger.info("\n Loading reviews...")

        reviews_path = self.processed_dir / "reviews_with_features.csv"

        if not reviews_path.exists():
            logger.error(f" Reviews file not found: {reviews_path}")
            raise FileNotFoundError(f"Missing: {reviews_path}")

        df = pd.read_csv(reviews_path, encoding='utf-8')
        logger.info(f" Loaded {len(df)} reviews")

        self.stats['total_reviews'] = len(df)
        return df

    def predict_xgboost(self, df: pd.DataFrame) -> np.ndarray:
        """Predict with XGBoost model"""
        logger.info("\n Running XGBoost predictions...")

        if self.xgboost_model is None:
            logger.error(" XGBoost model not loaded!")
            return np.zeros(len(df))

        # Extract features that match training
        try:
            # Get feature columns (exclude text and target columns)
            feature_cols = [col for col in self.feature_names
                          if col in df.columns and col not in ['text', 'label', 'is_fake']]

            X = df[feature_cols].values

            # Scale features
            if self.feature_scaler:
                X_scaled = self.feature_scaler.transform(X)
            else:
                X_scaled = X

            # Predict probabilities
            probas = self.xgboost_model.predict_proba(X_scaled)
            fake_scores = probas[:, 1]  # Probability of being fake

            logger.info(f" XGBoost predictions complete")
            logger.info(f"   Mean fake score: {fake_scores.mean():.3f}")

            return fake_scores

        except Exception as e:
            logger.error(f" XGBoost prediction failed: {e}")
            return np.zeros(len(df))

    def predict_bert(self, df: pd.DataFrame) -> np.ndarray:
        """Predict with BERT model"""
        logger.info("\n Running BERT predictions...")

        if self.bert_model is None:
            logger.error(" BERT model not loaded!")
            return np.zeros(len(df))

        device = next(self.bert_model.parameters()).device
        fake_scores = []

        # Process in batches
        batch_size = 16
        texts = df['text'].fillna('').tolist()

        for i in tqdm(range(0, len(texts), batch_size), desc="BERT inference"):
            batch_texts = texts[i:i+batch_size]

            try:
                # Tokenize
                inputs = self.bert_tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                )

                # Move to device
                inputs = {k: v.to(device) for k, v in inputs.items()}

                # Predict
                with torch.no_grad():
                    outputs = self.bert_model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=1)
                    batch_fake_scores = probs[:, 1].cpu().numpy()

                fake_scores.extend(batch_fake_scores)

            except Exception as e:
                logger.error(f" Batch {i} failed: {e}")
                fake_scores.extend([0.5] * len(batch_texts))

        fake_scores = np.array(fake_scores)

        logger.info(f" BERT predictions complete")
        logger.info(f"   Mean fake score: {fake_scores.mean():.3f}")

        return fake_scores

    def ensemble_predictions(self, xgb_scores: np.ndarray,
                           bert_scores: np.ndarray) -> tuple:
        """Combine XGBoost and BERT predictions"""
        logger.info("\n Creating ensemble predictions...")

        # Weighted average (you can adjust weights based on model performance)
        weights = {
            'xgboost': 0.45,
            'bert': 0.55
        }

        ensemble_scores = (
            weights['xgboost'] * xgb_scores +
            weights['bert'] * bert_scores
        )

        # Determine confidence based on agreement
        agreement = np.abs(xgb_scores - bert_scores)

        confidence = np.where(
            agreement < 0.1, 'high',
            np.where(agreement < 0.3, 'medium', 'low')
        )

        # Final classification
        is_fake = ensemble_scores > 0.5

        # Update stats
        self.stats['models_agree'] = np.sum(agreement < 0.1)
        self.stats['models_disagree'] = np.sum(agreement >= 0.3)
        self.stats['high_confidence'] = np.sum(confidence == 'high')
        self.stats['medium_confidence'] = np.sum(confidence == 'medium')
        self.stats['low_confidence'] = np.sum(confidence == 'low')
        self.stats['fake_detected'] = np.sum(is_fake)
        self.stats['real_detected'] = np.sum(~is_fake)

        logger.info(f" Ensemble complete")
        logger.info(f"   Mean ensemble score: {ensemble_scores.mean():.3f}")
        logger.info(f"   Models agree: {self.stats['models_agree']} ({self.stats['models_agree']/len(ensemble_scores)*100:.1f}%)")

        return ensemble_scores, confidence, is_fake

    def add_predictions_to_dataframe(self, df: pd.DataFrame,
                                    xgb_scores: np.ndarray,
                                    bert_scores: np.ndarray,
                                    ensemble_scores: np.ndarray,
                                    confidence: np.ndarray,
                                    is_fake: np.ndarray) -> pd.DataFrame:
        """Add prediction columns to dataframe"""
        logger.info("\n Adding predictions to dataframe...")

        df_predictions = df.copy()

        # Add prediction columns
        df_predictions['xgboost_score'] = xgb_scores
        df_predictions['bert_score'] = bert_scores
        df_predictions['ensemble_score'] = ensemble_scores
        df_predictions['confidence'] = confidence
        df_predictions['is_fake'] = is_fake
        df_predictions['fake_probability'] = ensemble_scores
        df_predictions['predicted_at'] = datetime.now().isoformat()

        logger.info(" Predictions added to dataframe")
        return df_predictions

    def save_predictions(self, df: pd.DataFrame):
        """Save predictions to files"""
        logger.info("\n Saving predictions...")

        # Save CSV
        csv_path = self.final_dir / "reviews_with_predictions.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8')
        logger.info(f" Saved CSV: {csv_path}")

        # Save JSON (sample for demo)
        json_data = df.head(100).to_dict(orient='records')
        json_path = self.final_dir / "reviews_with_predictions_sample.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        logger.info(f" Saved JSON sample: {json_path}")

        # Save full JSON (can be large)
        full_json_path = self.final_dir / "reviews_with_predictions.json"
        df.to_json(full_json_path, orient='records', force_ascii=False, indent=2)
        logger.info(f" Saved full JSON: {full_json_path}")

        # Save only fake reviews
        fake_reviews = df[df['is_fake'] == True]
        fake_csv_path = self.final_dir / "fake_reviews_detected.csv"
        fake_reviews.to_csv(fake_csv_path, index=False, encoding='utf-8')
        logger.info(f" Saved fake reviews: {fake_csv_path} ({len(fake_reviews)} reviews)")

    def generate_report(self, df: pd.DataFrame):
        """Generate prediction report"""
        logger.info("\n Generating prediction report...")

        report_lines = []
        report_lines.append("="*60)
        report_lines.append("FAKE REVIEW DETECTION REPORT")
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("="*60)
        report_lines.append("")

        report_lines.append(" OVERALL STATISTICS")
        report_lines.append("-" * 40)
        report_lines.append(f"Total reviews analyzed:    {self.stats['total_reviews']:,}")
        report_lines.append(f"Fake reviews detected:     {self.stats['fake_detected']:,} ({self.stats['fake_detected']/self.stats['total_reviews']*100:.1f}%)")
        report_lines.append(f"Real reviews detected:     {self.stats['real_detected']:,} ({self.stats['real_detected']/self.stats['total_reviews']*100:.1f}%)")
        report_lines.append("")

        report_lines.append(" CONFIDENCE LEVELS")
        report_lines.append("-" * 40)
        report_lines.append(f"High confidence:           {self.stats['high_confidence']:,} ({self.stats['high_confidence']/self.stats['total_reviews']*100:.1f}%)")
        report_lines.append(f"Medium confidence:         {self.stats['medium_confidence']:,} ({self.stats['medium_confidence']/self.stats['total_reviews']*100:.1f}%)")
        report_lines.append(f"Low confidence:            {self.stats['low_confidence']:,} ({self.stats['low_confidence']/self.stats['total_reviews']*100:.1f}%)")
        report_lines.append("")

        report_lines.append(" MODEL AGREEMENT")
        report_lines.append("-" * 40)
        report_lines.append(f"Models agree (<10% diff):  {self.stats['models_agree']:,} ({self.stats['models_agree']/self.stats['total_reviews']*100:.1f}%)")
        report_lines.append(f"Models disagree (>30%):    {self.stats['models_disagree']:,} ({self.stats['models_disagree']/self.stats['total_reviews']*100:.1f}%)")
        report_lines.append("")

        # Distribution by rating
        if 'rating' in df.columns:
            report_lines.append(" FAKE REVIEWS BY RATING")
            report_lines.append("-" * 40)
            fake_by_rating = df[df['is_fake'] == True].groupby('rating').size()
            for rating, count in fake_by_rating.items():
                pct = count / self.stats['fake_detected'] * 100
                report_lines.append(f"Rating {rating}:                {count:5,} ({pct:5.1f}% of fake reviews)")
            report_lines.append("")

        # Top fake reviews
        report_lines.append(" TOP 10 MOST SUSPICIOUS REVIEWS")
        report_lines.append("-" * 40)
        top_fake = df.nlargest(10, 'fake_probability')
        for i, (_, row) in enumerate(top_fake.iterrows(), 1):
            text = row['text'][:50] if 'text' in row else 'N/A'
            score = row['fake_probability']
            rating = row.get('rating', 'N/A')
            report_lines.append(f"{i:2}. Score: {score:.3f} | Rating: {rating} | {text}...")
        report_lines.append("")

        report_lines.append("="*60)
        report_lines.append(" PREDICTION COMPLETE")
        report_lines.append("="*60)

        # Print and save
        report_text = "\n".join(report_lines)
        print("\n" + report_text)

        report_path = self.final_dir / "prediction_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        logger.info(f" Report saved: {report_path}")

        # Save stats JSON (convert ALL numpy types to native Python types)
        import numpy as np
        stats_path = self.final_dir / "prediction_stats.json"
        
        def convert_to_native(obj):
            """Recursively convert numpy types to native Python types"""
            if isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(v) for v in obj]
            else:
                return obj
        
        stats_native = convert_to_native(self.stats)
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats_native, f, indent=2)
        logger.info(f" Stats saved: {stats_path}")

    def run(self):
        """Main pipeline execution"""
        logger.info("="*60)
        logger.info(" STARTING FAKE REVIEW PREDICTIONS")
        logger.info("="*60)

        # Step 1: Load models
        self.load_models()

        # Step 2: Load reviews
        df_reviews = self.load_reviews()

        # Step 3: XGBoost predictions
        xgb_scores = self.predict_xgboost(df_reviews)

        # Step 4: BERT predictions
        bert_scores = self.predict_bert(df_reviews)

        # Step 5: Ensemble
        ensemble_scores, confidence, is_fake = self.ensemble_predictions(
            xgb_scores, bert_scores
        )

        # Step 6: Add to dataframe
        df_with_predictions = self.add_predictions_to_dataframe(
            df_reviews, xgb_scores, bert_scores,
            ensemble_scores, confidence, is_fake
        )

        # Step 7: Save
        self.save_predictions(df_with_predictions)

        # Step 8: Report
        self.generate_report(df_with_predictions)

        logger.info("\n" + "="*60)
        logger.info(" SUCCESS! Predictions complete")
        logger.info("="*60)


def main():
    """Main execution"""
    try:
        predictor = FakeReviewPredictor()
        predictor.run()

    except Exception as e:
        logger.exception(f" Fatal error: {e}")
        raise


if __name__ == "__main__":
    main()