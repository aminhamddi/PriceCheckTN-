#!/usr/bin/env python3
"""
Train ML models with MLflow integration for fake review detection.
"""

import sys
sys.path.append('..')

import pandas as pd
import numpy as np
from pathlib import Path
import joblib

# ML imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score
)
import xgboost as xgb

# MLflow for experiment tracking
import mlflow
import mlflow.xgboost
import mlflow.sklearn

def evaluate_model(y_true, y_pred, y_proba):
    """Evaluate model and return metrics"""
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    auc = roc_auc_score(y_true, y_proba)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': auc
    }

def main():
    print(" Starting ML Training with MLflow Integration")
    print("="*60)

    # Setup MLflow
    mlflow.set_experiment("pricecheck_fake_review_detection")
    print(" MLflow experiment set up: 'pricecheck_fake_review_detection'")

    # Load data
    df = pd.read_csv('data/processed/reviews_with_features.csv')
    print(f" Loaded {len(df)} reviews with features")

    # Prepare data
    feature_columns = [
        'text_length', 'word_count', 'avg_word_length',
        'exclamation_count', 'question_count', 'capital_ratio',
        'digit_count', 'unique_word_ratio', 'sentence_count',
        'sentiment_polarity', 'sentiment_subjectivity',
        'rating', 'is_extreme_rating', 'is_very_short', 'is_very_long'
    ]

    X = df[feature_columns]
    y = df['is_fake']

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f" Data prepared")
    print(f"   Training: {len(X_train)} samples")
    print(f"   Testing: {len(X_test)} samples")

    # Train models with MLflow tracking
    models_results = []

    # 1. Logistic Regression
    print("\n Training Logistic Regression...")
    with mlflow.start_run(run_name="logistic_regression"):
        params = {'max_iter': 1000, 'random_state': 42}
        mlflow.log_params(params)

        lr_model = LogisticRegression(**params)
        lr_model.fit(X_train_scaled, y_train)

        lr_pred = lr_model.predict(X_test_scaled)
        lr_proba = lr_model.predict_proba(X_test_scaled)[:, 1]
        lr_metrics = evaluate_model(y_test, lr_pred, lr_proba)

        mlflow.log_metrics(lr_metrics)
        mlflow.sklearn.log_model(lr_model, "model")

        models_results.append(('Logistic Regression', lr_metrics))
        print(f" Accuracy: {lr_metrics['accuracy']:.4f} ({lr_metrics['accuracy']*100:.2f}%)")
        print(f" F1 Score: {lr_metrics['f1_score']:.4f}")
        print(f" AUC: {lr_metrics['auc']:.4f}")

    # 2. Random Forest
    print("\n Training Random Forest...")
    with mlflow.start_run(run_name="random_forest"):
        params = {'n_estimators': 100, 'random_state': 42, 'n_jobs': -1}
        mlflow.log_params(params)

        rf_model = RandomForestClassifier(**params)
        rf_model.fit(X_train, y_train)

        rf_pred = rf_model.predict(X_test)
        rf_proba = rf_model.predict_proba(X_test)[:, 1]
        rf_metrics = evaluate_model(y_test, rf_pred, rf_proba)

        mlflow.log_metrics(rf_metrics)
        mlflow.sklearn.log_model(rf_model, "model")

        models_results.append(('Random Forest', rf_metrics))
        print(f" Accuracy: {rf_metrics['accuracy']:.4f} ({rf_metrics['accuracy']*100:.2f}%)")
        print(f" F1 Score: {rf_metrics['f1_score']:.4f}")
        print(f" AUC: {rf_metrics['auc']:.4f}")

        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        print(" Top features:", feature_importance.head(3)['feature'].tolist())

    # 3. XGBoost
    print("\n Training XGBoost...")
    with mlflow.start_run(run_name="xgboost"):
        params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': 42,
            'n_jobs': -1
        }
        mlflow.log_params(params)

        xgb_model = xgb.XGBClassifier(**params)
        xgb_model.fit(X_train, y_train)

        xgb_pred = xgb_model.predict(X_test)
        xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
        xgb_metrics = evaluate_model(y_test, xgb_pred, xgb_proba)

        mlflow.log_metrics(xgb_metrics)
        mlflow.xgboost.log_model(xgb_model, "model")

        models_results.append(('XGBoost', xgb_metrics))
        print(f" Accuracy: {xgb_metrics['accuracy']:.4f} ({xgb_metrics['accuracy']*100:.2f}%)")
        print(f" F1 Score: {xgb_metrics['f1_score']:.4f}")
        print(f" AUC: {xgb_metrics['auc']:.4f}")
    # Model comparison
    print("\n MODEL COMPARISON")
    print("="*50)
    for name, metrics in models_results:
        print("20")

    # Save best model
    print("\n Saving best model...")
    best_model = max(models_results, key=lambda x: x[1]['accuracy'])
    best_name, best_metrics = best_model

    print(f" Best model: {best_name} (Accuracy: {best_metrics['accuracy']:.4f})")

    # Save model files
    model_dir = Path('models')
    model_dir.mkdir(exist_ok=True)

    joblib.dump(xgb_model, model_dir / 'fake_review_detector_xgboost.pkl')
    joblib.dump(scaler, model_dir / 'feature_scaler.pkl')

    with open(model_dir / 'feature_names.txt', 'w') as f:
        f.write('\n'.join(feature_columns))

    print(f" Models saved to: {model_dir}")

    # Register best model in MLflow Registry
    print("\n Registering best model in MLflow Registry...")

    try:
        # Get the XGBoost run
        experiment = mlflow.get_experiment_by_name("pricecheck_fake_review_detection")
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        xgb_run = runs[runs['tags.mlflow.runName'] == 'xgboost'].iloc[0]

        model_uri = f"runs:/{xgb_run['run_id']}/model"

        # Register model
        registered_model = mlflow.register_model(
            model_uri=model_uri,
            name="xgboost_fake_review_detector"
        )

        print(f" Model registered as version {registered_model.version}")

        # Transition to staging
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name="xgboost_fake_review_detector",
            version=registered_model.version,
            stage="Staging"
        )

        print(" Model transitioned to Staging stage")

    except Exception as e:
        print(f" Model registration failed: {e}")

    print("\n" + "="*60)
    print(" ML TRAINING WITH MLFLOW COMPLETE!")
    print("="*60)
    print(" View experiments: mlflow ui")
    print(" Compare models in MLflow UI")
    print(" Models ready for production deployment")
    print("="*60)

if __name__ == "__main__":
    main()
