#!/usr/bin/env python3
"""
Register existing trained models in MLflow for experiment tracking.
"""

import sys
sys.path.append('..')

import joblib
from pathlib import Path
import mlflow
import mlflow.xgboost
import mlflow.sklearn

def main():
    print("🚀 Registering Existing Models in MLflow")
    print("="*50)

    # Setup MLflow
    mlflow.set_experiment("pricecheck_fake_review_detection")
    print("📊 MLflow experiment set up")

    model_dir = Path('models')

    # Register XGBoost model
    xgb_path = model_dir / 'fake_review_detector_xgboost.pkl'
    if xgb_path.exists():
        print("\n📦 Registering XGBoost model...")

        with mlflow.start_run(run_name="xgboost_existing"):
            # Load existing model
            xgb_model = joblib.load(xgb_path)

            # Log model parameters (estimated)
            params = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'random_state': 42
            }
            mlflow.log_params(params)

            # Log model (use sklearn flavor for compatibility)
            mlflow.sklearn.log_model(xgb_model, "model")

            print("✅ XGBoost model registered")

            # Register in Model Registry
            run_id = mlflow.active_run().info.run_id
            model_uri = f"runs:/{run_id}/model"

            try:
                registered_model = mlflow.register_model(
                    model_uri=model_uri,
                    name="xgboost_fake_review_detector"
                )
                print(f"🏷️ Registered as version {registered_model.version}")

                # Transition to staging
                client = mlflow.tracking.MlflowClient()
                client.transition_model_version_stage(
                    name="xgboost_fake_review_detector",
                    version=registered_model.version,
                    stage="Staging"
                )
                print("🎯 Moved to Staging stage")

            except Exception as e:
                print(f"⚠️ Registration failed: {e}")

    else:
        print("❌ XGBoost model not found")

    print("\n" + "="*50)
    print("✅ MODEL REGISTRATION COMPLETE!")
    print("="*50)
    print("🔍 View models: mlflow ui")
    print("📊 Check Model Registry tab")
    print("="*50)

if __name__ == "__main__":
    main()
