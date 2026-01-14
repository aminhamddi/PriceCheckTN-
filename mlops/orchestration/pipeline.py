"""\
Prefect Pipeline Orchestration for PriceCheckTN MLOps\
"""

from prefect import flow, task, get_run_logger
from datetime import datetime
import subprocess
import sys
import os

@task(name="Run DVC Pipeline", retries=2, retry_delay_seconds=30)
def run_dvc_pipeline():
    """Run the complete DVC pipeline"""
    logger = get_run_logger()
    logger.info(" Starting DVC pipeline...")

    try:
        result = subprocess.run([
            "dvc", "repro"
        ], capture_output=True, text=True, cwd=".")

        if result.returncode != 0:
            logger.error(f" DVC pipeline failed: {result.stderr}")
            raise Exception(f"DVC pipeline failed: {result.stderr}")

        logger.info(" DVC pipeline completed successfully")
        return True

    except Exception as e:
        logger.error(f" DVC pipeline error: {e}")
        raise

@task(name="Train BERT Model", retries=1)
def train_bert_model():
    """Train BERT model with MLflow integration"""
    logger = get_run_logger()
    logger.info(" Starting BERT training...")

    try:
        result = subprocess.run([
            sys.executable, "mlops/training/bert_training.py",
            "--epochs", "3",
            "--batch-size", "16"
        ], capture_output=True, text=True, cwd=".")

        if result.returncode != 0:
            logger.error(f"❌ BERT training failed: {result.stderr}")
            raise Exception(f"BERT training failed: {result.stderr}")

        logger.info(" BERT training completed successfully")
        return True

    except Exception as e:
        logger.error(f" BERT training error: {e}")
        raise

@task(name="Update Model Registry", retries=1)
def update_model_registry():
    """Promote model to production if tests pass"""
    logger = get_run_logger()
    logger.info(" Updating model registry...")

    try:
        # Check if we should promote (simplified for demo)
        result = subprocess.run([
            sys.executable, "mlops/model_registry/cli.py",
            "promote", "bert_fake_review_detector"
        ], capture_output=True, text=True, cwd=".")

        if result.returncode != 0:
            logger.warning(f"  Model promotion failed: {result.stderr}")
            return False

        logger.info(" Model promoted to production")
        return True

    except Exception as e:
        logger.error(f" Model registry error: {e}")
        return False

@task(name="Notify Completion", retries=1)
def notify_completion(success: bool):
    """Notify about pipeline completion"""
    logger = get_run_logger()
    if success:
        logger.info(" Pipeline completed successfully!")
    else:
        logger.error(" Pipeline completed with errors")

@flow(
    name="PriceCheckTN MLOps Pipeline",
    description="Complete MLOps pipeline with DVC, MLflow, and Model Registry",
    retries=1,
    retry_delay_seconds=60
)
def pricechecktn_mlops_pipeline():
    """Main MLOps pipeline flow"""
    logger = get_run_logger()
    logger.info(" Starting PriceCheckTN MLOps Pipeline")

    # Run complete workflow
    dvc_success = run_dvc_pipeline()
    bert_success = train_bert_model()
    registry_success = update_model_registry()

    # Overall success
    pipeline_success = dvc_success and bert_success

    # Notify completion
    notify_completion(pipeline_success)

    return {
        "status": "success" if pipeline_success else "partial",
        "dvc": dvc_success,
        "bert_training": bert_success,
        "model_registry": registry_success,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    # Run pipeline locally
    result = pricechecktn_mlops_pipeline()
    print(f"Pipeline result: {result}")
if __name__ == "__main__":
    # Run pipeline locally
    result = pricechecktn_mlops_pipeline()
    print(f"Pipeline result: {result}")
