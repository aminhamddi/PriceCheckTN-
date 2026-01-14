"""MLflow Experiment Tracking Client

Centralized MLflow tracking utilities for PriceCheckTN.
"""

import mlflow
import os
from datetime import datetime
from typing import Dict, Any, Optional
from config.base import get_config

class MLflowClient:
    """MLflow experiment tracking client with proper configuration"""

    def __init__(self, experiment_name: Optional[str] = None):
        """Initialize MLflow client

        Args:
            experiment_name: Name of the experiment. If None, uses default from config.
        """
        # Get config dynamically to avoid circular imports
        from config.base import get_config
        cfg = get_config()
        
        self.experiment_name = experiment_name or cfg.MLFLOW_EXPERIMENT_NAME
        self.tracking_uri = cfg.MLFLOW_TRACKING_URI

        # Set MLflow tracking URI
        mlflow.set_tracking_uri(self.tracking_uri)

        # Set experiment
        mlflow.set_experiment(self.experiment_name)

    def start_run(self, run_name: Optional[str] = None) -> mlflow.ActiveRun:
        """Start an MLflow run

        Args:
            run_name: Optional name for the run

        Returns:
            Active MLflow run
        """
        if run_name is None:
            run_name = f"{self.experiment_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        return mlflow.start_run(run_name=run_name)

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters to MLflow

        Args:
            params: Dictionary of parameters to log
        """
        mlflow.log_params(params)

    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        """Log a single metric to MLflow

        Args:
            key: Metric name
            value: Metric value
            step: Optional step number
        """
        mlflow.log_metric(key, value, step=step)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log multiple metrics to MLflow

        Args:
            metrics: Dictionary of metrics
            step: Optional step number
        """
        for key, value in metrics.items():
            self.log_metric(key, value, step)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """Log an artifact to MLflow

        Args:
            local_path: Local path to the artifact
            artifact_path: Optional path within the artifact directory
        """
        mlflow.log_artifact(local_path, artifact_path)

    def log_model(
        self,
        model,
        artifact_path: str,
        flavor: Optional[str] = None,
        **kwargs
    ) -> None:
        """Log a model to MLflow

        Args:
            model: Model to log
            artifact_path: Path within the artifact directory
            flavor: MLflow flavor (e.g., 'sklearn', 'transformers')
            **kwargs: Additional arguments for the specific flavor
        """
        if flavor == "transformers":
            mlflow.transformers.log_model(model, artifact_path, **kwargs)
        elif flavor == "sklearn":
            mlflow.sklearn.log_model(model, artifact_path, **kwargs)
        elif flavor == "pytorch":
            mlflow.pytorch.log_model(model, artifact_path, **kwargs)
        else:
            mlflow.log_model(model, artifact_path, **kwargs)

    def log_dict(self, data: Dict[str, Any], filename: str) -> None:
        """Log a dictionary as a JSON artifact

        Args:
            data: Dictionary to log
            filename: Name of the JSON file
        """
        import json
        import tempfile

        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f, indent=2)
            temp_path = f.name

        # Log the file
        self.log_artifact(temp_path, filename)

        # Clean up
        os.unlink(temp_path)

    def get_experiment_id(self) -> str:
        """Get the current experiment ID

        Returns:
            Experiment ID
        """
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        return experiment.experiment_id if experiment else None

    def end_run(self) -> None:
        """End the current MLflow run"""
        mlflow.end_run()

# Global MLflow client instance
mlflow_client = MLflowClient()

def get_mlflow_client() -> MLflowClient:
    """Get the global MLflow client instance

    Returns:
        MLflowClient instance
    """
    return mlflow_client

def log_training_run(
    model,
    params: Dict[str, Any],
    metrics: Dict[str, float],
    artifact_path: str = "model",
    flavor: str = "transformers",
    run_name: Optional[str] = None
) -> str:
    """Convenience function to log a complete training run

    Args:
        model: Trained model to log
        params: Training parameters
        metrics: Evaluation metrics
        artifact_path: Path for the model artifact
        flavor: MLflow flavor
        run_name: Optional run name

    Returns:
        Run ID of the logged run
    """
    client = get_mlflow_client()

    with client.start_run(run_name=run_name):
        # Log parameters
        client.log_params(params)

        # Log metrics
        client.log_metrics(metrics)

        # Log model
        client.log_model(model, artifact_path, flavor=flavor)

        # Get run info
        run_id = mlflow.active_run().info.run_id

    return run_id