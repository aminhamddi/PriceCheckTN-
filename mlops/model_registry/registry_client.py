"""Model Registry Client

MLflow-based model registry with enhanced functionality for PriceCheckTN.
"""

import mlflow
from mlflow.tracking import MlflowClient
from typing import Dict, Optional, List
from mlops.experiment_tracking import get_mlflow_client

class ModelRegistry:
    """MLflow-based model registry with enhanced functionality"""

    def __init__(self):
        """Initialize model registry client"""
        self.client = get_mlflow_client()
        self.mlflow_client = MlflowClient()

    def register_model(
        self,
        model_uri: str,
        name: str,
        tags: Optional[Dict[str, str]] = None,
        description: Optional[str] = None
    ) -> str:
        """Register a model in the MLflow registry

        Args:
            model_uri: URI of the model to register
            name: Model name in registry
            tags: Optional tags
            description: Optional description

        Returns:
            Model version
        """
        result = self.mlflow_client.create_model_version(
            name=name,
            source=model_uri,
            run_id=mlflow.active_run().info.run_id,
            tags=tags or {},
            description=description
        )
        return result.version

    def transition_stage(
        self,
        model_name: str,
        version: int,
        stage: str,
        archive_existing_versions: bool = False
    ) -> None:
        """Transition model version to a different stage

        Args:
            model_name: Name of the model
            version: Version number
            stage: Target stage (Staging, Production, Archived)
            archive_existing_versions: Whether to archive existing versions in target stage
        """
        self.mlflow_client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage,
            archive_existing_versions=archive_existing_versions
        )

    def get_model_version(self, model_name: str, version: int) -> dict:
        """Get model version details

        Args:
            model_name: Name of the model
            version: Version number

        Returns:
            Model version details
        """
        return self.mlflow_client.get_model_version(model_name, version)

    def list_model_versions(self, model_name: str) -> List[dict]:
        """List all versions of a model

        Args:
            model_name: Name of the model

        Returns:
            List of model versions
        """
        return self.mlflow_client.search_model_versions(f"name='{model_name}'")

    def get_production_model(self, model_name: str) -> Optional[str]:
        """Get the production version of a model

        Args:
            model_name: Name of the model

        Returns:
            Model URI if exists, None otherwise
        """
        versions = self.list_model_versions(model_name)
        for version in versions:
            if version.current_stage == "Production":
                return f"models:/{model_name}/{version.version}"
        return None

    def get_latest_version(self, model_name: str) -> Optional[dict]:
        """Get the latest version of a model

        Args:
            model_name: Name of the model

        Returns:
            Latest model version details
        """
        versions = self.list_model_versions(model_name)
        if not versions:
            return None
        return max(versions, key=lambda x: x.version)

    def archive_old_versions(self, model_name: str, keep_last: int = 3) -> None:
        """Archive old model versions, keeping the most recent ones

        Args:
            model_name: Name of the model
            keep_last: Number of recent versions to keep
        """
        versions = self.list_model_versions(model_name)
        versions.sort(key=lambda x: x.version, reverse=True)

        for version in versions[keep_last:]:
            if version.current_stage != "Archived":
                self.transition_stage(model_name, version.version, "Archived")

    def promote_to_production(self, model_name: str, version: int) -> None:
        """Promote a model version to production, archiving existing production version

        Args:
            model_name: Name of the model
            version: Version to promote
        """
        # Archive existing production version
        production_version = self.get_production_model(model_name)
        if production_version:
            # Extract version number from URI
            version_num = int(production_version.split("/")[-1])
            self.transition_stage(model_name, version_num, "Archived")

        # Promote new version
        self.transition_stage(model_name, version, "Production")