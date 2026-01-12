"""Model Serving Utilities

Utilities for loading and serving models from the registry.
"""

import mlflow
from typing import Optional, Dict, Any
from mlops.model_registry import get_registry
import logging

logger = logging.getLogger(__name__)

def load_production_model(model_name: str):
    """Load a production model from registry

    Args:
        model_name: Name of the model to load

    Returns:
        Loaded model

    Raises:
        ValueError: If no production model exists
    """
    registry = get_registry()
    model_uri = registry.get_production_model(model_name)

    if model_uri is None:
        raise ValueError(f"No production model found for {model_name}")

    logger.info(f"Loading production model: {model_name}")
    return mlflow.pyfunc.load_model(model_uri)

def load_staging_model(model_name: str):
    """Load a staging model from registry

    Args:
        model_name: Name of the model to load

    Returns:
        Loaded model

    Raises:
        ValueError: If no staging model exists
    """
    registry = get_registry()
    versions = registry.list_model_versions(model_name)

    for version in versions:
        if version.current_stage == "Staging":
            model_uri = f"models:/{model_name}/{version.version}"
            logger.info(f"Loading staging model: {model_name} v{version.version}")
            return mlflow.pyfunc.load_model(model_uri)

    raise ValueError(f"No staging model found for {model_name}")

def load_model_by_version(model_name: str, version: int):
    """Load a specific model version

    Args:
        model_name: Name of the model
        version: Version number to load

    Returns:
        Loaded model
    """
    model_uri = f"models:/{model_name}/{version}"
    logger.info(f"Loading model: {model_name} v{version}")
    return mlflow.pyfunc.load_model(model_uri)

def get_model_metadata(model_name: str, version: Optional[int] = None) -> Dict[str, Any]:
    """Get metadata for a model version

    Args:
        model_name: Name of the model
        version: Specific version (None for latest)

    Returns:
        Model metadata dictionary
    """
    registry = get_registry()
    if version is None:
        versions = registry.list_model_versions(model_name)
        if not versions:
            raise ValueError(f"No versions found for model {model_name}")
        version_info = max(versions, key=lambda x: x.version)
    else:
        version_info = registry.get_model_version(model_name, version)

    return {
        "name": model_name,
        "version": version_info.version,
        "stage": version_info.current_stage,
        "run_id": version_info.run_id,
        "timestamp": version_info.creation_timestamp,
        "description": version_info.description,
        "tags": version_info.tags
    }

def get_model_artifact_path(model_name: str, version: Optional[int] = None) -> str:
    """Get the artifact path for a model version

    Args:
        model_name: Name of the model
        version: Specific version (None for latest)

    Returns:
        Artifact path
    """
    if version is None:
        return f"models:/{model_name}"
    else:
        return f"models:/{model_name}/{version}"

def list_available_models() -> Dict[str, Dict[str, Any]]:
    """List all available models in the registry

    Returns:
        Dictionary of model names to their latest version info
    """
    registry = get_registry()
    # This would normally use mlflow.search_registered_models()
    # For now, we'll return a placeholder
    return {
        "bert_fake_review_detector": {
            "latest_version": 1,
            "production_version": None,
            "staging_version": 1
        }
    }

def create_model_serving_config(model_name: str, version: Optional[int] = None) -> Dict[str, Any]:
    """Create a serving configuration for a model

    Args:
        model_name: Name of the model
        version: Specific version (None for production)

    Returns:
        Serving configuration dictionary
    """
    if version is None:
        # Use production version
        model_uri = get_model_artifact_path(model_name)
    else:
        model_uri = get_model_artifact_path(model_name, version)

    return {
        "model_uri": model_uri,
        "model_name": model_name,
        "version": version or "production",
        "framework": "mlflow",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string"}
            },
            "required": ["text"]
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "prediction": {"type": "integer"},
                "probability": {"type": "number"},
                "label": {"type": "string"}
            }
        }
    }