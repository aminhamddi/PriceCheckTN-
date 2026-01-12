"""Model Registry Module

MLflow-based model registry for PriceCheckTN.
"""

from .registry_client import ModelRegistry

# Global registry instance
registry = ModelRegistry()

def get_registry() -> ModelRegistry:
    """Get the global model registry instance

    Returns:
        ModelRegistry instance
    """
    return registry

__all__ = ["ModelRegistry", "get_registry", "registry"]