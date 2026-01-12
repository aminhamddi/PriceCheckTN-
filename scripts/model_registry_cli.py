#!/usr/bin/env python3
"""Model Registry CLI

Command-line interface for managing models in the MLflow registry.
"""

import click
import logging
from mlops.model_registry import get_registry
from mlops.model_registry.serving_utils import get_model_metadata

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@click.group()
def cli():
    """Model Registry CLI"""
    pass

@cli.command()
@click.argument('model_name')
@click.argument('version', type=int)
@click.argument('stage')
def transition(model_name, version, stage):
    """Transition model version to a stage"""
    registry = get_registry()

    try:
        # Validate stage
        valid_stages = ['Staging', 'Production', 'Archived']
        if stage not in valid_stages:
            logger.error(f"❌ Invalid stage: {stage}. Must be one of: {valid_stages}")
            return

        # Get current version info
        version_info = registry.get_model_version(model_name, version)
        current_stage = version_info.current_stage

        logger.info(f"Transitioning {model_name} v{version} from {current_stage} to {stage}")

        # Perform transition
        registry.transition_stage(model_name, version, stage)

        logger.info(f"✅ Successfully transitioned {model_name} v{version} to {stage}")

    except Exception as e:
        logger.error(f"❌ Failed to transition model: {e}")

@cli.command()
@click.argument('model_name')
def list_versions(model_name):
    """List all versions of a model"""
    registry = get_registry()

    try:
        versions = registry.list_model_versions(model_name)

        if not versions:
            logger.info(f"📋 No versions found for model: {model_name}")
            return

        logger.info(f"📋 Versions for model: {model_name}")
        logger.info("=" * 60)

        for version in versions:
            logger.info(f"Version {version.version}:")
            logger.info(f"  Stage: {version.current_stage}")
            logger.info(f"  Run ID: {version.run_id}")
            logger.info(f"  Timestamp: {version.creation_timestamp}")
            logger.info(f"  Description: {version.description}")
            logger.info(f"  Tags: {version.tags}")
            logger.info("-" * 40)

    except Exception as e:
        logger.error(f"❌ Failed to list versions: {e}")

@cli.command()
@click.argument('model_name')
def promote(model_name):
    """Promote staging model to production"""
    registry = get_registry()

    try:
        # Get all versions
        versions = registry.list_model_versions(model_name)

        if not versions:
            logger.error(f"❌ No versions found for model: {model_name}")
            return

        # Find staging version
        staging_versions = [v for v in versions if v.current_stage == "Staging"]

        if not staging_versions:
            logger.error(f"❌ No staging version found for {model_name}")
            return

        # Get latest staging version
        latest_staging = max(staging_versions, key=lambda x: x.version)

        # Check if there's a production version
        production_versions = [v for v in versions if v.current_stage == "Production"]

        if production_versions:
            latest_production = max(production_versions, key=lambda x: x.version)
            logger.info(f"📋 Archiving current production version {latest_production.version}")
            registry.transition_stage(model_name, latest_production.version, "Archived")

        # Promote staging to production
        logger.info(f"📋 Promoting {model_name} v{latest_staging.version} to Production")
        registry.transition_stage(model_name, latest_staging.version, "Production")

        logger.info(f"✅ Successfully promoted {model_name} v{latest_staging.version} to Production")

    except Exception as e:
        logger.error(f"❌ Failed to promote model: {e}")

@cli.command()
@click.argument('model_name')
def info(model_name):
    """Get detailed information about a model"""
    try:
        metadata = get_model_metadata(model_name)
        logger.info(f"📋 Model Information: {model_name}")
        logger.info("=" * 50)
        logger.info(f"Name: {metadata['name']}")
        logger.info(f"Version: {metadata['version']}")
        logger.info(f"Stage: {metadata['stage']}")
        logger.info(f"Run ID: {metadata['run_id']}")
        logger.info(f"Timestamp: {metadata['timestamp']}")
        logger.info(f"Description: {metadata['description']}")
        logger.info(f"Tags: {metadata['tags']}")
        logger.info("=" * 50)

    except Exception as e:
        logger.error(f"❌ Failed to get model info: {e}")

@cli.command()
@click.argument('model_name')
@click.argument('version', type=int)
def archive(model_name, version):
    """Archive a specific model version"""
    registry = get_registry()

    try:
        # Get version info
        version_info = registry.get_model_version(model_name, version)
        current_stage = version_info.current_stage

        if current_stage == "Archived":
            logger.info(f"📋 Model {model_name} v{version} is already archived")
            return

        logger.info(f"📋 Archiving {model_name} v{version} (currently {current_stage})")
        registry.transition_stage(model_name, version, "Archived")

        logger.info(f"✅ Successfully archived {model_name} v{version}")

    except Exception as e:
        logger.error(f"❌ Failed to archive model: {e}")

@cli.command()
@click.argument('model_name')
def cleanup(model_name):
    """Clean up old model versions, keeping only the most recent"""
    registry = get_registry()

    try:
        versions = registry.list_model_versions(model_name)
        if not versions:
            logger.info(f"📋 No versions to clean up for {model_name}")
            return

        logger.info(f"📋 Cleaning up old versions of {model_name}")
        logger.info(f"Found {len(versions)} versions")

        # Archive all but the 3 most recent versions
        registry.archive_old_versions(model_name, keep_last=3)

        logger.info(f"✅ Cleanup completed. Kept 3 most recent versions.")

    except Exception as e:
        logger.error(f"❌ Failed to clean up model versions: {e}")

if __name__ == "__main__":
    cli()