# 🧪 Experiment Tracking

## Overview

This module provides MLflow integration for tracking experiments, metrics, and model performance in the PriceCheckTN project.

## Features

- **Experiment Tracking**: Track all ML experiments with parameters, metrics, and artifacts
- **Model Registry**: Register and version trained models
- **Metric Logging**: Log training metrics and performance indicators
- **Artifact Storage**: Store model files, plots, and other artifacts

## Components

### MLflow Client (`mlflow_client.py`)

The main client for interacting with MLflow:

```python
from mlops.experiment_tracking.mlflow_client import get_mlflow_client

# Get MLflow client
mlflow_client = get_mlflow_client()

# Start an experiment
with mlflow_client.start_experiment("bert_training"):
    # Your training code here
    mlflow_client.log_metric("accuracy", 0.95)
    mlflow_client.log_param("epochs", 10)
```

### Integration with Training Scripts

The BERT training script automatically logs to MLflow:

```python
# In scripts/train_bert.py
mlflow_client = get_mlflow_client()
mlflow_client.start_experiment("bert_fake_review_detection")
mlflow_client.log_params(training_config)
mlflow_client.log_metrics(evaluation_results)
mlflow_client.log_artifact(model_path)
```

## Configuration

Configuration is handled through the main `config` object:

```python
from config import config

# MLflow settings
MLFLOW_TRACKING_URI = config.MLFLOW_TRACKING_URI
MLFLOW_EXPERIMENT_NAME = config.MLFLOW_EXPERIMENT_NAME
```

## Usage

### Starting an Experiment

```python
from mlops.experiment_tracking.mlflow_client import get_mlflow_client

mlflow_client = get_mlflow_client()
mlflow_client.start_experiment("my_experiment")
```

### Logging Metrics

```python
mlflow_client.log_metric("accuracy", 0.95)
mlflow_client.log_metric("loss", 0.05)
```

### Logging Parameters

```python
mlflow_client.log_param("learning_rate", 0.001)
mlflow_client.log_param("batch_size", 32)
```

### Logging Artifacts

```python
mlflow_client.log_artifact("model.pkl")
mlflow_client.log_artifact("training_plot.png")
```

## Monitoring

View experiment results in the MLflow UI:

```bash
mlflow ui
```

Then navigate to `http://localhost:5000` in your browser.

## Best Practices

1. **Experiment Naming**: Use descriptive names like `bert_fake_review_v1`
2. **Metric Logging**: Log both training and validation metrics
3. **Artifact Organization**: Store related files in subdirectories
4. **Parameter Tracking**: Log all hyperparameters for reproducibility

## Troubleshooting

- **Connection Issues**: Check `MLFLOW_TRACKING_URI` in config
- **Permission Errors**: Ensure MLflow server is running and accessible
- **Missing Metrics**: Verify experiment is properly started before logging