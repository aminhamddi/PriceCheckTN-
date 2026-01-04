"""
BERT Fine-tuning Trainer
"""

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
from pathlib import Path
from loguru import logger
import json
from datetime import datetime


class BERTFakeReviewTrainer:
    """
    Trainer for BERT-based fake review detection
    """

    def __init__(
            self,
            model_name: str = 'distilbert-base-multilingual-cased',
            num_labels: int = 2,
            output_dir: str = 'models/saved_models/bert_fake_review'
    ):
        """
        Args:
            model_name: Pretrained model to use
                - 'distilbert-base-multilingual-cased' (Multilingual, FAST)
                - 'camembert-base' (French only)
                - 'aubmindlab/bert-base-arabertv2' (Arabic only)
            num_labels: Number of classes (2 for binary)
            output_dir: Where to save trained model
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load model and tokenizer
        logger.info(f"🤖 Loading model: {model_name}")

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Check GPU availability
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"💻 Using device: {self.device}")

        if self.device.type == 'cuda':
            logger.info(f"   GPU: {torch.cuda.get_device_name(0)}")

        self.model.to(self.device)

        self.trainer = None
        self.training_history = {}

    def compute_metrics(self, eval_pred):
        """
        Compute metrics during training
        """
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)

        # Calculate metrics
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='binary'
        )

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def train(
            self,
            train_dataset,
            eval_dataset,
            epochs: int = 3,
            batch_size: int = 16,
            learning_rate: float = 2e-5,
            warmup_steps: int = 500,
            weight_decay: float = 0.01,
            early_stopping_patience: int = 2,
            save_total_limit: int = 2
    ):
        """
        Fine-tune BERT model

        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
            warmup_steps: Warmup steps for learning rate scheduler
            weight_decay: Weight decay for optimizer
            early_stopping_patience: Patience for early stopping
            save_total_limit: Max number of checkpoints to keep
        """
        logger.info("🚀 Starting BERT fine-tuning...")
        logger.info(f"   Epochs: {epochs}")
        logger.info(f"   Batch size: {batch_size}")
        logger.info(f"   Learning rate: {learning_rate}")

        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir / 'checkpoints'),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            logging_dir=str(self.output_dir / 'logs'),
            logging_steps=10,
            evaluation_strategy='epoch',
            save_strategy='epoch',
            load_best_model_at_end=True,
            metric_for_best_model='f1',
            save_total_limit=save_total_limit,
            push_to_hub=False,
            report_to='none',  # Disable wandb/tensorboard if not needed
        )

        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)]
        )

        # Train
        logger.info("⏳ Training started...")
        train_result = self.trainer.train()

        # Save training history
        self.training_history = {
            'train_loss': train_result.training_loss,
            'train_runtime': train_result.metrics['train_runtime'],
            'train_samples_per_second': train_result.metrics['train_samples_per_second'],
        }

        logger.success("✅ Training complete!")
        logger.info(f"   Final loss: {train_result.training_loss:.4f}")
        logger.info(f"   Time: {train_result.metrics['train_runtime']:.2f}s")

        return train_result

    def evaluate(self, eval_dataset):
        """
        Evaluate model on test set
        """
        logger.info("📊 Evaluating model...")

        eval_results = self.trainer.evaluate(eval_dataset)

        logger.success("✅ Evaluation complete!")
        logger.info(f"   Accuracy: {eval_results['eval_accuracy']:.4f}")
        logger.info(f"   F1: {eval_results['eval_f1']:.4f}")
        logger.info(f"   Precision: {eval_results['eval_precision']:.4f}")
        logger.info(f"   Recall: {eval_results['eval_recall']:.4f}")

        return eval_results

    def predict(self, texts: list):
        """
        Predict on new texts

        Args:
            texts: List of review texts

        Returns:
            predictions: List of predicted labels
            probabilities: List of confidence scores
        """
        # Tokenize
        encodings = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )

        encodings = {k: v.to(self.device) for k, v in encodings.items()}

        # Predict
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**encodings)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)

        return predictions.cpu().numpy(), probabilities.cpu().numpy()

    def save_model(self, save_path: str = None):
        """
        Save fine-tuned model
        """
        if save_path is None:
            save_path = self.output_dir / 'final_model'
        else:
            save_path = Path(save_path)

        save_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"💾 Saving model to {save_path}")

        # Save model and tokenizer
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

        # Save training history
        history_path = save_path / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)

        # Save metadata
        metadata = {
            'model_name': self.model_name,
            'num_labels': self.num_labels,
            'trained_at': datetime.now().isoformat(),
            'device': str(self.device)
        }

        metadata_path = save_path / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.success(f"✅ Model saved to {save_path}")

    @classmethod
    def load_model(cls, model_path: str):
        """
        Load a saved fine-tuned model
        """
        model_path = Path(model_path)

        logger.info(f"📥 Loading model from {model_path}")

        # Load metadata
        metadata_path = model_path / 'metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            logger.info(f"   Original model: {metadata['model_name']}")
            logger.info(f"   Trained: {metadata['trained_at']}")

        # Create instance
        trainer = cls(
            model_name=str(model_path),  # Load from local path
            num_labels=2
        )

        logger.success("✅ Model loaded successfully")

        return trainer