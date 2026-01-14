"""\
Train BERT model for fake review detection with MLflow integration\
"""

import sys
import os
import mlflow
from datetime import datetime

sys.path.append('.')

from models.bert.dataset import prepare_datasets_from_csv
from models.bert.trainer import BERTFakeReviewTrainer
from loguru import logger
import argparse
from mlops.experiment_tracking import log_training_run
from mlops.model_registry import get_registry

def main():
    """Main training function"""

    # Parse arguments
    parser = argparse.ArgumentParser(description='Train BERT for fake review detection')
    parser.add_argument('--data', type=str, default='data/raw/reviews/fake_reviews.csv',
                        help='Path to dataset CSV')
    parser.add_argument('--model', type=str, default='distilbert-base-multilingual-cased',
                        help='Pretrained model name')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-5,
                        help='Learning rate')
    parser.add_argument('--experiment', type=str, default='bert_fake_review_detection',
                        help='MLflow experiment name')
    parser.add_argument('--run-name', type=str, default=None,
                        help='MLflow run name')

    args = parser.parse_args()

    # Setup logging
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    logger.add("logs/bert_training.log", rotation="10 MB")

    print("\n" + "=" * 70)
    print(" BERT FINE-TUNING FOR FAKE REVIEW DETECTION")
    print("=" * 70)
    print(f"   Model: {args.model}")
    print(f"   Data: {args.data}")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Learning rate: {args.lr}")
    print(f"   Experiment: {args.experiment}")
    print("=" * 70)

    # Start MLflow run
    if args.run_name is None:
        args.run_name = f"bert-training-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    logger.info("\n Starting MLflow run...")
    mlflow.set_experiment(args.experiment)

    with mlflow.start_run(run_name=args.run_name):
        # Log parameters
        mlflow.log_params({
            "model_name": args.model,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "data_path": args.data
        })

        # Prepare datasets
        logger.info("\n Preparing datasets...")

        train_dataset, test_dataset, label_mapping = prepare_datasets_from_csv(
            csv_path=args.data,
            text_column='text_',
            label_column='label',
            test_size=0.2,
            tokenizer_name=args.model,
            max_length=512
        )

        logger.info(f" Label mapping: {label_mapping}")

        # Initialize trainer
        logger.info("\n Initializing BERT trainer...")

        trainer = BERTFakeReviewTrainer(
            model_name=args.model,
            num_labels=2,
            output_dir='models/saved_models/bert_fake_review'
        )

        # Train
        logger.info("\n Starting training...\n")

        train_result = trainer.train(
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr
        )

        # Evaluate
        logger.info("\n Final evaluation...\n")

        eval_results = trainer.evaluate(test_dataset)

        # Log metrics
        mlflow.log_metrics({
            "accuracy": eval_results['eval_accuracy'],
            "f1_score": eval_results['eval_f1'],
            "precision": eval_results['eval_precision'],
            "recall": eval_results['eval_recall'],
            "loss": eval_results.get('eval_loss', 0.0)
        })

        # Save model
        logger.info("\n Saving model...\n")

        trainer.save_model()

        # Log model with MLflow
        logger.info("\n Logging model to MLflow...")
        mlflow.transformers.log_model(
            transformer_model=trainer.model,
            artifact_path="bert_fake_review",
            registered_model_name="bert_fake_review_detector"
        )

        # Register model in model registry
        logger.info("\n Registering model in registry...")
        registry = get_registry()

        # Get the run ID and create model URI
        run_id = mlflow.active_run().info.run_id
        model_uri = f"runs:/{run_id}/bert_fake_review"

        # Register the model
        version = registry.register_model(
            model_uri=model_uri,
            name="bert_fake_review_detector",
            tags={
                "model_type": "transformers",
                "task": "fake_review_detection",
                "framework": "pytorch",
                "dataset": "tunisianet_reviews"
            },
            description="BERT model for fake review detection trained on Tunisianet data"
        )

        logger.info(f" Model registered as version {version}")

        # Transition to staging
        registry.transition_stage("bert_fake_review_detector", version, "Staging")
        logger.info(f" Model version {version} transitioned to Staging")

        # Print summary
        print("\n" + "=" * 70)
        print(" TRAINING COMPLETE!")
        print("=" * 70)
        print(f"   Accuracy: {eval_results['eval_accuracy']:.2%}")
        print(f"   F1 Score: {eval_results['eval_f1']:.4f}")
        print(f"   Precision: {eval_results['eval_precision']:.4f}")
        print(f"   Recall: {eval_results['eval_recall']:.4f}")
        print(f"   Loss: {eval_results.get('eval_loss', 0.0):.4f}")
        print("=" * 70)
        print(f"\n Model saved to: models/saved_models/bert_fake_review/final_model")
        print(f" Model registered: bert_fake_review_detector v{version} (Staging)")
        print(f" MLflow run: {args.experiment}/{args.run_name}")
        print("=" * 70)
        print("\n💡 Next steps:")
        print("   - Test the staging model")
        print("   - Use CLI to promote to production when ready:")
        print(f"     python scripts/model_registry_cli.py promote bert_fake_review_detector")
        print("=" * 70)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n  Training interrupted by user")
        if mlflow.active_run():
            mlflow.end_run()
    except Exception as e:
        logger.exception(f" Error: {e}")
        if mlflow.active_run():
            mlflow.end_run()