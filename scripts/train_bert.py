"""
Train BERT model for fake review detection
"""

import sys

sys.path.append('.')

from models.bert.dataset import prepare_datasets_from_csv
from models.bert.trainer import BERTFakeReviewTrainer
from loguru import logger
import argparse


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

    args = parser.parse_args()

    # Setup logging
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    logger.add("logs/bert_training.log", rotation="10 MB")

    print("\n" + "=" * 70)
    print("🤖 BERT FINE-TUNING FOR FAKE REVIEW DETECTION")
    print("=" * 70)
    print(f"   Model: {args.model}")
    print(f"   Data: {args.data}")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Learning rate: {args.lr}")
    print("=" * 70)

    # Prepare datasets
    logger.info("\n📦 Preparing datasets...")

    train_dataset, test_dataset, label_mapping = prepare_datasets_from_csv(
        csv_path=args.data,
        text_column='text_',
        label_column='label',
        test_size=0.2,
        tokenizer_name=args.model,
        max_length=512
    )

    logger.info(f"✓ Label mapping: {label_mapping}")

    # Initialize trainer
    logger.info("\n🤖 Initializing BERT trainer...")

    trainer = BERTFakeReviewTrainer(
        model_name=args.model,
        num_labels=2,
        output_dir='models/saved_models/bert_fake_review'
    )

    # Train
    logger.info("\n🚀 Starting training...\n")

    train_result = trainer.train(
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )

    # Evaluate
    logger.info("\n📊 Final evaluation...\n")

    eval_results = trainer.evaluate(test_dataset)

    # Save model
    logger.info("\n💾 Saving model...\n")

    trainer.save_model()

    # Print summary
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE!")
    print("=" * 70)
    print(f"   Accuracy: {eval_results['eval_accuracy']:.2%}")
    print(f"   F1 Score: {eval_results['eval_f1']:.4f}")
    print(f"   Precision: {eval_results['eval_precision']:.4f}")
    print(f"   Recall: {eval_results['eval_recall']:.4f}")
    print("=" * 70)
    print(f"\n📁 Model saved to: models/saved_models/bert_fake_review/final_model")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
    except Exception as e:
        logger.exception(f"❌ Error: {e}")
