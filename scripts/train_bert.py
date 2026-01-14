#!/usr/bin/env python3
"""
Script pour entraîner le modèle BERT pour la détection de faux avis
"""

import sys
import os
import argparse
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from nlp.training.bert_trainer import BERTFakeReviewTrainer
from nlp.dataset import prepare_datasets_from_csv
from loguru import logger

def main():
    """Fonction principale d'entraînement"""
    
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
    parser.add_argument('--output', type=str, default='models/saved_models/bert_fake_review',
                        help='Output directory')
    
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
    print(f"   Output: {args.output}")
    print("=" * 70)
    
    # Check if data exists
    if not os.path.exists(args.data):
        print(f"\nERROR: Data file not found: {args.data}")
        print("\nYou need a dataset to train BERT. Options:")
        print("1. Download from Kaggle: https://www.kaggle.com/datasets/...")
        print("2. Use the existing reviews in data/raw/reviews/")
        print("3. Create a simple dataset for testing")
        
        # Create a simple test dataset
        print("\nCreating a simple test dataset...")
        os.makedirs('data/raw/reviews', exist_ok=True)
        
        import pandas as pd
        test_data = pd.DataFrame({
            'text_': [
                "This product is amazing! Best purchase ever!",
                "Terrible quality, waste of money",
                "Good product but expensive",
                "Excellent value for money",
                "Poor quality, disappointed"
            ],
            'label': [0, 1, 0, 0, 1]  # 0 = real, 1 = fake
        })
        test_data.to_csv(args.data, index=False)
        print(f"Created test dataset: {args.data}")
    
    # Prepare datasets
    logger.info("\n Preparing datasets...")
    
    try:
        train_dataset, test_dataset, label_mapping = prepare_datasets_from_csv(
            csv_path=args.data,
            text_column='text_',
            label_column='label',
            test_size=0.2,
            tokenizer_name=args.model,
            max_length=512
        )
        
        logger.info(f" Label mapping: {label_mapping}")
        logger.info(f" Train samples: {len(train_dataset)}")
        logger.info(f" Test samples: {len(test_dataset)}")
        
    except Exception as e:
        logger.error(f"Failed to prepare datasets: {e}")
        print(f"\nERROR: Could not load dataset")
        print("Please ensure your CSV has columns: 'text_' and 'label'")
        return
    
    # Initialize trainer
    logger.info("\n Initializing BERT trainer...")
    
    trainer = BERTFakeReviewTrainer(
        model_name=args.model,
        num_labels=2,
        output_dir=args.output
    )
    
    # Train
    logger.info("\n Starting training...\n")
    
    try:
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
        
        # Save model
        logger.info("\n Saving model...\n")
        
        trainer.save_model()
        
        # Print summary
        print("\n" + "=" * 70)
        print(" TRAINING COMPLETE!")
        print("=" * 70)
        print(f"   Accuracy: {eval_results['eval_accuracy']:.2%}")
        print(f"   F1 Score: {eval_results['eval_f1']:.4f}")
        print(f"   Precision: {eval_results['eval_precision']:.4f}")
        print(f"   Recall: {eval_results['eval_recall']:.4f}")
        print("=" * 70)
        print(f"\n Model saved to: {args.output}/final_model")
        print("=" * 70)
        print("\nNext steps:")
        print("   - Test the model with: python nlp/prediction.py")
        print("   - The model will be automatically detected by the pipeline")
        print("=" * 70)
        
    except Exception as e:
        logger.exception(f"Training failed: {e}")
        print(f"\nTraining failed: {e}")
        return

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    except Exception as e:
        print(f"\nError: {e}")