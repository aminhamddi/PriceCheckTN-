"""
Dataset preparation for BERT training
"""

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from loguru import logger
import numpy as np


class FakeReviewDataset(Dataset):
    """Dataset for fake review detection"""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def prepare_datasets_from_csv(
    csv_path,
    text_column='text_',
    label_column='label',
    test_size=0.2,
    tokenizer_name='distilbert-base-multilingual-cased',
    max_length=512,
    random_state=42
):
    """
    Prepare train and test datasets from CSV
    
    Args:
        csv_path: Path to CSV file
        text_column: Column name for text
        label_column: Column name for labels
        test_size: Proportion for test set
        tokenizer_name: HuggingFace model name
        max_length: Maximum sequence length
        random_state: Random seed
    
    Returns:
        train_dataset, test_dataset, label_mapping
    """
    
    logger.info(f"Loading dataset from {csv_path}")
    
    # Load data
    df = pd.read_csv(csv_path)
    
    # Validate columns
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in CSV. Available: {list(df.columns)}")
    
    if label_column not in df.columns:
        raise ValueError(f"Column '{label_column}' not found in CSV. Available: {list(df.columns)}")
    
    # Clean data
    df = df.dropna(subset=[text_column, label_column])
    df = df[df[text_column].astype(str).str.strip() != '']
    
    logger.info(f"Loaded {len(df)} samples")
    
    # Get unique labels
    unique_labels = sorted(df[label_column].unique())
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
    
    logger.info(f"Label mapping: {label_mapping}")
    
    # Map labels
    df['label_encoded'] = df[label_column].map(label_mapping)
    
    # Handle small datasets
    if len(df) < 10:
        logger.warning(f"Small dataset detected ({len(df)} samples). Using all data for training.")
        # Use all data for training, create minimal test set
        train_texts = df[text_column].values
        train_labels = df['label_encoded'].values
        test_texts = df[text_column].values[:2]  # Just 2 samples for test
        test_labels = df['label_encoded'].values[:2]
    else:
        # Split data
        min_test_size = max(1, int(len(df) * test_size))
        actual_test_size = min(min_test_size, len(df) - 2)
        
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            df[text_column].values,
            df['label_encoded'].values,
            test_size=actual_test_size,
            random_state=random_state,
            stratify=df['label_encoded'].values if len(df['label_encoded'].unique()) > 1 else None
        )
    
    logger.info(f"Train samples: {len(train_texts)}")
    logger.info(f"Test samples: {len(test_texts)}")
    
    # Initialize tokenizer
    logger.info(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Create datasets
    train_dataset = FakeReviewDataset(
        texts=train_texts,
        labels=train_labels,
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    test_dataset = FakeReviewDataset(
        texts=test_texts,
        labels=test_labels,
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    return train_dataset, test_dataset, label_mapping


def create_simple_dataset():
    """Create a simple dataset for testing"""
    
    import os
    os.makedirs('data/raw/reviews', exist_ok=True)
    
    # Simple dataset with more samples
    data = {
        'text_': [
            "This product is amazing! Best purchase ever!",
            "Terrible quality, waste of money",
            "Good product but expensive",
            "Excellent value for money",
            "Poor quality, disappointed",
            "Highly recommended!",
            "Don't buy this, it's a scam",
            "Great features and works perfectly",
            "Not worth the price",
            "Perfect for my needs",
            "Amazing quality, exceeded expectations",
            "Very disappointed, broke after one day",
            "Good value for the price",
            "Outstanding product, love it!",
            "Worst purchase ever, avoid at all costs"
        ],
        'label': [0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1]  # 0 = real, 1 = fake
    }
    
    df = pd.DataFrame(data)
    df.to_csv('data/raw/reviews/fake_reviews.csv', index=False)
    
    print("✅ Created simple dataset: data/raw/reviews/fake_reviews.csv")
    print(f"   Samples: {len(df)}")
    print(f"   Real reviews: {sum(df['label'] == 0)}")
    print(f"   Fake reviews: {sum(df['label'] == 1)}")
    
    return df


if __name__ == "__main__":
    # Test dataset creation
    print("Testing dataset preparation...")
    
    # Create simple dataset if it doesn't exist
    if not os.path.exists('data/raw/reviews/fake_reviews.csv'):
        create_simple_dataset()
    
    # Test loading
    try:
        train, test, mapping = prepare_datasets_from_csv(
            'data/raw/reviews/fake_reviews.csv'
        )
        print(f"\n✅ Dataset ready!")
        print(f"   Train: {len(train)} samples")
        print(f"   Test: {len(test)} samples")
        print(f"   Mapping: {mapping}")
    except Exception as e:
        print(f"❌ Error: {e}")