"""
Dataset preparation for BERT fine-tuning
"""

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import pandas as pd
from typing import List, Dict
from loguru import logger


class FakeReviewDataset(Dataset):
    """
    PyTorch Dataset for fake review detection
    Compatible with Hugging Face Transformers
    """

    def __init__(
            self,
            texts: List[str],
            labels: List[int],
            tokenizer_name: str = 'distilbert-base-multilingual-cased',
            max_length: int = 512
    ):
        """
        Args:
            texts: List of review texts
            labels: List of labels (0 = real, 1 = fake)
            tokenizer_name: Pretrained tokenizer to use
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

        logger.info(f"✓ Initialized dataset: {len(texts)} samples")
        logger.info(f"✓ Using tokenizer: {tokenizer_name}")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        """Get a single tokenized sample"""

        text = str(self.texts[idx])
        label = int(self.labels[idx])

        # Tokenize
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def prepare_datasets_from_csv(
        csv_path: str,
        text_column: str = 'text_',
        label_column: str = 'label',
        test_size: float = 0.2,
        tokenizer_name: str = 'distilbert-base-multilingual-cased',
        max_length: int = 512
):
 
    from sklearn.model_selection import train_test_split

    logger.info(f"📥 Loading data from {csv_path}")

    # Load data
    df = pd.read_csv(csv_path)

    logger.info(f"✓ Loaded {len(df)} reviews")
    logger.info(f"✓ Columns: {list(df.columns)}")

    # Get texts and labels
    texts = df[text_column].astype(str).tolist()

    # Convert labels to binary (0/1)
    if label_column in df.columns:
        # Assuming labels are 'CG' (fake) and 'OR' (real)
        # or similar string labels
        unique_labels = df[label_column].unique()
        logger.info(f"✓ Unique labels: {unique_labels}")

        # Create label mapping
        if df[label_column].dtype == 'object':
            # String labels - create mapping
            label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
            labels = df[label_column].map(label_mapping).tolist()
            logger.info(f"✓ Label mapping: {label_mapping}")
        else:
            # Already numeric
            labels = df[label_column].tolist()
            label_mapping = {i: i for i in unique_labels}
    else:
        raise ValueError(f"Label column '{label_column}' not found!")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels,
        test_size=test_size,
        random_state=42,
        stratify=labels
    )

    logger.info(f"✓ Train size: {len(X_train)}")
    logger.info(f"✓ Test size: {len(X_test)}")

    # Create datasets
    train_dataset = FakeReviewDataset(
        X_train, y_train,
        tokenizer_name=tokenizer_name,
        max_length=max_length
    )

    test_dataset = FakeReviewDataset(
        X_test, y_test,
        tokenizer_name=tokenizer_name,
        max_length=max_length
    )

    return train_dataset, test_dataset, label_mapping


def prepare_datasets_from_dataframe(
        df: pd.DataFrame,
        text_column: str = 'text',
        label_column: str = 'label',
        test_size: float = 0.2,
        tokenizer_name: str = 'distilbert-base-multilingual-cased'
):
    """
    Prepare datasets directly from pandas DataFrame
    """
    from sklearn.model_selection import train_test_split

    texts = df[text_column].astype(str).tolist()
    labels = df[label_column].tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels,
        test_size=test_size,
        random_state=42,
        stratify=labels
    )

    train_dataset = FakeReviewDataset(X_train, y_train, tokenizer_name=tokenizer_name)
    test_dataset = FakeReviewDataset(X_test, y_test, tokenizer_name=tokenizer_name)

    return train_dataset, test_dataset