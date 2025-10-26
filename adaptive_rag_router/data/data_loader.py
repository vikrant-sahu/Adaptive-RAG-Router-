# File: adaptive_rag_router/data/data_loader.py
"""
Production-ready data loader with environment detection
"""

import os  # ADDED: Missing import
import logging
from typing import Dict, Tuple, Optional
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from collections import Counter

logger = logging.getLogger(__name__)

class CLINC150DataLoader:
    """Production data loader for CLINC150 with cloud optimization"""

    DOMAIN_MAPPING = {
        'banking': 0, 'credit_cards': 1, 'work': 2, 'travel': 3, 'utility': 4,
        'auto_&_commute': 5, 'home': 6, 'kitchen_&_dining': 7, 'small_talk': 8, 'meta': 9
    }

    DOMAIN_NAMES = list(DOMAIN_MAPPING.keys())

    def __init__(self, model_name: str = "distilbert-base-uncased", max_length: int = 128):
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = self._load_tokenizer()
        self.intent_names = None  # Will be populated when dataset is loaded
        
    def _load_tokenizer(self):
        """Load tokenizer with proper error handling"""
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token or '<pad>'
            return tokenizer
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise
    
    def load_dataset(self, split: str = "train", sample_size: Optional[int] = None):
        """
        Load CLINC150 dataset with Kaggle optimization

        Added cache_dir for Kaggle/Colab optimization
        """
        # Use Kaggle's working directory for caching
        cache_dir = None
        if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:  # FIXED: Correct env var
            cache_dir = "/kaggle/working/hf_cache"
            os.makedirs(cache_dir, exist_ok=True)
        elif 'COLAB_GPU' in os.environ:
            cache_dir = "/content/hf_cache"
            os.makedirs(cache_dir, exist_ok=True)

        dataset = load_dataset("clinc_oos", "plus", split=split, cache_dir=cache_dir)

        # Extract intent names from dataset features if available
        if self.intent_names is None and 'intent' in dataset.features:
            if hasattr(dataset.features['intent'], 'names'):
                self.intent_names = dataset.features['intent'].names
                logger.info(f"Loaded {len(self.intent_names)} intent names from dataset")

        if sample_size:
            dataset = dataset.select(range(min(sample_size, len(dataset))))

        return dataset
    
    def extract_domain_from_intent(self, intent) -> str:
        """Extract domain from intent name or index"""
        # Handle integer intent (ClassLabel index)
        if isinstance(intent, int):
            if self.intent_names and intent < len(self.intent_names):
                intent = self.intent_names[intent]
            else:
                logger.warning(f"Intent index {intent} out of range, defaulting to 'meta'")
                return 'meta'

        # Handle string intent
        if intent == 'oos':
            return 'meta'

        # Extract domain from intent name (e.g., "banking_balance" -> "banking")
        domain = intent.split('_')[0] if '_' in intent else intent
        return domain if domain in self.DOMAIN_MAPPING else 'meta'
    
    def preprocess_function(self, examples: Dict) -> Dict:
        """Tokenize and preprocess examples"""
        domains = [self.extract_domain_from_intent(intent) for intent in examples['intent']]
        domain_labels = [self.DOMAIN_MAPPING[domain] for domain in domains]
        
        tokenized = self.tokenizer(
            examples['text'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors=None
        )
        
        return {
            **tokenized,
            'labels': domain_labels,
            'domain': domains,
            'original_intent': examples['intent']
        }
    
    def get_data_loaders(self, batch_size: int = 32, sample_size: Optional[int] = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Get train, validation, and test data loaders"""
        # Load and process datasets
        train_dataset = self.load_dataset("train", sample_size)
        val_dataset = self.load_dataset("validation", sample_size)
        test_dataset = self.load_dataset("test", sample_size)
        
        # Preprocess
        train_dataset = train_dataset.map(self.preprocess_function, batched=True)
        val_dataset = val_dataset.map(self.preprocess_function, batched=True) 
        test_dataset = test_dataset.map(self.preprocess_function, batched=True)
        
        # Remove original columns to save memory
        columns_to_remove = ['text', 'intent']
        # FIXED: Check if columns exist before removing
        for col in columns_to_remove:
            if col in train_dataset.column_names:
                train_dataset = train_dataset.remove_columns([col])
            if col in val_dataset.column_names:
                val_dataset = val_dataset.remove_columns([col])
            if col in test_dataset.column_names:
                test_dataset = test_dataset.remove_columns([col])
        
        # Set format for PyTorch - ADDED for compatibility
        train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        logger.info(f"Created data loaders - Train: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}")
        
        return train_loader, val_loader, test_loader
    
    def get_quick_demo_data(self, num_samples: int = 100):
        """Get small dataset for quick demos"""
        return self.get_data_loaders(batch_size=8, sample_size=num_samples)

    def _stratified_split_indices(self, labels: list, split_ratio: float, seed: int = 42) -> Tuple[list, list]:
        """
        Perform stratified split on dataset indices based on labels.

        Args:
            labels: List of class labels
            split_ratio: Ratio for first split (e.g., 0.7 means 70% first, 30% second)
            seed: Random seed for reproducibility

        Returns:
            Tuple of (first_split_indices, second_split_indices)
        """
        from sklearn.model_selection import train_test_split

        indices = list(range(len(labels)))
        first_indices, second_indices = train_test_split(
            indices,
            train_size=split_ratio,
            stratify=labels,
            random_state=seed
        )

        return first_indices, second_indices

    def _log_class_distribution(self, dataset, split_name: str):
        """Log the distribution of domain classes in a dataset split."""
        # Extract domain labels
        domains = [self.extract_domain_from_intent(item['intent']) for item in dataset]
        domain_labels = [self.DOMAIN_MAPPING[domain] for domain in domains]

        # Count occurrences
        label_counts = Counter(domain_labels)

        logger.info(f"\n{split_name} class distribution:")
        total = len(domain_labels)
        for label_idx in sorted(label_counts.keys()):
            domain_name = self.DOMAIN_NAMES[label_idx]
            count = label_counts[label_idx]
            percentage = (count / total) * 100
            logger.info(f"  {domain_name:20s}: {count:5d} samples ({percentage:5.2f}%)")

        # Check if all 10 classes are present
        if len(label_counts) == 10:
            logger.info(f"  ✓ All 10 domain classes present in {split_name}")
        else:
            missing = set(range(10)) - set(label_counts.keys())
            missing_names = [self.DOMAIN_NAMES[i] for i in missing]
            logger.warning(f"  ⚠ Missing classes in {split_name}: {missing_names}")

    def get_custom_split_loaders(
        self,
        batch_size: int = 32,
        train_val_ratio: float = 0.7,
        val_split: float = 0.15,
        seed: int = 42
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Get data loaders with custom train/validation/test split using stratified sampling.

        Ensures all 10 domain classes are proportionally represented in each split.

        Args:
            batch_size: Batch size for data loaders
            train_val_ratio: Ratio of data to use for train+validation (default: 0.7 for 70%)
            val_split: Ratio of train_val data to use for validation (default: 0.15)
            seed: Random seed for reproducibility

        Returns:
            Tuple of (train_loader, val_loader, test_loader)

        Example:
            With train_val_ratio=0.7 and val_split=0.15:
            - 70% of data is used for training and validation
            - Of that 70%, 15% is used for validation (10.5% of total)
            - Of that 70%, 85% is used for training (59.5% of total)
            - Remaining 30% is used for testing

            Each split maintains proportional representation of all 10 domain classes.
        """
        from datasets import concatenate_datasets

        logger.info(f"Creating stratified custom split with {train_val_ratio*100:.0f}% train+val, {(1-train_val_ratio)*100:.0f}% test")

        # Load all available splits and concatenate them
        train_data = self.load_dataset("train")
        val_data = self.load_dataset("validation")
        test_data = self.load_dataset("test")

        # Concatenate all data
        full_dataset = concatenate_datasets([train_data, val_data, test_data])

        # Extract domain labels for stratification
        logger.info("Extracting domain labels for stratified sampling...")
        domains = [self.extract_domain_from_intent(item['intent']) for item in full_dataset]
        domain_labels = [self.DOMAIN_MAPPING[domain] for domain in domains]

        # Log original distribution
        logger.info(f"\nTotal dataset size: {len(full_dataset)} samples")
        original_counts = Counter(domain_labels)
        for label_idx in sorted(original_counts.keys()):
            domain_name = self.DOMAIN_NAMES[label_idx]
            count = original_counts[label_idx]
            logger.info(f"  {domain_name:20s}: {count:5d} samples")

        # Stratified split: first split into train+val and test
        train_val_indices, test_indices = self._stratified_split_indices(
            domain_labels, train_val_ratio, seed
        )

        train_val_dataset = full_dataset.select(train_val_indices)
        test_dataset = full_dataset.select(test_indices)

        # Extract labels for train_val split
        train_val_domains = [self.extract_domain_from_intent(item['intent']) for item in train_val_dataset]
        train_val_labels = [self.DOMAIN_MAPPING[domain] for domain in train_val_domains]

        # Stratified split: further split train_val into train and validation
        train_indices_local, val_indices_local = self._stratified_split_indices(
            train_val_labels, 1 - val_split, seed
        )

        train_dataset = train_val_dataset.select(train_indices_local)
        val_dataset = train_val_dataset.select(val_indices_local)

        total_size = len(full_dataset)
        logger.info(f"\n✓ Stratified split completed:")
        logger.info(f"  Train: {len(train_dataset):6d} samples ({len(train_dataset)/total_size*100:.1f}%)")
        logger.info(f"  Val:   {len(val_dataset):6d} samples ({len(val_dataset)/total_size*100:.1f}%)")
        logger.info(f"  Test:  {len(test_dataset):6d} samples ({len(test_dataset)/total_size*100:.1f}%)")

        # Log class distributions for verification
        self._log_class_distribution(train_dataset, "TRAIN")
        self._log_class_distribution(val_dataset, "VALIDATION")
        self._log_class_distribution(test_dataset, "TEST")

        # Preprocess datasets
        train_dataset = train_dataset.map(self.preprocess_function, batched=True)
        val_dataset = val_dataset.map(self.preprocess_function, batched=True)
        test_dataset = test_dataset.map(self.preprocess_function, batched=True)

        # Remove original columns to save memory
        columns_to_remove = ['text', 'intent']
        for col in columns_to_remove:
            if col in train_dataset.column_names:
                train_dataset = train_dataset.remove_columns([col])
            if col in val_dataset.column_names:
                val_dataset = val_dataset.remove_columns([col])
            if col in test_dataset.column_names:
                test_dataset = test_dataset.remove_columns([col])

        # Set format for PyTorch
        train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, test_loader