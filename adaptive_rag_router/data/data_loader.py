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

    def get_custom_split_loaders(
        self,
        batch_size: int = 32,
        train_val_ratio: float = 0.7,
        val_split: float = 0.15,
        seed: int = 42
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Get data loaders with custom train/validation/test split.

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
        """
        from datasets import concatenate_datasets

        logger.info(f"Creating custom split with {train_val_ratio*100:.0f}% train+val, {(1-train_val_ratio)*100:.0f}% test")

        # Load all available splits and concatenate them
        train_data = self.load_dataset("train")
        val_data = self.load_dataset("validation")
        test_data = self.load_dataset("test")

        # Concatenate all data
        full_dataset = concatenate_datasets([train_data, val_data, test_data])

        # Shuffle the full dataset
        full_dataset = full_dataset.shuffle(seed=seed)

        # Calculate split sizes
        total_size = len(full_dataset)
        train_val_size = int(total_size * train_val_ratio)

        # Split into train+val and test
        train_val_dataset = full_dataset.select(range(train_val_size))
        test_dataset = full_dataset.select(range(train_val_size, total_size))

        # Further split train_val into train and validation
        train_val_size_actual = len(train_val_dataset)
        val_size = int(train_val_size_actual * val_split)
        train_size = train_val_size_actual - val_size

        train_dataset = train_val_dataset.select(range(train_size))
        val_dataset = train_val_dataset.select(range(train_size, train_val_size_actual))

        logger.info(f"Split sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        logger.info(f"Percentages - Train: {len(train_dataset)/total_size*100:.1f}%, Val: {len(val_dataset)/total_size*100:.1f}%, Test: {len(test_dataset)/total_size*100:.1f}%")

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