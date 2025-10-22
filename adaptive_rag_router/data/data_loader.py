# File: adaptive_rag_router/data/data_loader.py
"""
Production-ready data loader with environment detection
"""

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
    
    def load_dataset(self, split: str = "train", sample_size: Optional[int] = None) -> Dataset:
        """Load dataset with optional sampling for quick experiments"""
        try:
            dataset = load_dataset("clinc_oos", split=split)
        except Exception as e:
            logger.warning(f"Failed to load clinc_oos: {e}")
            # Fallback - you might want to implement a custom dataset loader
            raise
        
        if sample_size and len(dataset) > sample_size:
            dataset = dataset.select(range(sample_size))
            logger.info(f"Using sampled dataset with {sample_size} examples")
        
        return dataset
    
    def extract_domain_from_intent(self, intent: str) -> str:
        """Extract domain from intent name"""
        if intent == 'oos':
            return 'meta'
        domain = intent.split('_')[0]
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
        train_dataset = train_dataset.remove_columns(columns_to_remove)
        val_dataset = val_dataset.remove_columns(columns_to_remove)
        test_dataset = test_dataset.remove_columns(columns_to_remove)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        logger.info(f"Created data loaders - Train: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}")
        
        return train_loader, val_loader, test_loader
    
    def get_quick_demo_data(self, num_samples: int = 100):
        """Get small dataset for quick demos"""
        return self.get_data_loaders(batch_size=8, sample_size=num_samples)