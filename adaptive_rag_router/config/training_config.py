# File: adaptive_rag_router/config/training_config.py
"""
Unified configuration for both GitHub and Kaggle
"""

from dataclasses import dataclass
from typing import List, Dict, Any
import os

@dataclass
class TrainingConfig:
    # Data
    dataset_name: str = "clinc_oos"
    max_length: int = 128
    batch_size: int = 32
    eval_batch_size: int = 64
    
    # Training
    learning_rate: float = 2e-4
    num_epochs: int = 10
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    logging_steps: int = 50
    eval_steps: int = 200
    save_steps: int = 500
    
    # LoRA
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = None
    
    # Environment
    output_dir: str = "./model_checkpoints"
    use_wandb: bool = False
    seed: int = 42
    
    def __post_init__(self):
        if self.lora_target_modules is None:
            self.lora_target_modules = ["query", "key", "value", "dense"]
        
        # Auto-detect environment
        self.is_kaggle = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ
        self.is_colab = 'COLAB_GPU' in os.environ
        
        # Adjust for cloud environments
        if self.is_kaggle or self.is_colab:
            self.batch_size = 16  # Smaller batches for free tiers
            self.num_epochs = 5   # Fewer epochs for quick results

# Model configurations
MODEL_CONFIGS = {
    "distilbert": {
        "model_name": "distilbert-base-uncased",
        "lora_rank": 8,
        "learning_rate": 3e-4,
    },
    "roberta": {
        "model_name": "roberta-base", 
        "lora_rank": 16,
        "learning_rate": 2e-4,
    },
    "deberta": {
        "model_name": "microsoft/deberta-v3-base",
        "lora_rank": 16, 
        "learning_rate": 2e-4,
    }
}