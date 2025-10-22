"""
Training configuration for Adaptive RAG Router
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class ModelConfig:
    model_name: str
    num_labels: int = 10  # CLINC150 domains
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1

@dataclass
class LoRAConfig:
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = None
    bias: str = "none"
    task_type: str = "SEQ_CLS"

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["query", "key", "value", "dense"]

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
    
    # Optimization
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "cosine"
    
    # Experimental
    use_amp: bool = True
    seed: int = 42

@dataclass
class BenchmarkConfig:
    n_samples: int = 1000
    temperature: float = 0.0
    max_tokens: int = 50
    cost_tracking: bool = True

# Default configurations
DEFAULT_LORA_CONFIGS = {
    "roberta-base": LoRAConfig(r=16, lora_alpha=32),
    "distilbert-base-uncased": LoRAConfig(r=8, lora_alpha=16),
    "microsoft/deberta-v3-base": LoRAConfig(r=16, lora_alpha=32),
}

MODEL_MAPPING = {
    "roberta": "roberta-base",
    "distilbert": "distilbert-base-uncased", 
    "deberta": "microsoft/deberta-v3-base"
}