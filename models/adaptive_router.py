# File: models/adaptive_router.py
"""
Adaptive RAG Router model with PEFT/LoRA
"""

import torch
import torch.nn as nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoConfig,
    TrainingArguments,
    Trainer
)
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    PeftModel
)
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class AdaptiveRAGRouter:
    """Adaptive RAG Router with PEFT-based intent classification"""
    
    def __init__(
        self,
        model_name: str = "roberta-base",
        num_domains: int = 10,
        lora_config: Optional[Dict] = None,
        device: str = None
    ):
        self.model_name = model_name
        self.num_domains = num_domains
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Default LoRA configuration
        if lora_config is None:
            lora_config = {
                "r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.05,
                "target_modules": ["query", "key", "value", "dense"],
                "bias": "none"
            }
        
        self.lora_config = lora_config
        self.model = None
        self.tokenizer = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the model with LoRA configuration"""
        try:
            # Load model configuration
            config = AutoConfig.from_pretrained(
                self.model_name,
                num_labels=self.num_domains,
                hidden_dropout_prob=0.1,
                attention_probs_dropout_prob=0.1
            )
            
            # Load base model
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                config=config
            )
            
            # Initialize LoRA
            peft_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                **self.lora_config
            )
            
            self.model = get_peft_model(self.model, peft_config)
            self.model.to(self.device)
            
            # Load tokenizer
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token or '<pad>'
            
            logger.info(f"Initialized {self.model_name} with LoRA")
            logger.info(f"Trainable parameters: {self.model.print_trainable_parameters()}")
            
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            raise
    
    def train(
        self,
        train_loader,
        val_loader,
        output_dir: str = "./results",
        num_epochs: int = 10,
        learning_rate: float = 2e-4,
        **training_kwargs
    ):
        """Train the model"""
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=64,
            warmup_ratio=0.1,
            weight_decay=0.01,
            logging_dir=f"{output_dir}/logs",
            logging_steps=50,
            eval_steps=200,
            evaluation_strategy="steps",
            save_steps=500,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            report_to=None,  # Disable wandb by default
            **training_kwargs
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_loader.dataset,
            eval_dataset=val_loader.dataset,
            compute_metrics=self._compute_metrics,
        )
        
        # Train model
        trainer.train()
        
        # Save model
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        return trainer
    
    def predict(self, texts: list, batch_size: int = 32) -> Dict[str, list]:
        """Predict domains for given texts"""
        self.model.eval()
        
        predictions = []
        probabilities = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=128,
                return_tensors="pt"
            ).to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                batch_probs = torch.softmax(outputs.logits, dim=-1)
                batch_preds = torch.argmax(batch_probs, dim=-1)
            
            predictions.extend(batch_preds.cpu().numpy())
            probabilities.extend(batch_probs.cpu().numpy())
        
        # Convert to domain names
        from data.data_loader import CLINC150DataLoader
        domain_names = [CLINC150DataLoader.DOMAIN_NAMES[pred] for pred in predictions]
        
        return {
            "domains": domain_names,
            "predictions": predictions,
            "probabilities": probabilities
        }
    
    def _compute_metrics(self, eval_pred):
        """Compute metrics for evaluation"""
        import numpy as np
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        accuracy = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average="weighted")
        precision = precision_score(labels, predictions, average="weighted")
        recall = recall_score(labels, predictions, average="weighted")
        
        return {
            "accuracy": accuracy,
            "f1": f1,
            "precision": precision,
            "recall": recall
        }
    
    def save(self, path: str):
        """Save model and tokenizer"""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
    
    def load(self, path: str):
        """Load model and tokenizer"""
        from peft import PeftModel
        
        self.model = PeftModel.from_pretrained(self.model, path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)

# Model factory function
def create_router_model(
    model_type: str = "roberta",
    lora_rank: int = 16,
    device: str = None
) -> AdaptiveRAGRouter:
    """Factory function to create router models"""
    
    model_mapping = {
        "roberta": "roberta-base",
        "distilbert": "distilbert-base-uncased",
        "deberta": "microsoft/deberta-v3-base"
    }
    
    if model_type not in model_mapping:
        raise ValueError(f"Model type {model_type} not supported. Choose from {list(model_mapping.keys())}")
    
    lora_config = {
        "r": lora_rank,
        "lora_alpha": lora_rank * 2,  # Standard practice
        "lora_dropout": 0.05,
        "target_modules": ["query", "key", "value", "dense"],
        "bias": "none"
    }
    
    return AdaptiveRAGRouter(
        model_name=model_mapping[model_type],
        lora_config=lora_config,
        device=device
    )