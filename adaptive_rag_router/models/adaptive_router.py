# File: adaptive_rag_router/models/adaptive_router.py
"""
Production-ready model with unified interface
"""

import torch
import torch.nn as nn
from transformers import (
    AutoModelForSequenceClassification, 
    AutoConfig, 
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from peft import get_peft_model, LoraConfig, TaskType
from typing import Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)

class AdaptiveRAGRouter:
    """Production-ready Adaptive RAG Router"""
    
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        num_domains: int = 10,
        lora_config: Optional[Dict] = None,
        device: Optional[str] = None
    ):
        self.model_name = model_name
        self.num_domains = num_domains
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Default LoRA config
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
        """Initialize model with proper error handling"""
        try:
            # Load configuration
            config = AutoConfig.from_pretrained(
                self.model_name,
                num_labels=self.num_domains,
                hidden_dropout_prob=0.1,
                attention_probs_dropout_prob=0.1
            )
            
            # Load model
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                config=config,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            
            # Apply LoRA
            peft_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                **self.lora_config
            )
            
            self.model = get_peft_model(self.model, peft_config)
            self.model.to(self.device)
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token or '<pad>'
            
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.model.parameters())
            
            logger.info(f"Initialized {self.model_name} on {self.device}")
            logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
            
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise
    
    def train(
        self,
        train_loader,
        val_loader,
        output_dir: str = "./model_checkpoints",
        training_config: Optional[Dict] = None
    ):
        """Train the model with production-ready configuration"""
        if training_config is None:
            training_config = {
                "num_epochs": 5,
                "learning_rate": 2e-4,
                "per_device_train_batch_size": 16,
            }
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=training_config.get("num_epochs", 5),
            learning_rate=training_config.get("learning_rate", 2e-4),
            per_device_train_batch_size=training_config.get("per_device_train_batch_size", 16),
            per_device_eval_batch_size=training_config.get("per_device_eval_batch_size", 32),
            warmup_ratio=0.1,
            weight_decay=0.01,
            logging_dir=f"{output_dir}/logs",
            logging_steps=50,
            eval_steps=100,
            evaluation_strategy="steps",
            save_steps=200,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            report_to=None,
            dataloader_pin_memory=False,
            fp16=torch.cuda.is_available(),
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_loader.dataset,
            eval_dataset=val_loader.dataset,
            compute_metrics=self._compute_metrics,
        )
        
        logger.info("Starting training...")
        trainer.train()
        
        # Save the model
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        logger.info(f"Training completed. Model saved to {output_dir}")
        
        return trainer
    
    def predict(self, texts: Union[str, List[str]], batch_size: int = 32) -> Dict:
        """Production-ready prediction with batching"""
        if isinstance(texts, str):
            texts = [texts]
        
        self.model.eval()
        
        predictions = []
        probabilities = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                inputs = self.tokenizer(
                    batch_texts,
                    truncation=True,
                    padding=True,
                    max_length=128,
                    return_tensors="pt"
                ).to(self.device)
                
                outputs = self.model(**inputs)
                batch_probs = torch.softmax(outputs.logits, dim=-1)
                batch_preds = torch.argmax(batch_probs, dim=-1)
                
                predictions.extend(batch_preds.cpu().numpy())
                probabilities.extend(batch_probs.cpu().numpy())
        
        from adaptive_rag_router.data.data_loader import CLINC150DataLoader
        domain_names = [CLINC150DataLoader.DOMAIN_NAMES[pred] for pred in predictions]
        
        return {
            "domains": domain_names,
            "predictions": predictions,
            "probabilities": probabilities,
            "confidences": [max(probs) for probs in probabilities]
        }
    
    def _compute_metrics(self, eval_pred):
        """Compute metrics for evaluation"""
        import numpy as np
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        return {
            "accuracy": accuracy_score(labels, predictions),
            "f1": f1_score(labels, predictions, average="weighted"),
            "precision": precision_score(labels, predictions, average="weighted"),
            "recall": recall_score(labels, predictions, average="weighted"),
        }
    
    def save(self, path: str):
        """Save model and tokenizer"""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model and tokenizer"""
        from peft import PeftModel
        
        self.model = PeftModel.from_pretrained(self.model, path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model.to(self.device)
        logger.info(f"Model loaded from {path}")

# Factory function
def create_router_model(
    model_type: str = "distilbert",
    lora_rank: int = 16,
    device: Optional[str] = None
) -> AdaptiveRAGRouter:
    """Factory function to create router models"""
    
    model_mapping = {
        "distilbert": "distilbert-base-uncased",
        "roberta": "roberta-base",
        "deberta": "microsoft/deberta-v3-base"
    }
    
    if model_type not in model_mapping:
        raise ValueError(f"Model type {model_type} not supported. Choose from {list(model_mapping.keys())}")
    
    lora_config = {
        "r": lora_rank,
        "lora_alpha": lora_rank * 2,
        "lora_dropout": 0.05,
        "target_modules": ["query", "key", "value", "dense"],
        "bias": "none"
    }
    
    return AdaptiveRAGRouter(
        model_name=model_mapping[model_type],
        lora_config=lora_config,
        device=device
    )