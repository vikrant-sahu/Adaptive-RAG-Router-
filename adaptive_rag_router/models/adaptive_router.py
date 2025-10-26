# File: adaptive_rag_router/models/adaptive_router.py
"""
Production-ready Adaptive RAG Router
CLEAN VERSION - Tested on Kaggle GPU
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
import os

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
        
        # Detect environment
        self.is_kaggle = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ
        self.is_colab = 'COLAB_GPU' in os.environ
        
        # Default LoRA config
        if lora_config is None:
            lora_config = {
                "r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.05,
                "target_modules": ["q_lin", "k_lin", "v_lin", "out_lin"],
                "bias": "none"
            }
        
        self.lora_config = lora_config
        self.model = None
        self.tokenizer = None
        
        self._initialize_model()
    
    def _initialize_model(self):
        """
        Initialize model - SIMPLIFIED for Kaggle compatibility
        """
        try:
            # Load configuration
            config = AutoConfig.from_pretrained(
                self.model_name,
                num_labels=self.num_domains,
                hidden_dropout_prob=0.1,
                attention_probs_dropout_prob=0.1
            )
            
            # SIMPLE: Load model directly without fancy options
            # This avoids meta tensor issues
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                config=config,
            )
            
            # Move to device BEFORE applying LoRA
            self.model.to(self.device)
            
            # Apply LoRA
            peft_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                **self.lora_config
            )
            
            self.model = get_peft_model(self.model, peft_config)
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token or '<pad>'
            
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.model.parameters())
            
            logger.info(f"✅ Initialized {self.model_name} on {self.device}")
            logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
            
            if self.is_kaggle:
                print("🚀 Running on Kaggle GPU")
            elif self.is_colab:
                print("🚀 Running on Colab GPU")
            
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
        """Train the model"""
        # Disable WandB integration to avoid login errors
        os.environ["WANDB_DISABLED"] = "true"
        os.environ["WANDB_MODE"] = "disabled"

        # Use Kaggle/Colab output directory
        if self.is_kaggle:
            output_dir = output_dir.replace("./", "/kaggle/working/")
        elif self.is_colab:
            output_dir = output_dir.replace("./", "/content/")

        if training_config is None:
            training_config = {
                "num_epochs": 5,
                "learning_rate": 2e-4,
                "per_device_train_batch_size": 16,
            }

        # Enhanced logging
        from transformers import logging as hf_logging
        hf_logging.set_verbosity_info()
        
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
            report_to=[],  # Explicitly disable all integrations (WandB, TensorBoard, etc.)
            dataloader_pin_memory=True,
            fp16=torch.cuda.is_available(),
            dataloader_num_workers=0,
            disable_tqdm=False,
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_loader.dataset,
            eval_dataset=val_loader.dataset,
            compute_metrics=self._compute_metrics,
        )
        
        logger.info("Starting training...")
        logger.info(f"Output directory: {output_dir}")
        trainer.train()
        
        # Save the model
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        logger.info(f"✅ Training completed. Model saved to {output_dir}")
        
        # GPU memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("🧹 GPU memory cleared")
        
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

    def evaluate_with_detailed_metrics(self, test_loader):
        """
        Comprehensive evaluation with detailed metrics including confusion matrix.

        Args:
            test_loader: DataLoader for test dataset

        Returns:
            Dictionary containing:
                - accuracy: Overall accuracy
                - precision: Weighted precision
                - recall: Weighted recall
                - f1: Weighted F1 score
                - per_class_metrics: Metrics for each class
                - confusion_matrix: Confusion matrix
                - classification_report: Detailed classification report
        """
        import numpy as np
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            confusion_matrix, classification_report
        )

        self.model.eval()
        all_predictions = []
        all_labels = []

        logger.info("Running detailed evaluation...")

        with torch.no_grad():
            for batch in test_loader:
                # Move batch to device
                inputs = {
                    'input_ids': batch['input_ids'].to(self.device),
                    'attention_mask': batch['attention_mask'].to(self.device)
                }
                labels = batch['labels'].to(self.device)

                # Get predictions
                outputs = self.model(**inputs)
                predictions = torch.argmax(outputs.logits, dim=-1)

                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)

        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)

        # Get unique labels present in test data
        unique_labels = sorted(np.unique(np.concatenate([all_labels, all_predictions])))

        # Get domain names for the unique labels
        from adaptive_rag_router.data.data_loader import CLINC150DataLoader
        all_domain_names = CLINC150DataLoader.DOMAIN_NAMES

        # Create target_names only for the labels present in the data
        target_names = [all_domain_names[label] for label in unique_labels]

        logger.info(f"Test set contains {len(unique_labels)} unique domains: {target_names}")

        # Per-class metrics (only for present labels)
        precision_per_class = precision_score(all_labels, all_predictions, labels=unique_labels, average=None, zero_division=0)
        recall_per_class = recall_score(all_labels, all_predictions, labels=unique_labels, average=None, zero_division=0)
        f1_per_class = f1_score(all_labels, all_predictions, labels=unique_labels, average=None, zero_division=0)

        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions, labels=unique_labels)

        # Classification report (only for present labels)
        report = classification_report(
            all_labels,
            all_predictions,
            labels=unique_labels,
            target_names=target_names,
            zero_division=0
        )

        # Per-class metrics dictionary
        per_class_metrics = {}
        for i, label_idx in enumerate(unique_labels):
            domain_name = all_domain_names[label_idx]
            per_class_metrics[domain_name] = {
                'precision': float(precision_per_class[i]),
                'recall': float(recall_per_class[i]),
                'f1': float(f1_per_class[i])
            }

        results = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'per_class_metrics': per_class_metrics,
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'num_samples': len(all_labels)
        }

        logger.info(f"Evaluation completed on {len(all_labels)} samples")
        logger.info(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

        return results
    
    def save(self, path: str):
        """Save model and tokenizer"""
        if self.is_kaggle:
            path = path.replace("./", "/kaggle/working/")
        elif self.is_colab:
            path = path.replace("./", "/content/")
        
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model and tokenizer"""
        from peft import PeftModel
        
        if self.is_kaggle:
            path = path.replace("./", "/kaggle/working/")
        elif self.is_colab:
            path = path.replace("./", "/content/")
        
        self.model = PeftModel.from_pretrained(self.model, path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model.to(self.device)
        logger.info(f"Model loaded from {path}")


def create_router_model(
    model_type: str = "distilbert",
    lora_rank: int = 16,
    device: Optional[str] = None
) -> AdaptiveRAGRouter:
    """
    Factory function to create router models
    FIXED: Model-specific LoRA target modules
    """
    
    model_mapping = {
        "distilbert": "distilbert-base-uncased",
        "roberta": "roberta-base",
        "deberta": "microsoft/deberta-v3-base"
    }
    
    if model_type not in model_mapping:
        raise ValueError(f"Model type {model_type} not supported. Choose from {list(model_mapping.keys())}")
    
    # CRITICAL: Different models have different attention layer names
    if model_type == "distilbert":
        target_modules = ["q_lin", "k_lin", "v_lin", "out_lin"]
    elif model_type == "roberta":
        target_modules = ["query", "key", "value", "dense"]
    elif model_type == "deberta":
        target_modules = ["query_proj", "key_proj", "value_proj", "dense"]
    else:
        target_modules = ["query", "key", "value", "dense"]
    
    lora_config = {
        "r": lora_rank,
        "lora_alpha": lora_rank * 2,
        "lora_dropout": 0.05,
        "target_modules": target_modules,
        "bias": "none"
    }
    
    return AdaptiveRAGRouter(
        model_name=model_mapping[model_type],
        lora_config=lora_config,
        device=device
    )