# File: adaptive_rag_router/training/trainer.py
"""
Production training module
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional
from adaptive_rag_router.data.data_loader import CLINC150DataLoader
from adaptive_rag_router.models.adaptive_router import create_router_model

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Production model trainer with environment awareness"""
    
    def __init__(self, output_dir: str = "./model_checkpoints"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Auto-detect environment
        self.is_cloud = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ or 'COLAB_GPU' in os.environ
        
    def train_model(
        self, 
        model_type: str = "distilbert",
        training_config: Optional[Dict] = None,
        sample_size: Optional[int] = None
    ) -> Dict:
        """Train a model with given configuration"""
        
        if training_config is None:
            training_config = {}
        
        logger.info(f"Training {model_type} model...")
        
        # Get model configuration
        from adaptive_rag_router.config.training_config import MODEL_CONFIGS
        if model_type not in MODEL_CONFIGS:
            raise ValueError(f"Model {model_type} not supported. Choose from {list(MODEL_CONFIGS.keys())}")
        
        model_config = MODEL_CONFIGS[model_type]
        
        # Adjust for cloud environments
        if self.is_cloud:
            training_config.setdefault("num_epochs", 3)
            training_config.setdefault("per_device_train_batch_size", 16)
            sample_size = sample_size or 1000  # Smaller samples for quick results
        
        # Initialize data loader
        data_loader = CLINC150DataLoader(model_name=model_config["model_name"])
        train_loader, val_loader, test_loader = data_loader.get_data_loaders(
            batch_size=training_config.get("per_device_train_batch_size", 16),
            sample_size=sample_size
        )
        
        # Initialize model
        model = create_router_model(
            model_type=model_type,
            lora_rank=model_config["lora_rank"]
        )
        
        # Create output directory
        model_output_dir = os.path.join(
            self.output_dir, 
            f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        # Train model
        training_config["learning_rate"] = model_config["learning_rate"]
        trainer = model.train(
            train_loader=train_loader,
            val_loader=val_loader,
            output_dir=model_output_dir,
            training_config=training_config
        )
        
        # Evaluate
        test_results = trainer.evaluate(test_loader.dataset)
        
        # Save results
        results = {
            "model": model_type,
            "test_accuracy": test_results["eval_accuracy"],
            "test_f1": test_results["eval_f1"],
            "output_dir": model_output_dir,
            "training_config": training_config,
            "environment": "cloud" if self.is_cloud else "local"
        }
        
        with open(os.path.join(model_output_dir, "training_results.json"), "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Training completed for {model_type}")
        logger.info(f"Test Accuracy: {test_results['eval_accuracy']:.4f}")
        
        return results
    
    def train_quick_demo(self) -> Dict:
        """Quick demo training for cloud environments"""
        logger.info("Running quick demo training...")
        return self.train_model(
            model_type="distilbert",
            sample_size=500,
            training_config={
                "num_epochs": 2,
                "per_device_train_batch_size": 8
            }
        )