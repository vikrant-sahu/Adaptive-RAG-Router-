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
        sample_size: Optional[int] = None,
        use_custom_split: bool = False,
        train_val_ratio: float = 0.7,
        save_path: Optional[str] = None
    ) -> Dict:
        """
        Train a model with given configuration.

        Args:
            model_type: Type of model to train ('distilbert', 'roberta', 'deberta')
            training_config: Training configuration dictionary
            sample_size: Number of samples to use (for quick demos)
            use_custom_split: Whether to use custom 70/30 split (default: False)
            train_val_ratio: Ratio for train+val split if using custom split (default: 0.7)
            save_path: Custom path to save the model (default: auto-generated)

        Returns:
            Dictionary with training results and metrics
        """

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

        # Load data with appropriate split method
        if use_custom_split:
            logger.info(f"Using custom split: {train_val_ratio*100:.0f}% train+val, {(1-train_val_ratio)*100:.0f}% test")
            train_loader, val_loader, test_loader = data_loader.get_custom_split_loaders(
                batch_size=training_config.get("per_device_train_batch_size", 16),
                train_val_ratio=train_val_ratio,
                val_split=0.15,  # 15% of train+val for validation
                seed=42
            )
        else:
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
        if save_path:
            model_output_dir = save_path
        else:
            model_output_dir = os.path.join(
                self.output_dir,
                f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )

        os.makedirs(model_output_dir, exist_ok=True)

        # Train model
        training_config["learning_rate"] = model_config["learning_rate"]
        trainer = model.train(
            train_loader=train_loader,
            val_loader=val_loader,
            output_dir=model_output_dir,
            training_config=training_config
        )

        # Evaluate with detailed metrics
        logger.info("Running comprehensive evaluation on test set...")
        detailed_metrics = model.evaluate_with_detailed_metrics(test_loader)

        # Save results
        results = {
            "model": model_type,
            "test_accuracy": detailed_metrics["accuracy"],
            "test_precision": detailed_metrics["precision"],
            "test_recall": detailed_metrics["recall"],
            "test_f1": detailed_metrics["f1"],
            "per_class_metrics": detailed_metrics["per_class_metrics"],
            "confusion_matrix": detailed_metrics["confusion_matrix"],
            "num_test_samples": detailed_metrics["num_samples"],
            "output_dir": model_output_dir,
            "training_config": training_config,
            "environment": "cloud" if self.is_cloud else "local",
            "used_custom_split": use_custom_split,
            "train_val_ratio": train_val_ratio if use_custom_split else None
        }

        # Save detailed results
        with open(os.path.join(model_output_dir, "training_results.json"), "w") as f:
            json.dump(results, f, indent=2)

        # Save classification report
        with open(os.path.join(model_output_dir, "classification_report.txt"), "w") as f:
            f.write(detailed_metrics["classification_report"])

        logger.info(f"Training completed for {model_type}")
        logger.info(f"Test Accuracy: {detailed_metrics['accuracy']:.4f}")
        logger.info(f"Test F1: {detailed_metrics['f1']:.4f}")
        logger.info(f"Results saved to: {model_output_dir}")

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