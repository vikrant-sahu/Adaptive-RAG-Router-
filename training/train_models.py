# File: training/train_models.py
"""
Training script for all Adaptive RAG Router models
"""

import os
import torch
import logging
from datetime import datetime
from data.data_loader import CLINC150DataLoader
from models.adaptive_router import create_router_model
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """Trainer for Adaptive RAG Router models"""
    
    def __init__(self, output_dir: str = "./model_checkpoints"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Training configurations for different models
        self.model_configs = {
            "distilbert": {
                "model_type": "distilbert",
                "lora_rank": 8,
                "learning_rate": 3e-4,
                "batch_size": 64
            },
            "roberta": {
                "model_type": "roberta", 
                "lora_rank": 16,
                "learning_rate": 2e-4,
                "batch_size": 32
            },
            "deberta": {
                "model_type": "deberta",
                "lora_rank": 16,
                "learning_rate": 2e-4,
                "batch_size": 32
            }
        }
    
    def train_model(self, model_name: str, save_best: bool = True) -> dict:
        """Train a specific model"""
        if model_name not in self.model_configs:
            raise ValueError(f"Model {model_name} not supported")
        
        config = self.model_configs[model_name]
        logger.info(f"Training {model_name} with config: {config}")
        
        # Initialize data loader
        data_loader = CLINC150DataLoader(
            model_name=self._get_model_path(config["model_type"]),
            max_length=128
        )
        
        # Create data loaders
        train_loader, val_loader, test_loader = data_loader.create_data_loaders(
            batch_size=config["batch_size"]
        )
        
        # Initialize model
        model = create_router_model(
            model_type=config["model_type"],
            lora_rank=config["lora_rank"]
        )
        
        # Create output directory
        model_output_dir = os.path.join(
            self.output_dir, 
            f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        # Train model
        trainer = model.train(
            train_loader=train_loader,
            val_loader=val_loader,
            output_dir=model_output_dir,
            num_epochs=10,
            learning_rate=config["learning_rate"]
        )
        
        # Evaluate on test set
        test_results = trainer.evaluate(test_loader.dataset)
        
        # Save results
        results = {
            "model": model_name,
            "config": config,
            "test_accuracy": test_results["eval_accuracy"],
            "test_f1": test_results["eval_f1"],
            "output_dir": model_output_dir,
            "training_logs": trainer.state.log_history
        }
        
        # Save results to file
        with open(os.path.join(model_output_dir, "training_results.json"), "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Training completed for {model_name}")
        logger.info(f"Test Accuracy: {test_results['eval_accuracy']:.4f}")
        logger.info(f"Test F1: {test_results['eval_f1']:.4f}")
        
        return results
    
    def train_all_models(self) -> dict:
        """Train all models and return results"""
        results = {}
        
        for model_name in self.model_configs.keys():
            try:
                results[model_name] = self.train_model(model_name)
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                results[model_name] = {"error": str(e)}
        
        # Save comparative results
        comparative_results = self._create_comparative_report(results)
        with open(os.path.join(self.output_dir, "comparative_results.json"), "w") as f:
            json.dump(comparative_results, f, indent=2)
        
        return comparative_results
    
    def _get_model_path(self, model_type: str) -> str:
        """Get model path from type"""
        mapping = {
            "roberta": "roberta-base",
            "distilbert": "distilbert-base-uncased",
            "deberta": "microsoft/deberta-v3-base"
        }
        return mapping[model_type]
    
    def _create_comparative_report(self, results: dict) -> dict:
        """Create comparative report of all models"""
        report = {
            "training_date": datetime.now().isoformat(),
            "models": {},
            "summary": {}
        }
        
        best_accuracy = 0
        best_model = None
        
        for model_name, result in results.items():
            if "error" not in result:
                accuracy = result["test_accuracy"]
                report["models"][model_name] = {
                    "accuracy": accuracy,
                    "f1_score": result["test_f1"],
                    "config": result["config"],
                    "output_dir": result["output_dir"]
                }
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = model_name
        
        report["summary"] = {
            "best_model": best_model,
            "best_accuracy": best_accuracy,
            "total_models_trained": len([r for r in results.values() if "error" not in r])
        }
        
        return report

def main():
    """Main training function"""
    trainer = ModelTrainer()
    
    print("🚀 Starting Adaptive RAG Router Training")
    print("=" * 50)
    
    # Train all models
    results = trainer.train_all_models()
    
    print("\n📊 Training Results Summary:")
    print("=" * 50)
    for model_name, result in results["models"].items():
        print(f"{model_name:12} | Accuracy: {result['accuracy']:.4f} | F1: {result['f1_score']:.4f}")
    
    print(f"\n🏆 Best Model: {results['summary']['best_model']} "
          f"(Accuracy: {results['summary']['best_accuracy']:.4f})")

if __name__ == "__main__":
    main()