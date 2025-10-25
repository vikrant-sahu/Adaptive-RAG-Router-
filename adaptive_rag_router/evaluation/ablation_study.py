# File: adaptive_rag_router/evaluation/ablation_study.py
"""
LoRA Ablation Study - Compare different LoRA configurations
"""

import os
import json
import time
import logging
import pandas as pd
from typing import List, Dict
import torch
from adaptive_rag_router.data.data_loader import CLINC150DataLoader
from adaptive_rag_router.models.adaptive_router import create_router_model

logger = logging.getLogger(__name__)

class LoRAAblationStudy:
    """Conduct ablation studies on LoRA configurations"""
    
    def __init__(self, output_dir: str = "./ablation_results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def run_ablation(
        self,
        model_types: List[str] = None,
        lora_ranks: List[int] = None,
        num_samples: int = 2000
    ) -> pd.DataFrame:
        """Run complete ablation study"""
        
        if model_types is None:
            model_types = ["distilbert", "roberta"]
        
        if lora_ranks is None:
            lora_ranks = [4, 8, 16, 32]
        
        results = []
        
        logger.info(f"Starting ablation study with {len(model_types)} models and {len(lora_ranks)} ranks")
        
        for model_type in model_types:
            for rank in lora_ranks:
                logger.info(f"Testing {model_type} with LoRA rank {rank}")
                
                try:
                    result = self._test_configuration(
                        model_type=model_type,
                        lora_rank=rank,
                        num_samples=num_samples
                    )
                    results.append(result)
                    
                except Exception as e:
                    logger.error(f"Failed for {model_type} rank {rank}: {e}")
                    continue
        
        # Create DataFrame
        results_df = pd.DataFrame(results)
        
        # Save results
        output_path = os.path.join(self.output_dir, "ablation_results.csv")
        results_df.to_csv(output_path, index=False)
        logger.info(f"Ablation results saved to {output_path}")
        
        return results_df
    
    def _test_configuration(
        self,
        model_type: str,
        lora_rank: int,
        num_samples: int
    ) -> Dict:
        """Test a specific configuration"""
        
        # Create model
        model = create_router_model(model_type=model_type, lora_rank=lora_rank)
        
        # Load data
        data_loader = CLINC150DataLoader(model_name=model.model_name)
        _, val_loader, test_loader = data_loader.get_data_loaders(
            batch_size=32,
            sample_size=num_samples
        )
        
        # Quick training (for ablation we do minimal training)
        training_config = {
            "num_epochs": 2,
            "learning_rate": 2e-4,
            "per_device_train_batch_size": 16,
            "per_device_eval_batch_size": 32
        }
        
        # Get a small train set
        train_loader, _, _ = data_loader.get_data_loaders(
            batch_size=16,
            sample_size=min(500, num_samples)
        )
        
        # Train
        model.train(
            train_loader=train_loader,
            val_loader=val_loader,
            output_dir=os.path.join(self.output_dir, f"temp_{model_type}_r{lora_rank}"),
            training_config=training_config
        )
        
        # Evaluate
        accuracy, inference_time = self._evaluate_model(model, test_loader)
        
        # Calculate parameter stats
        trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.model.parameters())
        
        return {
            "model_type": model_type,
            "lora_rank": lora_rank,
            "accuracy": accuracy,
            "inference_time_ms": inference_time,
            "trainable_params": trainable_params,
            "total_params": total_params,
            "parameter_efficiency": trainable_params / total_params
        }
    
    def _evaluate_model(self, model, test_loader) -> tuple:
        """Evaluate model accuracy and inference time"""
        
        model.model.eval()
        correct = 0
        total = 0
        inference_times = []
        
        with torch.no_grad():
            for batch in test_loader:
                texts = ["sample text"] * len(batch["input_ids"])
                
                start_time = time.time()
                outputs = model.model(
                    input_ids=batch["input_ids"].to(model.device),
                    attention_mask=batch["attention_mask"].to(model.device)
                )
                inference_time = (time.time() - start_time) * 1000
                inference_times.append(inference_time)
                
                predictions = torch.argmax(outputs.logits, dim=-1)
                correct += (predictions == batch["labels"].to(model.device)).sum().item()
                total += len(batch["labels"])
        
        accuracy = correct / total if total > 0 else 0
        avg_inference_time = sum(inference_times) / len(inference_times) if inference_times else 0
        
        return accuracy, avg_inference_time