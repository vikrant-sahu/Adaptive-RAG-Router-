# File: benchmarks/llm_benchmark.py
"""
Benchmark against GPT-4 and Claude-3.5
"""

import os
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class LLMBenchmark:
    """Benchmark Adaptive RAG Router against proprietary LLMs"""
    
    def __init__(self, output_dir: str = "./benchmark_results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Cost rates (per 1M tokens)
        self.cost_rates = {
            "gpt-4": {"input": 30.0, "output": 60.0},  # $30/M input, $60/M output
            "gpt-3.5-turbo": {"input": 1.5, "output": 2.0},
            "claude-3-5-sonnet": {"input": 3.0, "output": 15.0},
            "our_model": {"inference": 0.8}  # $0.8 per 1M inferences
        }
        
    def benchmark_proprietary_llms(self, test_samples: List[Dict], llm_type: str = "gpt-4") -> Dict:
        """Benchmark against proprietary LLMs"""
        results = {
            "llm_type": llm_type,
            "predictions": [],
            "latencies": [],
            "costs": [],
            "accuracy": 0.0,
            "total_cost": 0.0
        }
        
        # Domain classification prompt
        system_prompt = """You are a domain classification system. Classify the user query into one of these 10 domains:
        banking, credit_cards, work, travel, utility, auto_&_commute, home, kitchen_&_dining, small_talk, meta
        
        Respond with ONLY the domain name, nothing else."""
        
        for sample in tqdm(test_samples, desc=f"Testing {llm_type}"):
            try:
                start_time = time.time()
                
                if llm_type.startswith("gpt"):
                    response = self._call_openai(system_prompt, sample["text"], llm_type)
                elif llm_type.startswith("claude"):
                    response = self._call_anthropic(system_prompt, sample["text"], llm_type)
                else:
                    raise ValueError(f"Unsupported LLM type: {llm_type}")
                
                latency = time.time() - start_time
                
                # Calculate cost (simplified)
                input_tokens = len(sample["text"].split()) // 0.75  # Rough estimate
                output_tokens = len(response.split()) // 0.75
                cost = self._calculate_cost(llm_type, input_tokens, output_tokens)
                
                results["predictions"].append(response.strip().lower())
                results["latencies"].append(latency)
                results["costs"].append(cost)
                
            except Exception as e:
                logger.error(f"Error with {llm_type} on sample: {e}")
                results["predictions"].append("error")
                results["latencies"].append(0.0)
                results["costs"].append(0.0)
        
        # Calculate accuracy
        correct = 0
        for i, pred in enumerate(results["predictions"]):
            if pred == test_samples[i]["domain"]:
                correct += 1
        
        results["accuracy"] = correct / len(test_samples)
        results["total_cost"] = sum(results["costs"])
        results["avg_latency"] = np.mean(results["latencies"])
        
        return results
    
    def benchmark_our_models(self, test_samples: List[Dict], model_paths: Dict) -> Dict:
        """Benchmark our trained models"""
        results = {}
        
        for model_name, model_path in model_paths.items():
            print(f"🧪 Benchmarking {model_name}...")
            
            try:
                # Load model
                from models.adaptive_router import AdaptiveRAGRouter
                model = AdaptiveRAGRouter()
                model.load(model_path)
                
                model_results = {
                    "predictions": [],
                    "latencies": [],
                    "accuracy": 0.0,
                    "total_cost": 0.0
                }
                
                texts = [sample["text"] for sample in test_samples]
                true_domains = [sample["domain"] for sample in test_samples]
                
                # Batch prediction
                start_time = time.time()
                predictions = model.predict(texts, batch_size=32)
                total_time = time.time() - start_time
                
                model_results["predictions"] = predictions["domains"]
                model_results["latencies"] = [total_time / len(texts)] * len(texts)
                
                # Calculate accuracy
                correct = sum(1 for i, pred in enumerate(predictions["domains"]) 
                            if pred == true_domains[i])
                model_results["accuracy"] = correct / len(texts)
                
                # Calculate cost
                cost_per_inference = self.cost_rates["our_model"]["inference"] / 1e6
                model_results["total_cost"] = cost_per_inference * len(texts)
                model_results["avg_latency"] = total_time / len(texts)
                
                results[model_name] = model_results
                
            except Exception as e:
                logger.error(f"Error benchmarking {model_name}: {e}")
                results[model_name] = {"error": str(e)}
        
        return results
    
    def run_complete_benchmark(self, num_samples: int = 500) -> Dict:
        """Run complete benchmark against all systems"""
        # Load test data
        from data.data_loader import CLINC150DataLoader
        data_loader = CLINC150DataLoader()
        test_dataset = data_loader.load_dataset("test")
        
        # Create test samples
        import random
        random.seed(42)
        indices = random.sample(range(len(test_dataset)), min(num_samples, len(test_dataset)))
        test_samples = []
        
        for idx in indices:
            sample = test_dataset[idx]
            domain = data_loader.extract_domain_from_intent(sample["intent"])
            test_samples.append({
                "text": sample["text"],
                "domain": domain,
                "true_intent": sample["intent"]
            })
        
        # Benchmark proprietary LLMs
        print("🚀 Benchmarking Proprietary LLMs...")
        llm_results = {}
        
        # Note: In actual implementation, you'd need API keys
        # llm_results["gpt-4"] = self.benchmark_proprietary_llms(test_samples, "gpt-4")
        # llm_results["claude-3-5-sonnet"] = self.benchmark_proprietary_llms(test_samples, "claude-3-5-sonnet")
        
        # Benchmark our models (you'll need to update these paths)
        print("🧪 Benchmarking Our Models...")
        model_paths = {
            "distilbert_lora": "./model_checkpoints/distilbert_best",
            "roberta_lora": "./model_checkpoints/roberta_best", 
            "deberta_lora": "./model_checkpoints/deberta_best"
        }
        
        our_results = self.benchmark_our_models(test_samples, model_paths)
        
        # Combine results
        all_results = {
            "proprietary_llms": llm_results,
            "our_models": our_results,
            "test_samples": test_samples,
            "benchmark_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "num_samples": num_samples
        }
        
        # Save results
        with open(os.path.join(self.output_dir, "complete_benchmark.json"), "w") as f:
            json.dump(all_results, f, indent=2)
        
        # Generate comparison report
        comparison_df = self._generate_comparison_report(all_results)
        comparison_df.to_csv(os.path.join(self.output_dir, "benchmark_comparison.csv"), index=False)
        
        self._generate_benchmark_plots(comparison_df)
        
        return all_results
    
    def _call_openai(self, system_prompt: str, user_message: str, model: str) -> str:
        """Call OpenAI API (placeholder - needs API key)"""
        # This is a placeholder - you need to implement actual API calls
        # import openai
        # client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # response = client.chat.completions.create(
        #     model=model,
        #     messages=[
        #         {"role": "system", "content": system_prompt},
        #         {"role": "user", "content": user_message}
        #     ],
        #     temperature=0.0
        # )
        # return response.choices[0].message.content
        
        # Simulated response for demonstration
        time.sleep(0.1)  # Simulate API latency
        domains = ["banking", "credit_cards", "work", "travel", "utility", 
                  "auto_&_commute", "home", "kitchen_&_dining", "small_talk", "meta"]
        return np.random.choice(domains)
    
    def _call_anthropic(self, system_prompt: str, user_message: str, model: str) -> str:
        """Call Anthropic API (placeholder - needs API key)"""
        # Similar to OpenAI - implement actual API calls
        time.sleep(0.08)  # Simulate API latency
        domains = ["banking", "credit_cards", "work", "travel", "utility", 
                  "auto_&_commute", "home", "kitchen_&_dining", "small_talk", "meta"]
        return np.random.choice(domains)
    
    def _calculate_cost(self, llm_type: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for API call"""
        if llm_type not in self.cost_rates:
            return 0.0
        
        rates = self.cost_rates[llm_type]
        cost = (input_tokens * rates["input"] / 1e6) + (output_tokens * rates["output"] / 1e6)
        return cost
    
    def _generate_comparison_report(self, results: Dict) -> pd.DataFrame:
        """Generate comparison report DataFrame"""
        comparison_data = []
        
        # Add our models
        for model_name, model_results in results["our_models"].items():
            if "error" not in model_results:
                comparison_data.append({
                    "model": model_name,
                    "type": "our_model",
                    "accuracy": model_results["accuracy"],
                    "avg_latency_ms": model_results["avg_latency"] * 1000,
                    "cost_per_1k": (model_results["total_cost"] / len(results["test_samples"])) * 1000,
                    "total_params": "N/A",  # Would need to track this
                    "trainable_params": "N/A"
                })
        
        # Add proprietary LLMs (when implemented)
        for llm_name, llm_results in results["proprietary_llms"].items():
            comparison_data.append({
                "model": llm_name,
                "type": "proprietary_llm",
                "accuracy": llm_results["accuracy"],
                "avg_latency_ms": llm_results["avg_latency"] * 1000,
                "cost_per_1k": (llm_results["total_cost"] / len(results["test_samples"])) * 1000,
                "total_params": "N/A",
                "trainable_params": "N/A"
            })
        
        return pd.DataFrame(comparison_data)
    
    def _generate_benchmark_plots(self, df: pd.DataFrame):
        """Generate benchmark comparison plots"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('LLM Benchmark Comparison', fontsize=16, fontweight='bold')
        
        # Plot 1: Accuracy Comparison
        sns.barplot(data=df, x='model', y='accuracy', ax=axes[0, 0], palette='viridis')
        axes[0, 0].set_title('Accuracy Comparison')
        axes[0, 0].set_xticklabels(axes[0, 0].get_xticklabels(), rotation=45)
        axes[0, 0].set_ylabel('Accuracy')
        
        # Plot 2: Latency Comparison
        sns.barplot(data=df, x='model', y='avg_latency_ms', ax=axes[0, 1], palette='plasma')
        axes[0, 1].set_title('Latency Comparison (ms)')
        axes[0, 1].set_xticklabels(axes[0, 1].get_xticklabels(), rotation=45)
        axes[0, 1].set_ylabel('Latency (ms)')
        
        # Plot 3: Cost Comparison
        sns.barplot(data=df, x='model', y='cost_per_1k', ax=axes[1, 0], palette='coolwarm')
        axes[1, 0].set_title('Cost per 1K Queries ($)')
        axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), rotation=45)
        axes[1, 0].set_ylabel('Cost ($)')
        
        # Plot 4: Cost vs Accuracy Scatter
        colors = ['red' if typ == 'proprietary_llm' else 'blue' for typ in df['type']]
        axes[1, 1].scatter(df['cost_per_1k'], df['accuracy'], c=colors, s=100, alpha=0.7)
        
        for i, row in df.iterrows():
            axes[1, 1].annotate(row['model'], (row['cost_per_1k'], row['accuracy']),
                               xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        axes[1, 1].set_xlabel('Cost per 1K Queries ($)')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].set_title('Cost vs Accuracy Trade-off')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', label='Proprietary LLM'),
            Patch(facecolor='blue', label='Our Model')
        ]
        axes[1, 1].legend(handles=legend_elements)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'llm_benchmark_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Run complete benchmark"""
    print("🚀 Starting LLM Benchmark Comparison")
    print("=" * 50)
    
    benchmark = LLMBenchmark()
    results = benchmark.run_complete_benchmark(num_samples=200)
    
    print("\n📊 Benchmark Completed!")
    print("Results saved to ./benchmark_results/")

if __name__ == "__main__":
    main()