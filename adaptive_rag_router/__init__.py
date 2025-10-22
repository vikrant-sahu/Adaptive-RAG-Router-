# File: adaptive_rag_router/__init__.py
"""
Adaptive RAG Router - Production-ready PEFT-based intent classification
"""

__version__ = "1.0.0"
__author__ = "Adaptive RAG Team"

from adaptive_rag_router.config.training_config import *
from adaptive_rag_router.data.data_loader import CLINC150DataLoader
from adaptive_rag_router.models.adaptive_router import AdaptiveRAGRouter, create_router_model
from adaptive_rag_router.training.trainer import ModelTrainer
from adaptive_rag_router.evaluation.ablation_study import LoRAAblationStudy
from adaptive_rag_router.benchmarks.llm_benchmark import LLMBenchmark

__all__ = [
    "CLINC150DataLoader",
    "AdaptiveRAGRouter", 
    "create_router_model",
    "ModelTrainer",
    "LoRAAblationStudy",
    "LLMBenchmark",
]