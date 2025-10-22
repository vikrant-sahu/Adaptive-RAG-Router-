# File: tests/test_components.py
"""
Comprehensive test suite for Adaptive RAG Router
"""

import unittest
import torch
import numpy as np
from data.data_loader import CLINC150DataLoader
from models.adaptive_router import AdaptiveRAGRouter, create_router_model

class TestDataLoader(unittest.TestCase):
    """Test data loading and preprocessing"""
    
    def setUp(self):
        self.data_loader = CLINC150DataLoader(max_length=128)
    
    def test_domain_extraction(self):
        """Test domain extraction from intent names"""
        test_cases = [
            ("banking_transfer", "banking"),
            ("credit_cards_balance", "credit_cards"),
            ("work_schedule", "work"),
            ("oos", "meta")
        ]
        
        for intent, expected_domain in test_cases:
            extracted = self.data_loader.extract_domain_from_intent(intent)
            self.assertEqual(extracted, expected_domain)
    
    def test_dataset_loading(self):
        """Test that datasets load correctly"""
        for split in ["train", "validation", "test"]:
            dataset = self.data_loader.load_dataset(split)
            self.assertGreater(len(dataset), 0)
            
            # Check structure
            sample = dataset[0]
            self.assertIn("text", sample)
            self.assertIn("intent", sample)
    
    def test_preprocessing(self):
        """Test data preprocessing"""
        processed = self.data_loader.get_processed_dataset("train")
        sample = processed[0]
        
        self.assertIn("input_ids", sample)
        self.assertIn("attention_mask", sample)
        self.assertIn("labels", sample)
        self.assertIn("domain", sample)

class TestModels(unittest.TestCase):
    """Test model initialization and functionality"""
    
    def test_model_creation(self):
        """Test model creation with different configurations"""
        model_types = ["distilbert", "roberta"]
        
        for model_type in model_types:
            with self.subTest(model_type=model_type):
                model = create_router_model(model_type=model_type, lora_rank=8)
                self.assertIsNotNone(model.model)
                self.assertIsNotNone(model.tokenizer)
    
    def test_prediction(self):
        """Test model prediction functionality"""
        model = create_router_model(model_type="distilbert", lora_rank=8)
        
        test_texts = [
            "What's my account balance?",
            "I need to transfer money",
            "What's the weather today?"
        ]
        
        results = model.predict(test_texts)
        
        self.assertIn("domains", results)
        self.assertIn("predictions", results)
        self.assertIn("probabilities", results)
        self.assertEqual(len(results["domains"]), len(test_texts))
    
    def test_lora_parameters(self):
        """Test that LoRA parameters are trainable"""
        model = create_router_model(model_type="roberta", lora_rank=16)
        
        trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.model.parameters())
        
        # LoRA should have significantly fewer trainable parameters
        self.assertLess(trainable_params, total_params * 0.1)  # <10% trainable

class TestIntegration(unittest.TestCase):
    """Test integration between components"""
    
    def test_end_to_end(self):
        """Test minimal end-to-end workflow"""
        # Load data
        data_loader = CLINC150DataLoader(max_length=128)
        train_loader, val_loader, _ = data_loader.create_data_loaders(batch_size=2)
        
        # Initialize model
        model = create_router_model(model_type="distilbert", lora_rank=4)
        
        # Test prediction on validation data
        sample_batch = next(iter(val_loader))
        texts = ["Test query 1", "Test query 2"]
        predictions = model.predict(texts)
        
        self.assertEqual(len(predictions["domains"]), len(texts))

def run_tests():
    """Run all tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add tests
    suite.addTests(loader.loadTestsFromTestCase(TestDataLoader))
    suite.addTests(loader.loadTestsFromTestCase(TestModels))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_tests()
    print(f"\nTests {'PASSED' if success else 'FAILED'}")