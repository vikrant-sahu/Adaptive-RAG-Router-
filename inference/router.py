from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import torch
import time

class AdaptiveRAGRouter:
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        base_model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=4
        )
        self.model = PeftModel.from_pretrained(base_model, model_path)
        self.model.eval()

        self.labels = {0: 'factual', 1: 'analytical', 2: 'conversational', 3: 'temporal'}

    def classify(self, query, confidence_threshold=0.85):
        inputs = self.tokenizer(query, return_tensors="pt", truncation=True, max_length=128)
        start_time = time.time()
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_idx].item()
        latency_ms = (time.time() - start_time) * 1000
        return {
            'intent': self.labels[pred_idx],
            'confidence': confidence,
            'latency_ms': latency_ms,
            'probs': {self.labels[i]: probs[0][i].item() for i in range(4)}
        }