import os
import sys

# Install dependencies
print("📦 Installing dependencies...")
# Install dependencies with safe versions
!pip install -q --no-cache-dir \
    "numpy<2" \
    "pyarrow==14.0.2" \
    "datasets==2.18.0" \
    "transformers==4.44.2" \
    "peft==0.10.0" \
    "accelerate==0.33.0" \
    "torch==2.3.0" 

!pip install scikit-learn matplotlib seaborn tqdm
    
!rm -rf ~/.cache/pip
print("✅ Dependencies installed")

# Clone your GitHub repository
print("\n📥 Cloning repository from GitHub...")
repo_url = "https://github.com/vikrant-sahu/Adaptive-RAG-Router-.git"  # CHANGE THIS!

# Remove if exists (for reruns)
!rm -rf /kaggle/working/adaptive-rag-router

# Clone repository
!git clone {repo_url} /kaggle/working/adaptive-rag-router

# Add to Python path
sys.path.insert(0, '/kaggle/working/adaptive-rag-router')

print("✅ Repository cloned and added to path")


# Show environment info
import torch
print(f"\n🖥️  Environment Info:")
print(f"  • Python path: {sys.path[0]}")
print(f"  • CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  • GPU: {torch.cuda.get_device_name(0)}")
    print(f"  • GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

print("\n🎉 Setup complete! Ready to run notebooks.")


try:
    from adaptive_rag_router import (
        CLINC150DataLoader,
        AdaptiveRAGRouter,
        create_router_model,
        ModelTrainer,
        LoRAAblationStudy,
        LLMBenchmark
    )
    print("✅ All imports successful!")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\nDebugging info:")
    !ls -la /kaggle/working/adaptive-rag-router/
    !ls -la /kaggle/working/adaptive-rag-router/adaptive_rag_router/