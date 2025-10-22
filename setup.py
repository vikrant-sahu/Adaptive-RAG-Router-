# File: setup.py
"""
Setup script for Adaptive RAG Router package
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="adaptive-rag-router",
    version="1.0.0",
    author="Adaptive RAG Team",
    author_email="your-email@example.com",
    description="Production-ready Adaptive RAG Router with PEFT-based intent classification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/adaptive-rag-router",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
        ],
        "benchmark": [
            "openai>=1.0.0",
            "anthropic>=0.7.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "adaptive-rag-train=training.train_models:main",
            "adaptive-rag-benchmark=benchmarks.llm_benchmark:main",
            "adaptive-rag-pipeline=scripts.run_full_pipeline:main",
        ],
    },
)