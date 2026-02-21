#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Setup script for UzABSA-LLM package.

For development installation:
    pip install -e .

For production installation:
    pip install .
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Core dependencies
install_requires = [
    "torch>=2.1.0",
    "transformers>=4.45.0",
    "datasets>=2.18.0",
    "accelerate>=0.27.0",
    "peft>=0.10.0",
    "trl>=0.8.0",
    "bitsandbytes>=0.43.0",
    "numpy>=1.24.0",
    "pandas>=2.0.0",
    "scikit-learn>=1.3.0",
    "pyyaml>=6.0",
    "tqdm>=4.66.0",
]

# Development dependencies
dev_requires = [
    "pytest>=8.0.0",
    "pytest-cov>=4.1.0",
    "black>=24.0.0",
    "flake8>=7.0.0",
    "isort>=5.13.0",
    "mypy>=1.8.0",
]

# Notebook dependencies
notebook_requires = [
    "jupyter>=1.0.0",
    "ipykernel>=6.25.0",
    "ipywidgets>=8.1.0",
]

# Logging dependencies
logging_requires = [
    "wandb>=0.16.0",
    "tensorboard>=2.15.0",
]

setup(
    name="uzabsa-llm",
    version="0.1.0",
    author="UzABSA Team",
    author_email="your.email@example.com",
    description="Fine-tuning LLMs for Uzbek Aspect-Based Sentiment Analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/UzABSA-LLM",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/UzABSA-LLM/issues",
        "Source": "https://github.com/yourusername/UzABSA-LLM",
    },
    packages=find_packages(exclude=["tests", "notebooks", "scripts"]),
    python_requires=">=3.10",
    install_requires=install_requires,
    extras_require={
        "dev": dev_requires,
        "notebook": notebook_requires,
        "logging": logging_requires,
        "all": dev_requires + notebook_requires + logging_requires,
    },
    entry_points={
        "console_scripts": [
            "uzabsa-prep=src.data_prep:main",
            "uzabsa-train=scripts.train_unsloth:main",
            "uzabsa-eval=scripts.evaluate:main",
            "uzabsa-infer=src.inference:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    keywords="nlp, absa, uzbek, llm, fine-tuning, sentiment-analysis",
)
