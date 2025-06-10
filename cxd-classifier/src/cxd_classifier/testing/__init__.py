"""
Testing module for CXD Classifier.

Contains testing framework, golden datasets, and optimization utilities
for validating and improving classifier performance.
"""

from .golden_dataset import (
    GoldenDataset,
    GoldenExample,
    DatasetLoader,
)

from .optimization import (
    ThresholdOptimizer,
    HyperparameterOptimizer,
    CrossValidator,
)

from .fixtures import (
    create_test_classifier,
    create_test_config,
    sample_texts,
    expected_results,
)

__all__ = [
    # Golden dataset
    "GoldenDataset",
    "GoldenExample",
    "DatasetLoader",
    
    # Optimization
    "ThresholdOptimizer",
    "HyperparameterOptimizer",
    "CrossValidator",
    
    # Test fixtures
    "create_test_classifier",
    "create_test_config",
    "sample_texts",
    "expected_results",
]
