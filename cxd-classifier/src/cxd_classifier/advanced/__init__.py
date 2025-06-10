"""
Advanced features module for CXD Classifier.

Contains cutting-edge features including explainability (XAI),
fine-tuning capabilities, and advanced analysis tools.
"""

from .explainer import (
    CXDExplainer,
    ExplanationResult,
    TokenImportanceAnalyzer,
)

from .fine_tuning import (
    CXDFineTuner,
    FineTuningDataset,
    TrainingConfig,
)

from .analysis import (
    ConcordanceAnalyzer,
    PatternAnalyzer,
    ConfidenceCalibrator,
)

__all__ = [
    # Explainability
    "CXDExplainer",
    "ExplanationResult",
    "TokenImportanceAnalyzer",
    
    # Fine-tuning
    "CXDFineTuner",
    "FineTuningDataset",
    "TrainingConfig",
    
    # Analysis
    "ConcordanceAnalyzer",
    "PatternAnalyzer",
    "ConfidenceCalibrator",
]
