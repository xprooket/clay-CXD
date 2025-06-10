"""
🧮 CXD Classifier - Cognitive Executive Dynamics

Advanced hybrid lexical-semantic classifier for analyzing cognitive functions:
- Control (C): Search, filter, decision, management
- Context (X): Relations, references, situational awareness  
- Data (D): Process, transform, generate, extract

This package provides a complete framework for CXD classification with
production-ready features including FAISS indexing, persistent caching,
and intelligent meta-classification.
"""

__version__ = "2.0.0"
__author__ = "CXD Team"
__email__ = "team@cxd.dev"
__license__ = "MIT"

# Core imports for easy access
from .core.types import CXDFunction, ExecutionState, CXDTag, CXDSequence
from .core.config import CXDConfig
from .classifiers.meta import MetaCXDClassifier
from .classifiers.factory import CXDClassifierFactory

# Version information
__all__ = [
    # Version info
    "__version__",
    "__author__", 
    "__email__",
    "__license__",
    
    # Core types
    "CXDFunction",
    "ExecutionState", 
    "CXDTag",
    "CXDSequence",
    
    # Configuration
    "CXDConfig",
    
    # Main classes
    "MetaCXDClassifier",
    "CXDClassifierFactory",
]

# Package metadata
def get_version() -> str:
    """Get the current version of the package."""
    return __version__

def get_info() -> dict:
    """Get package information."""
    return {
        "name": "cxd-classifier",
        "version": __version__,
        "author": __author__,
        "email": __email__,
        "license": __license__,
        "description": "Advanced hybrid lexical-semantic classifier for cognitive executive dynamics",
        "url": "https://github.com/cxd-team/cxd-classifier",
    }

# Convenience factory function
def create_classifier(classifier_type: str = "optimized", **kwargs) -> MetaCXDClassifier:
    """
    Create a CXD classifier instance.
    
    Args:
        classifier_type: Type of classifier ("basic", "optimized", "custom")
        **kwargs: Additional configuration parameters
        
    Returns:
        MetaCXDClassifier: Configured classifier instance
        
    Example:
        >>> classifier = create_classifier("optimized")
        >>> result = classifier.classify("Search for data in the database")
        >>> print(result.final_sequence.pattern)  # e.g., "C+D+"
    """
    return CXDClassifierFactory.create(classifier_type, **kwargs)

# Module-level configuration
import logging
import os

# Set up basic logging if not already configured
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

# Suppress noisy warnings from dependencies
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sentence_transformers")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# Optional: Set environment variable for better numpy performance
os.environ.setdefault("OMP_NUM_THREADS", "1")
