# Detailed Architecture of CXD Classifier

This document provides a comprehensive explanation of the architecture, components, and internal functioning of the **CXD Classifier** system.

## 1. Introduction

### 1.1. Purpose of CXD Classifier

CXD Classifier is an advanced text classification system designed to analyze text fragments and identify the activation of three main executive cognitive functions, known as the CXD ontology:

* **Control (C):** Involves operations like search, filtering, decision-making, and management.
* **Context (X):** Refers to understanding relationships, references to previous information, and situational awareness.
* **Data (D):** Encompasses processing, transformation, generation, or extraction of information.

The goal is to produce a CXD sequence (e.g., "C+X?D+") that represents the cognitive profile of the analyzed text.

### 1.2. CXD Ontology

The CXD ontology is based on:

* **CXD Functions (`CXDFunction` in `core/types.py`):**
    * `CONTROL` (C)
    * `CONTEXT` (X)
    * `DATA` (D)
* **Execution States (`ExecutionState` in `core/types.py`):**
    * `SUCCESS` (+): Function executed successfully.
    * `FAILURE` (-): Function failed.
    * `UNCERTAIN` (?): Uncertain or ambiguous result.
    * `PARTIAL` (~): Partial functioning or limited success.

A `CXDTag` combines a function with a state and confidence score (e.g., `C+` with confidence 0.9). A `CXDSequence` is an ordered list of `CXDTag`s representing the complete classification of a text.

## 2. General Architecture

CXD Classifier follows a modular and layered architecture, designed for flexibility, extensibility, and maintainability.

**(Conceptual Description of High-Level Diagram)**

* At the highest level, `OptimizedMetaCXDClassifier` acts as the orchestrator.
* This meta-classifier internally uses a `LexicalCXDClassifier` and an `OptimizedSemanticCXDClassifier`.
* The `OptimizedSemanticCXDClassifier` depends on:
    * An `EmbeddingModel` (to convert text to vectors).
    * A `CanonicalExampleProvider` (to obtain reference examples).
    * A `VectorStore` (to efficiently store and search canonical example vectors).
* The entire system is configured through a `CXDConfig` object.
* The `core` modules provide fundamental definitions (types, interfaces, configuration).
* The `providers` modules offer concrete implementations for model, example, and vector store interfaces.

### 2.1. Main Modules

* **`src/cxd_classifier/core/`:** The heart of the system.
    * `types.py`: Defines `CXDFunction`, `ExecutionState`, `CXDTag`, `CXDSequence`, `MetaClassificationResult`.
    * `config.py`: Configuration system with Pydantic (`CXDConfig` and sub-configurations).
    * `interfaces.py`: Abstract interfaces (`EmbeddingModel`, `CXDClassifier`, `VectorStore`, `CanonicalExampleProvider`, etc.).
    * `canonical.py`: Defines `CanonicalExample` and `CanonicalExampleSet` for semantic classifier reference data.
* **`src/cxd_classifier/classifiers/`:** Contains classifier implementations.
* **`src/cxd_classifier/providers/`:** Concrete implementations of interfaces defined in `core/interfaces.py`.
* **`src/cxd_classifier/utils/`:** (Present structure) Utilities for logging, metrics, etc.
* **`src/cxd_classifier/advanced/`:** (Present structure) For extended functionality like explainability and fine-tuning.
* **`src/cxd_classifier/testing/`:** (Present structure) For testing framework, golden datasets, etc.
* **`tests/`:** Automated tests, with `conftest.py` for Pytest fixtures.

## 3. Core Module (`src/cxd_classifier/core`)

### 3.1. `types.py`

Defines essential data structures:

* **`CXDFunction` (Enum):** `CONTROL`, `CONTEXT`, `DATA`. Includes descriptions and associated keywords.
* **`ExecutionState` (Enum):** `SUCCESS`, `FAILURE`, `UNCERTAIN`, `PARTIAL`. Includes descriptions and numerical values.
* **`CXDTag` (Dataclass):** Represents a CXD function with its state, confidence, evidence, and metadata. Methods like `is_successful`, `strength`.
* **`CXDSequence` (Dataclass):** An ordered sequence of `CXDTag`s. Represents complete classification. Includes properties like `pattern`, `dominant_function`, `average_confidence`.
* **`MetaClassificationResult` (Dataclass):** Encapsulates the complete meta-classifier result, including lexical, semantic, and final sequences, confidence scores, and corrections.

### 3.2. `config.py`

Provides a comprehensive configuration system using Pydantic.

* **`CXDConfig` (BaseSettings):** Main class that nests all other configurations.
* **Configuration Sections:** `PathsConfig`, `ModelsConfig` (with `EmbeddingConfig`, `MockModelConfig`), `AlgorithmsConfig` (with `ThresholdsConfig`, `SearchConfig`, `FusionConfig`), `FeaturesConfig`, `PerformanceConfig`, `LoggingConfig`, `ValidationConfig`, `CLIConfig`, `APIConfig`, `ExperimentalConfig`.
* **Loading and Saving:** Supports loading from YAML files (`load_from_yaml`), environment variables (`CXD_` prefix), and default values. Can save configuration to YAML.
* **Profiles:** Factory functions to create default, development, and production configurations (`create_default_config`, `create_development_config`, `create_production_config`).
* **Validation:** Pydantic ensures type validation and constraints (e.g., value ranges).

### 3.3. `interfaces.py`

Defines interfaces (contracts) for main components through abstract base classes (ABC).

* **`EmbeddingModel`:** For generating text embeddings (`encode`, `encode_batch`, `dimension`).
* **`CXDClassifier`:** For classifying text (`classify`, `get_performance_stats`).
* **`VectorStore`:** For storing and searching vectors (`add`, `search`, `save`, `load`).
* **`CanonicalExampleProvider`:** For managing canonical examples (`load_examples`, `get_checksum`).
* Other interfaces like `ConfigProvider`, `MetricsCollector`, `CacheProvider`, `StructuredLogger` are also defined.

### 3.4. `canonical.py`

* **`CanonicalExample` (Dataclass):** Represents a text example labeled with a `CXDFunction`, ID, tags, category, quality score, and other metadata. Used by the semantic classifier.
* **`CanonicalExampleSet` (Dataclass):** A collection of `CanonicalExample`s with metadata and statistics about the set.

## 4. Classifiers Module (`src/cxd_classifier/classifiers`)

### 4.1. `LexicalCXDClassifier` (`lexical.py`)

* **Operation:** Classifies text based on predefined regular expression patterns, keywords, and linguistic indicators.
* **Configuration:** Uses `CXDConfig` for confidence thresholds.
* **Internal Structure:**
    * `_build_patterns()`: Defines regular expressions with confidence weights for each CXD function and category (e.g., search, filter).
    * `_build_keywords()`: Defines keywords with confidence weights.
    * `_build_indicators()`: Defines more general linguistic indicators.
    * `classify()`: Normalizes text and calls `_analyze_function()` for each `CXDFunction`.
    * `_analyze_function()`: Calculates confidence score based on pattern, keyword, and indicator matches.
    * `_create_tags()`: Generates `CXDTag`s from scores.
* **Output:** A `CXDSequence`.

### 4.2. `SemanticCXDClassifier` (`semantic.py`)

* **Operation:** Classifies text by comparing its semantic embedding with embeddings of a canonical examples collection.
* **Dependencies:**
    * `EmbeddingModel`: To convert text to vectors.
    * `CanonicalExampleProvider`: To load reference examples.
    * `VectorStore`: To store and search canonical example vectors.
* **Process:**
    1. `_ensure_index_built()`: Ensures canonical examples have been loaded, converted to embeddings, and added to `VectorStore`.
    2. `classify()`:
        * Generates embedding of input text.
        * Searches `VectorStore` for `k` most similar examples.
        * `_aggregate_function_scores()`: Aggregates similarity scores by `CXDFunction`, weighting by example quality and giving more weight to closer matches.
        * `_create_cxd_tags()`: Creates `CXDTag`s from aggregated scores.
* **Output:** A `CXDSequence`.

### 4.3. `MetaCXDClassifier` (`meta.py`)

This is the base class that implements fusion logic. Not directly instantiated in production if using `OptimizedMetaCXDClassifier`.

* **Operation:** Combines results from a lexical and semantic classifier to produce a more robust final classification.
* **Initialization:** Takes instances of `LexicalCXDClassifier` and `SemanticCXDClassifier` (or their variants).
* **Process `classify_detailed()`:**
    1. Obtains independent classifications from lexical and semantic sub-classifiers.
    2. `_analyze_concordance()`: Calculates concordance score between two sequences (0.0 to 1.0). Compares dominant functions and presence of secondary functions.
    3. `_resolve_conflicts()`: Based on concordance and configuration (`concordance_threshold`, `semantic_override_enabled`):
        * **High Concordance (`_high_concordance_fusion`):** Uses lexical sequence as base. `_enhance_lexical_tag()` adjusts lexical tag confidences using corresponding semantic tags (weighted average). Adds high-confidence semantic tags if not lexically detected.
        * **Low Concordance (`_low_concordance_fusion`):** If `semantic_override_enabled` is true, uses semantic sequence as base, potentially slightly increasing its confidence. Adds lexical tags not covered by semantic if they have sufficient confidence (though reduced).
        * **Fallback:** If not, or if semantic override is disabled, may fall back to lexical only.
        * If no strong tags detected, may create fallback tag (`UNCERTAIN`).
    4. Calculates final confidence scores and updates statistics.
* **Output:** A `MetaClassificationResult` object containing intermediate sequences, final sequence, scores, and metadata.

### 4.4. `OptimizedSemanticCXDClassifier` (`optimized_semantic.py`)

Inherits from `SemanticCXDClassifier` and enhances it.

* **Component Optimization:**
    * `_create_optimized_embedding_model()`: Can use `CachedEmbeddingModel`.
    * `_create_optimized_vector_store()`: Prioritizes FAISS if available and configured.
* **Persistent Index Cache:**
    * `_initialize_optimized_index()`: Attempts to load pre-built index from disk (`_load_index_from_cache()`). If not possible or if `rebuild_cache` is true, builds index (`_build_index()`) and saves it (`_save_index_to_cache()`).
    * **Cache Invalidation:** Uses checksum of canonical examples (`example_provider.get_checksum()`) to validate cache. If checksum changes, index is rebuilt.
    * Cache files include vector index, example metadata, and checksum file.
* **Enhanced Statistics:** Includes `optimization_stats` (cache hits/misses, saved time).
* **Additional Methods:** `rebuild_index()`, `get_cache_info()`, `clear_all_caches()`, `validate_cache_integrity()`.

### 4.5. `OptimizedMetaCXDClassifier` (`optimized_meta.py`)

The main classifier recommended for production.

* **Key Inheritance:** Inherits **all** fusion logic (methods `_analyze_concordance`, `_resolve_conflicts`, `_high_concordance_fusion`, `_low_concordance_fusion`, etc.) from `MetaCXDClassifier`. This is fundamental to avoid duplicating 300+ lines of code.
* **Optimized Components:** In its initialization, creates and uses:
    * A `LexicalCXDClassifier`.
    * An `OptimizedSemanticCXDClassifier`.
* **Constructor (`__init__`):** Accepts `CXDConfig` and `**kwargs` that can be passed to `OptimizedSemanticCXDClassifier` (e.g., `enable_cache_persistence`, `rebuild_cache`).
* **Optimization-Specific Methods:** Delegates most of these to the underlying `OptimizedSemanticCXDClassifier`:
    * `get_optimization_stats()`
    * `get_cache_info()`
    * `rebuild_semantic_index()` (calls semantic's `rebuild_index()`)
    * `clear_all_caches()`
    * `validate_cache_integrity()`
* **Extended Functionality (Overrides):**
    * `get_performance_stats()`: Extends parent statistics with optimization and cache information.
    * `explain_classification()`: Enriches explanation with optimization details.
* **Factory Methods (Class and Module):**
    * `OptimizedMetaCXDClassifier.create_production_classifier(cls, ...)`
    * `OptimizedMetaCXDClassifier.create_development_classifier(cls, ...)`
    * `create_optimized_classifier(...)` (module level)
    * `create_fast_classifier(...)` (module level)
    These facilitate creating pre-configured instances for different scenarios.

## 5. Providers Module (`src/cxd_classifier/providers`)

### 5.1. `embedding_models.py`

* **`SentenceTransformerModel`:** `EmbeddingModel` implementation using `sentence-transformers` library. Handles model loading, device (CPU/CUDA/MPS), and text-to-embedding encoding. Includes optional in-memory cache.
* **`MockEmbeddingModel`:** Mock model for testing or environments without `sentence-transformers`. Generates deterministic embeddings based on text hash.
* **`CachedEmbeddingModel`:** Wrapper that adds persistent disk cache to any base `EmbeddingModel`.
* **`create_embedding_model()` (Factory):** Creates best available `EmbeddingModel` instance (prioritizes `SentenceTransformerModel` if library is installed).
* **`create_cached_model()` (Factory):** Wraps model with `CachedEmbeddingModel`.

### 5.2. `examples.py`

`CanonicalExampleProvider` implementations.

* **`YamlExampleProvider`:** Loads canonical examples from YAML file. Includes YAML structure validation and in-memory cache of parsed examples. Can save examples back to YAML. Calculates file checksum for external cache invalidation.
* **`JsonExampleProvider`:** Similar, but for JSON files.
* **`InMemoryExampleProvider`:** Stores examples in memory, useful for testing.
* **`CompositeExampleProvider`:** Combines examples from multiple providers.
* **`create_default_provider()` (Factory):** Looks for `canonical_examples.yaml` (preferred) or `canonical_examples.json` file in configuration directory.

### 5.3. `vector_store.py`

`VectorStore` implementations.

* **`FAISSVectorStore`:** Uses FAISS (if available) for high-efficiency vector storage and search. Supports different metrics (`cosine`, `euclidean`) and index types (`flat`, `ivf`, `hnsw`). Can save and load index from disk.
* **`NumpyVectorStore`:** Fallback implementation using NumPy. Suitable for smaller datasets.
* **`create_vector_store()` (Factory):** Creates most suitable `VectorStore` instance (prioritizes FAISS if available and `prefer_faiss` is true).

## 6. Detailed Classification Process (Flow)

1. A client instantiates `OptimizedMetaCXDClassifier`, usually with a `CXDConfig` (e.g., `create_production_classifier()`).
2. `OptimizedMetaCXDClassifier` initializes `LexicalCXDClassifier` and `OptimizedSemanticCXDClassifier`.
3. `OptimizedSemanticCXDClassifier` in turn initializes its `EmbeddingModel`, `CanonicalExampleProvider`, and `VectorStore`. During this, `OptimizedSemanticCXDClassifier` ensures its vector index is built (loading from cache or rebuilding if necessary).
4. Client calls `classifier.classify_detailed("input text")`.
5. `OptimizedMetaCXDClassifier` (through logic inherited from `MetaCXDClassifier`):
    a. Calls `lexical_classifier.classify("input text")` -> gets `lexical_sequence`.
    b. Calls `semantic_classifier.classify("input text")` (which is an `OptimizedSemanticCXDClassifier` instance):
        i. `OptimizedSemanticCXDClassifier` encodes text using its `EmbeddingModel`.
        ii. Searches its `VectorStore` for most similar canonical examples.
        iii. Aggregates scores and generates `semantic_sequence`.
    c. Analyzes concordance between `lexical_sequence` and `semantic_sequence`.
    d. Resolves conflicts and fuses sequences to create `final_sequence`.
    e. Packages everything into a `MetaClassificationResult`.
6. Result is returned to client.

## 7. Testing and Verification

The project includes several mechanisms to ensure correctness and robustness:

* **Manual/Structural Verification Scripts:**
    * `manual_verification.py` (in request as `cxd-classifier.py`): Checks existence of key files and internal structure of `OptimizedMetaCXDClassifier.py` (inheritance, imports, present/absent methods).
* **Architecture/Integration Verification Scripts:**
    * `verify_architecture.py`: Tests functionality of main components (types, config, embeddings, individual classifiers, and optimized meta-classifier). Crucially, verifies that fusion logic is **not** duplicated in `OptimizedMetaCXDClassifier` by comparing methods with those of `MetaCXDClassifier`.
    * `verify_optimized_meta.py`: Focuses on `OptimizedMetaCXDClassifier`, testing imports, inheritance, instantiation, components, correct fusion logic inheritance, and presence of optimization methods and factories.
* **Automated Tests (Pytest):**
    * The `tests/` directory and `tests/conftest.py` file indicate Pytest usage.
    * `conftest.py` defines reusable fixtures (e.g., `test_config`, `test_classifier`, `sample_texts`) and configures Pytest markers (`unit`, `integration`, `performance`, `slow`, `gpu`) to organize tests.

This combined approach of structural verification, integration testing, and (presumed) presence of detailed unit tests contributes to classifier reliability.

## 8. Usage and Extensibility

### 8.1. Basic Usage

```python
from cxd_classifier.classifiers import create_optimized_classifier
from cxd_classifier.core.config import CXDConfig

# Use default configuration or load a custom one
config = CXDConfig()
# config = CXDConfig.load_from_yaml("path/to/your/config.yaml")

classifier = create_optimized_classifier(config=config)

result = classifier.classify_detailed("Analyze this data and look for patterns.")
print(f"Final Sequence: {result.final_sequence}")
print(f"Confidence: {result.final_confidence}")
# Access more result details...
```

### 8.2. Extensibility

The interface-based architecture (`core/interfaces.py`) facilitates extension:

* **New EmbeddingModel:** Create a class that inherits from `EmbeddingModel` and implements `encode`, `encode_batch`, and `dimension`. Then, update configuration (`CXDConfig`) or factory to use your new model.
* **New CanonicalExampleProvider:** Inherit from `CanonicalExampleProvider` and implement `load_examples` and `get_checksum`.
* **New VectorStore:** Inherit from `VectorStore` and implement its abstract methods.
* **New Classifier Type (e.g., advanced rule-based):** Inherit from `CXDClassifier`. If you want to integrate it into `MetaCXDClassifier`, you might need to modify meta-classifier logic or create a new meta-classifier that incorporates it.