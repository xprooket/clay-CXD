# Internal API Technical Reference — CXD Classifier

This reference summarizes the main classes and functions available in the `CXD Classifier v2.0` system, grouped by functionality.

---

## 🔎 Classifiers

### `LexicalCXDClassifier`

- `classify(text: str) -> CXDSequence`
- Analysis based on patterns, keywords, and configured rules.

### `SemanticCXDClassifier`

- `classify(text: str) -> CXDSequence`
- Uses embeddings + examples for semantic classification

### `OptimizedSemanticCXDClassifier`

- Inherits from the above, adds:
  - `rebuild_index()`
  - `get_cache_info()`
  - `validate_cache_integrity()`

### `MetaCXDClassifier`

- `classify_detailed(text: str) -> MetaClassificationResult`
- Lexical + semantic fusion, concordance logic

### `OptimizedMetaCXDClassifier`

- Inherits from the above
- Extended methods:
  - `get_optimization_stats()`
  - `create_production_classifier(...)`
  - `clear_all_caches()`

---

## 🧠 Types and Structures (`core/types.py`)

### `CXDTag`

- Attributes: `function`, `state`, `confidence`, `evidence`, `metadata`
- Methods: `.is_successful()`, `.strength()`

### `CXDSequence`

- Sequence of `CXDTag`
- Properties: `.pattern`, `.dominant_function`, `.average_confidence`

### `MetaClassificationResult`

- Attributes: `lexical_sequence`, `semantic_sequence`, `final_sequence`, `confidence`, `concordance`, `debug_info`

---

## 🧰 Configuration (`core/config.py`)

### `CXDConfig`

- Loads from YAML, `.env`, environment
- Methods:
  - `load_from_yaml(...)`
  - `save_to_yaml(...)`
- Access:
  - `.models.embedding_model`
  - `.algorithms.thresholds.min_confidence`

---

## 🔌 Providers and Backends

### `create_embedding_model(name: str)`

- Returns `EmbeddingModel` instance
- Options: `sentence-transformers`, `openai`, `mock`

### `create_cached_model(base_model)`

- Adds cache to base model

### `YamlExampleProvider(path)`

- `.load_examples()` → `Dict[CXDFunction, List[CanonicalExample]]`
- `.get_checksum()`

### `create_vector_store(config: VectorStoreConfig)`

- Creates `FAISSVectorStore` or `NumpyVectorStore`

---

## 🧪 Testing (Pytest)

- `tests/conftest.py`: includes fixtures
- `test_classifier.py`, `test_factory.py`, etc.
- Markers: `@pytest.mark.unit`, `integration`, `performance`

---

## 🧩 Factory Methods

### `create_optimized_classifier(config)`

- Returns `OptimizedMetaCXDClassifier`

### `create_fast_classifier(config)`

- Lightweight version for fast tests

---

## 📁 Others (pending or in development)

- `StructuredLogger`, `CacheProvider`, `MetricsCollector` (prepared interfaces)
- `CLI`, `API`, `Explainers`, `FineTuning` (planned structure)

---

## 💡 Recommendation

For a complete and navigable reference, it's recommended to use `mkdocstrings` or `pdoc` on the `cxd_classifier` package.

End of document.