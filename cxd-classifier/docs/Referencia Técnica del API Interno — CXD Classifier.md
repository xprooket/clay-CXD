Esta referencia resume las principales clases y funciones disponibles en el sistema `CXD Classifier v2.0`, agrupadas por funcionalidad.

---

## 🔎 Clasificadores

### `LexicalCXDClassifier`

- `classify(text: str) -> CXDSequence`
    
- Análisis basado en patrones, keywords y reglas configuradas.
    

### `SemanticCXDClassifier`

- `classify(text: str) -> CXDSequence`
    
- Usa embeddings + ejemplos para clasificación semántica
    

### `OptimizedSemanticCXDClassifier`

- Hereda del anterior, añade:
    
    - `rebuild_index()`
        
    - `get_cache_info()`
        
    - `validate_cache_integrity()`
        

### `MetaCXDClassifier`

- `classify_detailed(text: str) -> MetaClassificationResult`
    
- Fusión léxica + semántica, lógica de concordancia
    

### `OptimizedMetaCXDClassifier`

- Hereda del anterior
    
- Métodos extendidos:
    
    - `get_optimization_stats()`
        
    - `create_production_classifier(...)`
        
    - `clear_all_caches()`
        

---

## 🧠 Tipos y estructuras (`core/types.py`)

### `CXDTag`

- Atributos: `function`, `state`, `confidence`, `evidence`, `metadata`
    
- Métodos: `.is_successful()`, `.strength()`
    

### `CXDSequence`

- Secuencia de `CXDTag`
    
- Propiedades: `.pattern`, `.dominant_function`, `.average_confidence`
    

### `MetaClassificationResult`

- Atributos: `lexical_sequence`, `semantic_sequence`, `final_sequence`, `confidence`, `concordance`, `debug_info`
    

---

## 🧰 Configuración (`core/config.py`)

### `CXDConfig`

- Carga desde YAML, `.env`, entorno
    
- Métodos:
    
    - `load_from_yaml(...)`
        
    - `save_to_yaml(...)`
        
- Accesos:
    
    - `.models.embedding_model`
        
    - `.algorithms.thresholds.min_confidence`
        

---

## 🔌 Proveedores y backends

### `create_embedding_model(name: str)`

- Retorna instancia de `EmbeddingModel`
    
- Opciones: `sentence-transformers`, `openai`, `mock`
    

### `create_cached_model(base_model)`

- Añade caché a modelo base
    

### `YamlExampleProvider(path)`

- `.load_examples()` → `Dict[CXDFunction, List[CanonicalExample]]`
    
- `.get_checksum()`
    

### `create_vector_store(config: VectorStoreConfig)`

- Crea `FAISSVectorStore` o `NumpyVectorStore`
    

---

## 🧪 Testing (Pytest)

- `tests/conftest.py`: incluye fixtures
    
- `test_classifier.py`, `test_factory.py`, etc.
    
- Marcadores: `@pytest.mark.unit`, `integration`, `performance`
    

---

## 🧩 Factory methods

### `create_optimized_classifier(config)`

- Devuelve `OptimizedMetaCXDClassifier`
    

### `create_fast_classifier(config)`

- Versión liviana para tests rápidos
    

---

## 📁 Otros (pendientes o en desarrollo)

- `StructuredLogger`, `CacheProvider`, `MetricsCollector` (interfaces preparadas)
    
- `CLI`, `API`, `Explainers`, `FineTuning` (estructura prevista)
    

---

## 💡 Recomendación

Para una referencia completa y navegable, se recomienda usar `mkdocstrings` o `pdoc` sobre el paquete `cxd_classifier`.

Fin del documento.