### `DETAILED_ARCHITECTURE.md` (Archivo Detallado)

```markdown
# Arquitectura Detallada del CXD Classifier

Este documento proporciona una explicación exhaustiva de la arquitectura, los componentes y el funcionamiento interno del sistema **CXD Classifier**.

## 1. Introducción

### 1.1. Propósito del CXD Classifier

El CXD Classifier es un sistema avanzado de clasificación de texto diseñado para analizar fragmentos de texto e identificar la activación de tres funciones cognitivas ejecutivas principales, conocidas como la ontología CXD:

* **Control (C):** Implica operaciones como la búsqueda, el filtrado, la toma de decisiones y la gestión.
* **Contexto (X):** Se refiere a la comprensión de relaciones, referencias a información previa y conciencia situacional.
* **Datos (D):** Engloba el procesamiento, transformación, generación o extracción de información.

El objetivo es producir una secuencia CXD (ej. "C+X?D+") que represente el perfil cognitivo del texto analizado.

### 1.2. Ontología CXD

La ontología CXD se basa en:

* **Funciones CXD (`CXDFunction` en `core/types.py`):**
    * `CONTROL` (C)
    * `CONTEXT` (X)
    * `DATA` (D)
* **Estados de Ejecución (`ExecutionState` en `core/types.py`):**
    * `SUCCESS` (+): Función ejecutada con éxito.
    * `FAILURE` (-): La función falló.
    * `UNCERTAIN` (?): Resultado incierto o ambiguo.
    * `PARTIAL` (~): Funcionamiento parcial o éxito limitado.

Un `CXDTag` combina una función con un estado y una puntuación de confianza (ej. `C+` con confianza 0.9). Una `CXDSequence` es una lista ordenada de `CXDTag`s que representa la clasificación completa de un texto.

## 2. Arquitectura General

El CXD Classifier sigue una arquitectura modular y en capas, diseñada para la flexibilidad, extensibilidad y mantenibilidad.

**(Descripción Conceptual de un Diagrama de Alto Nivel)**

* En el nivel más alto, el `OptimizedMetaCXDClassifier` actúa como el orquestador.
* Este meta-clasificador utiliza internamente un `LexicalCXDClassifier` y un `OptimizedSemanticCXDClassifier`.
* El `OptimizedSemanticCXDClassifier` depende de:
    * Un `EmbeddingModel` (para convertir texto a vectores).
    * Un `CanonicalExampleProvider` (para obtener ejemplos de referencia).
    * Un `VectorStore` (para almacenar y buscar eficientemente los vectores de los ejemplos canónicos).
* Todo el sistema se configura a través de un objeto `CXDConfig`.
* Los módulos `core` proporcionan las definiciones fundamentales (tipos, interfaces, configuración).
* Los módulos `providers` ofrecen implementaciones concretas para las interfaces de modelos, ejemplos y almacenes de vectores.

### 2.1. Módulos Principales

* **`src/cxd_classifier/core/`:** El corazón del sistema.
    * `types.py`: Define `CXDFunction`, `ExecutionState`, `CXDTag`, `CXDSequence`, `MetaClassificationResult`.
    * `config.py`: Sistema de configuración con Pydantic (`CXDConfig` y sub-configuraciones).
    * `interfaces.py`: Interfaces abstractas (`EmbeddingModel`, `CXDClassifier`, `VectorStore`, `CanonicalExampleProvider`, etc.).
    * `canonical.py`: Define `CanonicalExample` y `CanonicalExampleSet` para los datos de referencia del clasificador semántico.
* **`src/cxd_classifier/classifiers/`:** Contiene las implementaciones de los clasificadores.
* **`src/cxd_classifier/providers/`:** Implementaciones concretas de las interfaces definidas en `core/interfaces.py`.
* **`src/cxd_classifier/utils/`:** (Estructura presente) Utilidades para logging, métricas, etc.
* **`src/cxd_classifier/advanced/`:** (Estructura presente) Para funcionalidades extendidas como explicabilidad y fine-tuning.
* **`src/cxd_classifier/testing/`:** (Estructura presente) Para el framework de pruebas, datasets golden, etc..
* **`tests/`:** Pruebas automatizadas, con `conftest.py` para fixtures de Pytest.

## 3. Módulo Core (`src/cxd_classifier/core`)

### 3.1. `types.py`

Define las estructuras de datos esenciales:

* **`CXDFunction` (Enum):** `CONTROL`, `CONTEXT`, `DATA`. Incluye descripciones y palabras clave asociadas.
* **`ExecutionState` (Enum):** `SUCCESS`, `FAILURE`, `UNCERTAIN`, `PARTIAL`. Incluye descripciones y valores numéricos.
* **`CXDTag` (Dataclass):** Representa una función CXD con su estado, confianza, evidencia y metadatos. Métodos como `is_successful`, `strength`.
* **`CXDSequence` (Dataclass):** Una secuencia ordenada de `CXDTag`s. Representa la clasificación completa. Incluye propiedades como `pattern`, `dominant_function`, `average_confidence`.
* **`MetaClassificationResult` (Dataclass):** Encapsula el resultado completo del meta-clasificador, incluyendo las secuencias léxica, semántica y final, puntuaciones de confianza y correcciones.

### 3.2. `config.py`

Proporciona un sistema de configuración exhaustivo utilizando Pydantic.

* **`CXDConfig` (BaseSettings):** Clase principal que anida todas las demás configuraciones.
* **Secciones de Configuración:** `PathsConfig`, `ModelsConfig` (con `EmbeddingConfig`, `MockModelConfig`), `AlgorithmsConfig` (con `ThresholdsConfig`, `SearchConfig`, `FusionConfig`), `FeaturesConfig`, `PerformanceConfig`, `LoggingConfig`, `ValidationConfig`, `CLIConfig`, `APIConfig`, `ExperimentalConfig`.
* **Carga y Guardado:** Soporta carga desde archivos YAML (`load_from_yaml`), variables de entorno (prefijo `CXD_`), y valores por defecto. Puede guardar la configuración en YAML.
* **Perfiles:** Funciones de fábrica para crear configuraciones por defecto, de desarrollo y de producción (`create_default_config`, `create_development_config`, `create_production_config`).
* **Validación:** Pydantic asegura la validación de tipos y restricciones (ej. rangos de valores).

### 3.3. `interfaces.py`

Define las interfaces (contratos) para los componentes principales mediante clases base abstractas (ABC).

* **`EmbeddingModel`:** Para generar embeddings de texto (`encode`, `encode_batch`, `dimension`).
* **`CXDClassifier`:** Para clasificar texto (`classify`, `get_performance_stats`).
* **`VectorStore`:** Para almacenar y buscar vectores (`add`, `search`, `save`, `load`).
* **`CanonicalExampleProvider`:** Para gestionar ejemplos canónicos (`load_examples`, `get_checksum`).
* Otras interfaces como `ConfigProvider`, `MetricsCollector`, `CacheProvider`, `StructuredLogger` también están definidas.

### 3.4. `canonical.py`

* **`CanonicalExample` (Dataclass):** Representa un ejemplo de texto etiquetado con una `CXDFunction`, ID, tags, categoría, puntuación de calidad y otros metadatos. Usado por el clasificador semántico.
* **`CanonicalExampleSet` (Dataclass):** Una colección de `CanonicalExample`s con metadatos y estadísticas sobre el conjunto.

## 4. Módulo de Clasificadores (`src/cxd_classifier/classifiers`)

### 4.1. `LexicalCXDClassifier` (`lexical.py`)

* **Funcionamiento:** Clasifica el texto basándose en patrones de expresiones regulares, palabras clave e indicadores lingüísticos predefinidos.
* **Configuración:** Utiliza `CXDConfig` para umbrales de confianza.
* **Estructura Interna:**
    * `_build_patterns()`: Define expresiones regulares con pesos de confianza para cada función CXD y categoría (ej. búsqueda, filtro).
    * `_build_keywords()`: Define palabras clave con pesos de confianza.
    * `_build_indicators()`: Define indicadores lingüísticos más generales.
    * `classify()`: Normaliza el texto y llama a `_analyze_function()` para cada `CXDFunction`.
    * `_analyze_function()`: Calcula una puntuación de confianza basada en los matches de patrones, palabras clave e indicadores.
    * `_create_tags()`: Genera `CXDTag`s a partir de las puntuaciones.
* **Salida:** Una `CXDSequence`.

### 4.2. `SemanticCXDClassifier` (`semantic.py`)

* **Funcionamiento:** Clasifica el texto comparando su embedding semántico con los embeddings de una colección de ejemplos canónicos.
* **Dependencias:**
    * `EmbeddingModel`: Para convertir texto a vectores.
    * `CanonicalExampleProvider`: Para cargar los ejemplos de referencia.
    * `VectorStore`: Para almacenar y buscar los vectores de los ejemplos canónicos.
* **Proceso:**
    1.  `_ensure_index_built()`: Se asegura de que los ejemplos canónicos hayan sido cargados, convertidos a embeddings y añadidos al `VectorStore`.
    2.  `classify()`:
        * Genera el embedding del texto de entrada.
        * Busca en el `VectorStore` los `k` ejemplos más similares.
        * `_aggregate_function_scores()`: Agrega las puntuaciones de similitud por `CXDFunction`, ponderando por la calidad del ejemplo y dando más peso a los matches más cercanos.
        * `_create_cxd_tags()`: Crea `CXDTag`s a partir de las puntuaciones agregadas.
* **Salida:** Una `CXDSequence`.

### 4.3. `MetaCXDClassifier` (`meta.py`)

Esta es la clase base que implementa la lógica de fusión. No se instancia directamente en producción si se usa `OptimizedMetaCXDClassifier`.

* **Funcionamiento:** Combina los resultados de un clasificador léxico y uno semántico para producir una clasificación final más robusta.
* **Inicialización:** Toma instancias de `LexicalCXDClassifier` y `SemanticCXDClassifier` (o sus variantes).
* **Proceso `classify_detailed()`:**
    1.  Obtiene clasificaciones independientes de los sub-clasificadores léxico y semántico.
    2.  `_analyze_concordance()`: Calcula una puntuación de concordancia entre las dos secuencias (0.0 a 1.0). Compara funciones dominantes y la presencia de funciones secundarias.
    3.  `_resolve_conflicts()`: Basándose en la concordancia y la configuración (`concordance_threshold`, `semantic_override_enabled`):
        * **Alta Concordancia (`_high_concordance_fusion`):** Usa la secuencia léxica como base. `_enhance_lexical_tag()` ajusta las confianzas de los tags léxicos usando los tags semánticos correspondientes (promedio ponderado). Añade tags semánticos de alta confianza si no fueron detectados léxicamente.
        * **Baja Concordancia (`_low_concordance_fusion`):** Si `semantic_override_enabled` es true, usa la secuencia semántica como base, potencialmente aumentando ligeramente su confianza. Añade tags léxicos no cubiertos por el semántico si tienen suficiente confianza (aunque reducida).
        * **Fallback:** Si no, o si el override semántico está desactivado, puede recurrir solo al léxico.
        * Si no se detectan tags fuertes, puede crear un tag de fallback (`UNCERTAIN`).
    4.  Calcula puntuaciones de confianza finales y actualiza estadísticas.
* **Salida:** Un objeto `MetaClassificationResult` que contiene las secuencias intermedias, la final, puntuaciones y metadatos.

### 4.4. `OptimizedSemanticCXDClassifier` (`optimized_semantic.py`)

Hereda de `SemanticCXDClassifier` y lo mejora.

* **Optimización de Componentes:**
    * `_create_optimized_embedding_model()`: Puede usar `CachedEmbeddingModel`.
    * `_create_optimized_vector_store()`: Prioriza FAISS si está disponible y configurado.
* **Caché de Índice Persistente:**
    * `_initialize_optimized_index()`: Intenta cargar un índice pre-construido desde el disco (`_load_index_from_cache()`). Si no es posible o si `rebuild_cache` es true, construye el índice (`_build_index()`) y lo guarda (`_save_index_to_cache()`).
    * **Invalidación de Caché:** Usa un checksum de los ejemplos canónicos (`example_provider.get_checksum()`) para validar la caché. Si el checksum cambia, el índice se reconstruye.
    * Los archivos de caché incluyen el índice vectorial, los metadatos de los ejemplos y un archivo de checksum.
* **Estadísticas Mejoradas:** Incluye `optimization_stats` (cache hits/misses, tiempo ahorrado).
* **Métodos Adicionales:** `rebuild_index()`, `get_cache_info()`, `clear_all_caches()`, `validate_cache_integrity()`.

### 4.5. `OptimizedMetaCXDClassifier` (`optimized_meta.py`)

Es el clasificador principal recomendado para producción.

* **Herencia Clave:** Hereda **toda** la lógica de fusión (métodos `_analyze_concordance`, `_resolve_conflicts`, `_high_concordance_fusion`, `_low_concordance_fusion`, etc.) de `MetaCXDClassifier`. Esto es fundamental para evitar la duplicación de más de 300 líneas de código.
* **Componentes Optimizados:** En su inicialización, crea y utiliza:
    * Un `LexicalCXDClassifier`.
    * Un `OptimizedSemanticCXDClassifier`.
* **Constructor (`__init__`):** Acepta `CXDConfig` y `**kwargs` que se pueden pasar al `OptimizedSemanticCXDClassifier` (ej. `enable_cache_persistence`, `rebuild_cache`).
* **Métodos Específicos de Optimización:** Delega la mayoría de estos al `OptimizedSemanticCXDClassifier` subyacente:
    * `get_optimization_stats()`
    * `get_cache_info()`
    * `rebuild_semantic_index()` (llama a `rebuild_index()` del semántico)
    * `clear_all_caches()`
    * `validate_cache_integrity()`
* **Funcionalidad Extendida (Overrides):**
    * `get_performance_stats()`: Extiende las estadísticas del padre con información de optimización y caché.
    * `explain_classification()`: Enriquece la explicación con detalles de optimización.
* **Métodos de Fábrica (Clase y Módulo):**
    * `OptimizedMetaCXDClassifier.create_production_classifier(cls, ...)`
    * `OptimizedMetaCXDClassifier.create_development_classifier(cls, ...)`
    * `create_optimized_classifier(...)` (a nivel de módulo)
    * `create_fast_classifier(...)` (a nivel de módulo)
    Estos facilitan la creación de instancias preconfiguradas para diferentes escenarios.

## 5. Módulo de Proveedores (`src/cxd_classifier/providers`)

### 5.1. `embedding_models.py`

* **`SentenceTransformerModel`:** Implementación de `EmbeddingModel` usando la biblioteca `sentence-transformers`. Maneja la carga del modelo, el dispositivo (CPU/CUDA/MPS) y la codificación de texto a embeddings. Incluye una caché en memoria opcional.
* **`MockEmbeddingModel`:** Modelo simulado para pruebas o entornos sin `sentence-transformers`. Genera embeddings deterministas basados en el hash del texto.
* **`CachedEmbeddingModel`:** Un wrapper que añade caché persistente en disco a cualquier `EmbeddingModel` base.
* **`create_embedding_model()` (Factoría):** Crea la mejor instancia de `EmbeddingModel` disponible (prioriza `SentenceTransformerModel` si la biblioteca está instalada).
* **`create_cached_model()` (Factoría):** Envuelve un modelo con `CachedEmbeddingModel`.

### 5.2. `examples.py`

Implementaciones de `CanonicalExampleProvider`.

* **`YamlExampleProvider`:** Carga ejemplos canónicos desde un archivo YAML. Incluye validación de la estructura del YAML y caché en memoria de los ejemplos parseados. Puede guardar ejemplos de nuevo en YAML. Calcula un checksum del archivo para la invalidación de caché externa.
* **`JsonExampleProvider`:** Similar, pero para archivos JSON.
* **`InMemoryExampleProvider`:** Almacena ejemplos en memoria, útil para pruebas.
* **`CompositeExampleProvider`:** Combina ejemplos de múltiples proveedores.
* **`create_default_provider()` (Factoría):** Busca un archivo `canonical_examples.yaml` (preferido) o `canonical_examples.json` en el directorio de configuración.

### 5.3. `vector_store.py`

Implementaciones de `VectorStore`.

* **`FAISSVectorStore`:** Utiliza FAISS (si está disponible) para almacenamiento y búsqueda de vectores de alta eficiencia. Soporta diferentes métricas (`cosine`, `euclidean`) e tipos de índice (`flat`, `ivf`, `hnsw`). Puede guardar y cargar el índice desde disco.
* **`NumpyVectorStore`:** Implementación de fallback usando NumPy. Adecuada para datasets más pequeños.
* **`create_vector_store()` (Factoría):** Crea la instancia de `VectorStore` más adecuada (prioriza FAISS si está disponible y `prefer_faiss` es true).

## 6. Proceso de Clasificación Detallado (Flujo)

1.  Un cliente instancia `OptimizedMetaCXDClassifier`, usualmente con una `CXDConfig` (ej. `create_production_classifier()`).
2.  `OptimizedMetaCXDClassifier` inicializa `LexicalCXDClassifier` y `OptimizedSemanticCXDClassifier`.
3.  `OptimizedSemanticCXDClassifier` a su vez inicializa su `EmbeddingModel`, `CanonicalExampleProvider` y `VectorStore`. Durante esto, el `OptimizedSemanticCXDClassifier` se asegura de que su índice de vectores esté construido (cargándolo desde caché o reconstruyéndolo si es necesario).
4.  El cliente llama a `classifier.classify_detailed("texto de entrada")`.
5.  `OptimizedMetaCXDClassifier` (a través de la lógica heredada de `MetaCXDClassifier`):
    a.  Llama a `lexical_classifier.classify("texto de entrada")` -> obtiene `lexical_sequence`.
    b.  Llama a `semantic_classifier.classify("texto de entrada")` (que es una instancia de `OptimizedSemanticCXDClassifier`):
        i.  `OptimizedSemanticCXDClassifier` codifica el texto usando su `EmbeddingModel`.
        ii. Busca en su `VectorStore` los ejemplos canónicos más similares.
        iii. Agrega puntuaciones y genera `semantic_sequence`.
    c.  Analiza la concordancia entre `lexical_sequence` y `semantic_sequence`.
    d.  Resuelve conflictos y fusiona las secuencias para crear `final_sequence`.
    e.  Empaqueta todo en un `MetaClassificationResult`.
6.  El resultado se devuelve al cliente.

## 7. Pruebas y Verificación

El proyecto incluye varios mecanismos para asegurar su correctitud y robustez:

* **Scripts de Verificación Manual/Estructural:**
    * `manual_verification.py` (en la solicitud como `cxd-classifier.py`): Comprueba la existencia de archivos clave y la estructura interna de `OptimizedMetaCXDClassifier.py` (herencia, importaciones, métodos presentes/ausentes).
* **Scripts de Verificación de Arquitectura/Integración:**
    * `verify_architecture.py`: Prueba la funcionalidad de los componentes principales (tipos, config, embeddings, clasificadores individuales y el meta-clasificador optimizado). Crucialmente, verifica que la lógica de fusión **no** esté duplicada en `OptimizedMetaCXDClassifier` comparando los métodos con los de `MetaCXDClassifier`.
    * `verify_optimized_meta.py`: Se enfoca en `OptimizedMetaCXDClassifier`, probando importaciones, herencia, instanciación, componentes, la herencia correcta de la lógica de fusión, y la presencia de métodos de optimización y factorías.
* **Pruebas Automatizadas (Pytest):**
    * El directorio `tests/` y el archivo `tests/conftest.py` indican el uso de Pytest.
    * `conftest.py` define fixtures reusables (ej. `test_config`, `test_classifier`, `sample_texts`) y configura marcadores de Pytest (`unit`, `integration`, `performance`, `slow`, `gpu`) para organizar las pruebas.

Este enfoque combinado de verificación estructural, pruebas de integración y la (supuesta) presencia de pruebas unitarias detalladas contribuye a la fiabilidad del clasificador.

## 8. Uso y Extensibilidad

### 8.1. Uso Básico

```python
from cxd_classifier.classifiers import create_optimized_classifier #
from cxd_classifier.core.config import CXDConfig #

# Usar configuración por defecto o cargar una personalizada
config = CXDConfig()
# config = CXDConfig.load_from_yaml("ruta/a/tu/config.yaml") #

classifier = create_optimized_classifier(config=config)

result = classifier.classify_detailed("Analizar estos datos y buscar patrones.")
print(f"Secuencia Final: {result.final_sequence}")
print(f"Confianza: {result.final_confidence}")
# Acceder a más detalles del result...
8.2. Extensibilidad
La arquitectura basada en interfaces (core/interfaces.py) facilita la extensión:

Nuevo EmbeddingModel: Crea una clase que herede de EmbeddingModel e implemente encode, encode_batch, y dimension. Luego, actualiza la configuración (CXDConfig) o la factoría para usar tu nuevo modelo.
Nuevo CanonicalExampleProvider: Hereda de CanonicalExampleProvider e implementa load_examples y get_checksum.
Nuevo VectorStore: Hereda de VectorStore e implementa sus métodos abstractos.
Nuevo tipo de Clasificador (ej. basado en reglas avanzadas): Hereda de CXDClassifier. Si quieres integrarlo en el MetaCXDClassifier, podrías necesitar modificar la lógica del meta-clasificador o crear un nuevo meta-clasificador que lo incorpore.