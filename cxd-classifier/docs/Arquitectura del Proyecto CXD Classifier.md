Este documento describe la estructura modular del sistema `CXD Classifier v2.0` y las responsabilidades de cada componente.

---

## 🧠 Núcleo (`core/`)

| Módulo | Descripción |
|--------|-------------|
| `types.py` | Define los tipos fundamentales del sistema (etiquetas, secuencias, resultados) y utilidades para su manipulación. |
| `interfaces.py` | Define interfaces abstractas para los componentes clave del sistema (modelos, clasificadores, caches, etc.). |
| `config.py` | Carga y valida la configuración global desde YAML, `.env` o variables de entorno, usando `pydantic v2`. |
| `canonical.py` | Gestiona ejemplos canónicos de entrenamiento. Soporta conjuntos, validaciones y operaciones sobre ejemplos. |

---

## 🤖 Clasificadores (`classifiers/`)

| Módulo | Descripción |
|--------|-------------|
| `lexical.py` | Clasificador basado en patrones léxicos explícitos (reglas y búsqueda textual). |
| `semantic.py` | Clasificador semántico basado en embeddings, búsqueda vectorial y ejemplos canónicos. |
| `optimized_semantic.py` | Versión optimizada que integra cache, FAISS, validaciones de integridad y stats. |
| `meta.py` | Clasificador de fusión entre resultados léxicos y semánticos. |
| `optimized_meta.py` | Variante optimizada que incluye evaluación de rendimiento, caché y ajustes dinámicos. |
| `factory.py` | Fábrica que permite construir clasificadores por nombre/configuración (`create_optimized_classifier`, etc.). |
| `__init__.py` | Exporta todos los clasificadores y detecta si `factory.py` está presente. |

---

## 🔌 Proveedores (`providers/`)

| Módulo | Descripción |
|--------|-------------|
| `embedding_models.py` | Proveedores de embeddings: `SentenceTransformer`, `OpenAI`, `Mock`, con cache opcional. |
| `examples.py` | Proveedores de ejemplos canónicos desde YAML, JSON, memoria o múltiples fuentes combinadas. |
| `vector_store.py` | Almacenamiento vectorial con `FAISS` o `NumPy`, configurable y persistente. |

---

## 🛠️ Utilidades y CLI (pendientes)

- `cli/interactive.py`, `cli/cxd.py`: interfaces interactivas y CLI general.
- `utils/logging.py`, `utils/formatting.py`: herramientas auxiliares para logging, formateo, etc.

---

## 🧪 Tests

Estructura prevista:
tests/  
└── unit/  
└── integration/  
└── performance/



---
## ⚙️ Configuración  

- config/cxd_config.yaml: archivo central de configuración.
- Variables `CXD_*` para rutas, modo, caché, etc.``
