# 🎯 CXD Classifier — Cognitive Function Detection

**Advanced contextual classification with CXD ontology**  
Modular text analysis system based on embeddings, canonical examples, multiclass logic, and activation sequences.

---

## 🧠 Purpose

This system detects implicit cognitive functions in text:

- `C` = **Control** → management, search, decision-making
- `X` = **Context** → relationships, references, environment  
- `D` = **Data** → generation, extraction, transformation

The result is a CXD sequence like: `C+X?D+`

---

## ⚙️ Installation

```bash
pip install -e .[all]
```

### Available extras:

- `[faiss]` → for optimized vector store
- `[openai]` → external API integration
- `[dev]` → development tools, testing, linting

---

## 🧱 Architecture

| Component | Description |
|-----------|-------------|
| `core/` | CXD ontology, configuration, interfaces, and canonical structure |
| `classifiers/` | Lexical, semantic, fusion (meta), and optimized classifiers |
| `providers/` | Embeddings, examples, vector store |
| `cli/`, `utils/` | Auxiliary tools (planned) |
| `tests/` | Reusable fixtures, modular organization (`unit`, `integration`, `performance`) |

📄 See: [`docs/DETAILED_ARCHITECTURE.md`](docs/DETAILED_ARCHITECTURE.md)

---

## 🧪 Usage Example

```python
from cxd_classifier import create_optimized_classifier

clf = create_optimized_classifier()
result = clf.classify("Search for files related to the current project")

print(result.pattern)   # e.g. C+X?D-
print(result.tags)      # CXDTag(function='CONTROL', ...)
```

---

## 🧬 Available Classifiers

| Type | Module | Description |
|------|--------|-------------|
| `lexical` | `lexical.py` | Rules, patterns, and keywords |
| `semantic` | `semantic.py` | Embeddings + examples |
| `meta` | `meta.py` | C+X+D logical fusion |
| `optimized_meta` | `optimized_meta.py` | Performance + heuristics |
| `fast` | `factory.py` | Lightweight version for tests |
| `production` | `factory.py` | Production-ready configuration |

Create via: `CXDClassifierFactory.create(type="optimized_meta", config=...)`

---

## 📂 Canonical Examples (YAML)

Semantic classifiers use a set of function-labeled examples. Example:

```yaml
version: "1.0"
examples:
  CONTROL:
    - text: "Search for relevant information"
      tags: ["search", "data"]
```

📄 See: [`docs/Formato de Ejemplos Canónicos (YAML) — CXD Classifier.md`](docs/Formato%20de%20Ejemplos%20Canónicos%20%28YAML%29%20—%20CXD%20Classifier.md)

---

## 🔌 Configuration

- YAML: `config/cxd_config.yaml`
- Variables: `CXD_CONFIG`, `CXD_CACHE_DIR`, `CXD_MODE`, etc.
- Central system: `CXDConfig` (with override from `.env`, environment, CLI)

📄 See: [`src/cxd_classifier/core/config.py`](src/cxd_classifier/core/config.py)

---

## 🧪 Testing

```bash
pytest -v
```

Available fixtures in `conftest.py`  
Markers: `@pytest.mark.unit`, `integration`, `performance`, `slow`, `gpu`

---

## 📚 Documentation

- [Modular Structure](docs/DETAILED_ARCHITECTURE.md)
- [YAML Examples Format](docs/Formato%20de%20Ejemplos%20Canónicos%20%28YAML%29%20—%20CXD%20Classifier.md)
- [Internal API Reference](docs/Referencia%20Técnica%20del%20API%20Interno%20—%20CXD%20Classifier.md)

---

## 📈 Current Status

- ✅ Core (`core/`)
- ✅ Classifiers (`classifiers/`)
- ✅ Providers (`providers/`)
- ✅ Configuration and testing
- 🛠️ CLI / utils → pending
- 🧪 Explainability / visualization → advanced phase

---

## 🧩 Extensibility

Thanks to the interface-based system, you can extend:

- `EmbeddingModel` → new backends (e.g., HuggingFace)
- `VectorStore` → custom storage
- `ExampleProvider` → external database connections
- `CXDClassifier` → LLM-based or symbolic logic variants

---

## 🧭 Philosophy

Designed to be **intelligible, evaluable, and composable**, respecting principles of semantic traceability and functional control.

---

## 🤝 Contributing

Part of the Clay-CXD project. Contributions that improve classification accuracy or extend cognitive understanding are welcome.

## 📄 License

Licensed under the Apache License 2.0. See [LICENSE](../LICENSE) for details.
