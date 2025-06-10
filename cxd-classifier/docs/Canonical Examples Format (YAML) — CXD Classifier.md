# Canonical Examples Format (YAML) — CXD Classifier

This document comprehensively explains the structure, validation, and usage of the canonical examples file (`canonical_examples.yaml`), which forms a fundamental part of the semantic and meta classifier system operation in CXD.

---

## 🧱 YAML File Structure

A canonical examples file has the following general structure:

```yaml
version: "1.2"
description: "Canonical examples for CXD semantic training"
examples:
  CONTROL:
    - text: "Search for information in the database"
      id: "ctrl_001"
      tags: ["search", "data"]
      category: "search"
      quality_score: 0.9
      created_by: "team_a"
      last_modified: "2025-06-01"
  CONTEXT:
    - text: "This refers to a previous conversation"
      id: "ctx_003"
      tags: ["reference", "conversation"]
      category: "relational"
      quality_score: 0.85
```

---

## 🔍 `examples` Field

This is a dictionary that maps each `CXDFunction` (CONTROL, CONTEXT, DATA) to a list of examples. The value of each entry is a list of `CanonicalExample`.

---

## 🧩 Structure of a `CanonicalExample`

|Field|Type|Required|Description|
|---|---|---|---|
|`text`|`str`|✅|Example text. Semantic core.|
|`id`|`str`|⛔|Unique identifier. Auto-generated if omitted.|
|`tags`|`List[str]`|⛔|List of auxiliary tags.|
|`category`|`str`|⛔|Logical category of the example.|
|`quality_score`|`float`|⛔ (default: 1.0)|Indicates confidence or quality of the example (0.0 to 1.0).|
|`created_by`|`str`|⛔|Original author of the example.|
|`last_modified`|`str (date)`|⛔|Last modification date.|
|`metadata`|`dict[str, Any]`|⛔|Extensible additional information.|

---

## ✅ Validations Performed

During loading by `YamlExampleProvider`:

- Verifies that `examples` contains valid functions (`CONTROL`, `CONTEXT`, `DATA`)
- Each entry must contain non-empty `text`
- If no `id` exists, one is generated (`uuid4()` or hash)
- If fields are malformed, an error is thrown with detailed context
- A `checksum` of the file is computed for semantic cache invalidation

---

## 🔄 System Execution

Examples are internally converted to `CanonicalExample` objects, then are:

1. Converted to `CXDTag` when selected as semantic neighbors
2. Normalized and validated by `CanonicalExampleSet`
3. Embedded and stored in the `VectorStore`

---

## 🔃 Combined Usage (CompositeExampleProvider)

The system allows combining multiple sources:

```python
provider = CompositeExampleProvider([
    YamlExampleProvider("set1.yaml"),
    InMemoryExampleProvider([...]),
])
```

---

## 📂 Expected Location

By default, the file `canonical_examples.yaml` is expected at the path indicated by:

- `CXDConfig.models.examples_file`
- or variable `CXD_EXAMPLES_FILE`

---

## 🛠️ Generation and Maintenance

- You can generate examples manually or through automated tools
- It's recommended to maintain balanced examples per function and category
- You can version the file and maintain different variants per environment (`dev`, `prod`)

---

## 🧪 Complete Minimal Example

```yaml
version: "1.0"
examples:
  DATA:
    - text: "Generate a document summary"
      tags: ["summary", "processing"]
      quality_score: 0.95
      created_by: "model_v1"
      last_modified: "2025-05-20"
```

---

## 📌 Additional Notes

- The order of examples doesn't affect classification, but may influence if you use VectorStore variants with priorities
- If FAISS is used, the index is built after YAML loading, unless there's valid cache
- Changing the YAML automatically invalidates cache (by checksum)

---

End of document