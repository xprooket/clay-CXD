# 📁 clay/ — Core Clay Logic Modules

This directory contains the fundamental modules of the Clay persistent memory system. Here are implemented the reflection engines, socratic dialogue, synthetic memory, and basic memory management logic. The "assistant" coordinator also lives here.

---

## 📚 Directory Contents

| File | Purpose |
|------|---------|
| `__init__.py` | Module initialization. Defines version (`0.1.0`). |
| `assistant.py` | Main system controller: executes cognitive actions and connects memory, reflection, and dialogue. |
| `bootstrap.py` | Loads synthetic memories from JSON at system startup. |
| `memory.py` | Defines `Memory` and `MemoryStore`, the foundation of the persistent storage system. |
| `reflection.py` | `ReflectionEngine`: analyzes patterns, gaps, and themes in recent memories. |
| `socratic.py` | `SocraticEngine`: launches self-questioning dialogues when ambiguity or complex topics are detected. |
| `synthetic.py` | `SyntheticMemoryLoader`: loads base wisdom, technical, and historical knowledge. |

---

## 🧠 Key Components

### 🧱 `Memory` and `MemoryStore` (`memory.py`)

- Basic unit: `Memory`
  - `content`, `type`, `confidence`
- SQLite persistence (for now)
- Search, addition, update functions

### 🧠 `ReflectionEngine`

- Runs periodically or on demand
- Detects:
  - Recurring themes
  - Knowledge gaps
  - Confidence trends
- Stores reflections as memories

### 🧠 `SocraticEngine`

- Activated by:
  - Uncertainty
  - Deep questions
  - Unjustified assumptions
- Formulates questions + generates insights
- Stores synthesis as `socratic_dialogue` type memory

### 🧠 `SyntheticMemoryLoader`

- Loads 3 types of synthetic memory:
  - `synthetic_wisdom` (system's base philosophy)
  - `synthetic_technical` (architecture and usage)
  - `synthetic_history` (project origin)

---

## 🔁 Main Interactions

The `assistant.py` module orchestrates the following:

1. `remember(content)` → saves memory
2. `recall(query)` → searches relevant memories
3. `reflect()` → activates `ReflectionEngine`
4. `think_with_memory(input)` → retrieves context + generates response
5. `socratic_analysis(input)` → invokes `SocraticEngine`

---

## 🧩 Future Considerations

- Separate memory types by database (currently everything goes to the same `.db`)
- Add more heuristics for reflection/socratic activation
- Possibility of combining results with embeddings
- Complete assistant modularization

---

This directory is the **functional heart of Clay**. Here resides the cognitive logic that transforms a memory store into a system with continuous learning, introspection, and contextual sense.
