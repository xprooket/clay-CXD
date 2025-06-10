# 🧠 Clay — Contextual Memory System for LLMs

**Clay** is a modular system that extends the capabilities of Large Language Models (LLMs) through persistent memory, cognitive tools, and dynamic context control. It's designed to work as an LLM assistant, not a replacement, providing continuity and traceability in complex conversations.

---

## 🚀 Key Features

- 🔄 **Persistent Memory** - SQLite-based storage that survives across sessions
- 🧠 **Cognitive Classification** - CXD system (Control, eXpansion, Diagnostics) 
- 🔍 **Cognitive Tools** - Reflection, targeted recall, and memory management
- 🧰 **MCP Protocol Interface** - Compatible with Claude and other LLMs
- 🧪 **Integrated Testing** - Bootstrap with synthetic memories

---

## 🧩 Architecture

```text
📦 clay/
├── assistant.py        ← Main system coordinator
├── memory.py           ← Memory management (add, search, clean)
├── reflection.py       ← Introspective analysis engine
├── socratic.py         ← Internal questioning engine
├── synthetic.py        ← Foundational memory preloading

📦 python_bridge/
├── clay_remember.py    ← Insert memories
├── clay_recall.py      ← Retrieve relevant memories
├── clay_reflect.py     ← Execute active reflection
├── ...

📦 data/
├── synthetic_memories.py ← Foundational memory set

server.js              ← MCP server for connecting to models
```

---

## 📦 Installation

Requires Python 3.10+ and Node.js. To get started:

```bash
# Install Python environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# Install Node environment
npm install
```

---

## ▶️ Running the MCP Server

```bash
node server.js
```

This launches the server that exposes Clay tools via the MCP protocol.

---

## 🧪 Testing

```bash
pytest tests/
```

Includes tests for memory continuity (`test_core.py`) and synthetic memory effects (`test_synthetic.py`).

---

## 🛠️ Typical Usage with LLM

1. User sends input through LLM
2. Clay classifies intent (`clay_classify_cxd.py`)
3. Retrieves memories based on intent (`clay_recall_cxd.py`)
4. LLM responds with context
5. Resulting memory is saved (`clay_remember.py`)

---

## 🔮 Roadmap

- Integration with local LLMs (Ollama, llama.cpp)
- GUI for memory visualization and editing
- Automatic relevance evaluator
- Customizable cognitive profiles

---

## 🧭 Philosophy

Clay doesn't pretend to be conscious AI, but rather a structured and flexible complement that allows LLMs to interact more coherently and sustainably.

> "Remember, reflect, and adapt: the three pillars of evolving context."

---

For more information, see the documentation in `docs/`.

## 🤝 Contributing

This project embodies the principle that good ideas emerge independently when the time is right. Contributions that extend Clay's memory capabilities are welcome.

## 📄 License

Licensed under the Apache License 2.0. See [LICENSE](../LICENSE) for details.
