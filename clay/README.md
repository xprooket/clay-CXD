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
# Clone the repository
git clone https://github.com/xprooket/clay-CXD.git
cd Clay-CXD/clay

# Install Python environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# Install Node environment
npm install
```

---

## ⚙️ Claude Desktop Configuration

To use Clay with Claude Desktop, add this configuration to your MCP settings:

### Windows:
```json
{
  "mcpServers": {
    "clay-memory": {
      "command": "node",
      "args": ["C:\\path\\to\\your\\Clay-CXD\\clay\\server.js"],
      "cwd": "C:\\path\\to\\your\\Clay-CXD\\clay",
      "env": {"NODE_PATH": "C:\\path\\to\\your\\Clay-CXD\\clay\\node_modules"}
    }
  }
}
```

### macOS/Linux:
```json
{
  "mcpServers": {
    "clay-memory": {
      "command": "node",
      "args": ["/path/to/your/Clay-CXD/clay/server.js"],
      "cwd": "/path/to/your/Clay-CXD/clay",
      "env": {"NODE_PATH": "/path/to/your/Clay-CXD/clay/node_modules"}
    }
  }
}
```

**Replace the paths** with your actual installation directory.

### Configuration File Locations:

- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Linux**: `~/.config/Claude/claude_desktop_config.json`

---

## ▶️ Running the MCP Server

```bash
node server.js
```

This launches the server that exposes Clay tools via the MCP protocol.

Once configured and running, Clay tools will be available in Claude Desktop:
- `remember` - Store new memories
- `recall_cxd` - Search memories with cognitive filtering
- `think_with_memory` - Process input with memory context
- `reflect` - Trigger pattern analysis
- `socratic_dialogue` - Engage in self-questioning
- And more...

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