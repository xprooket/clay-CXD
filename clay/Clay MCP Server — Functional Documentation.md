# 🧠 Clay MCP Server — Functional Documentation

This document describes the purpose, operation and architecture of the `server.js` file, which acts as an MCP (Model Context Protocol) server to integrate Clay with LLM tools and external systems.

---

## ✨ Latest Features (v2.1.0)

### 🛠️ Auto-Configuration System

**New in v2.1.0**: Clay now automatically prepares fresh instances with essential operational knowledge.

#### How Auto-Configuration Works
1. **First Tool Call Detection**: When `status()` or `recall()` is called for the first time in a session
2. **Session Flag Creation**: Creates `.first_status_shown` or `.first_recall_shown` flag files
3. **Usage Guide Display**: Automatically shows essential operational information
4. **Knowledge Integration**: Provides clear guidance on tool selection and best practices

#### Operational Guide v1.0
- **Tool Selection**: Clear guidance on when to use `recall_cxd()` vs `recall()` and other tools
- **Common Use Cases**: Pre-loaded examples for typical user scenarios
- **Best Practices**: Integrated documentation for optimal system usage
- **Professional Integration**: Consistent interaction patterns for production environments
- **Troubleshooting**: Built-in guidance for common issues and questions

#### Technical Implementation
- **Bootstrap Memories**: Usage guide stored as high-priority synthetic memory
- **Automatic Loading**: Loaded via `bootstrap_synthetic_memories()` with highest confidence
- **Session Persistence**: Flag files prevent repeated guidance in same session
- **Cross-Tool Availability**: Both `clay_status.py` and `clay_recall.py` support auto-configuration

### 🔧 Bug Fixes
- **Search Pattern Matching**: Fixed pattern matching issue that prevented bootstrap memory searches
- **Search Consistency**: `recall("bootstrap=critical")` now works reliably
- **Status Integration**: Session flags properly tracked across both recall and status tools

---

## 🎯 Purpose

This script implements an **MCP server** that exposes Clay memory tools as executable commands from a standard protocol (`@modelcontextprotocol/sdk`). It serves as a **bridge between JavaScript and Python**, connecting LLM modules with Clay's persistent memory system.

---

## 🏗️ General Structure

- **Libraries used**:
  - `@modelcontextprotocol/sdk/server`: MCP implementation
  - `child_process.spawn`: to execute Python scripts from Node
  - `path`, `fs`, `url`: path and file management

- **Key paths**:
  - `CLAY_DIR`: project base path
  - `PYTHON_SCRIPTS_DIR`: folder where Clay scripts are located (like `clay_remember.py`, `clay_reflect.py`, etc.)

---

## ⚙️ Operation

### Tool Execution

Each time an LLM calls a registered tool, the server:
1. Looks for the corresponding Python script (`clay_<tool>.py`)
2. Executes it with `spawn` and serialized arguments
3. Captures stdout/stderr
4. Returns the result to the MCP client (as text or JSON)

Includes logic for:
- **automatic timeout** (30s)
- **error control** and timestamped traces

### Tool Registration

Available tools (like `remember`, `recall`, `reflect`) are defined in a `CLAY_TOOLS` object with:
- `name`
- `description`
- `inputSchema` for validation

These tools are listed and called via the `ListToolsRequestSchema` and `CallToolRequestSchema` handlers.

---

## 🔐 Python Execution Environment

The script executes Python from a fixed path (for now):
```
D:\claude\clay\venv\Scripts\python.exe
```

This can be parameterized in future versions. Additionally, it defines environment variables like:
- `PYTHONPATH = CLAY_DIR`
- `PYTHONUTF8 = 1`

---

## 🚦 Initialization and Transport

- Uses `StdioServerTransport` to communicate with the MCP protocol
- The server starts asynchronously and listens on stdin/stdout
- Captures `SIGINT` and `SIGTERM` signals for clean shutdown

---

## ✅ Current Status

The server is stable, fast and functional. **Now production-ready with v2.1.0 improvements**. It's capable of:
- Executing any Python tool located in `python_bridge/`
- **Auto-configuring fresh instances** with essential operational knowledge
- **Session management** via flag files for consistent user experience
- **Bootstrap memory integration** with reliable pattern matching
- Returning JSON responses to the MCP system
- Logging errors, times and paths

**Production validation**: 24+ GitHub clones, 3+ independent users successfully deployed.

---

## 🧩 Possible Next Steps

- Parameterize Python interpreter path via `.env`
- Add more registered tools (currently incomplete)
- Possibility of exposing tools with aliases
- Add optional authentication or call validation

---

This file is the foundation of **Clay's operational interface**. Everything that remembers, reflects, classifies or modifies in its memory passes through here when invoked from an MCP environment or external frontend.