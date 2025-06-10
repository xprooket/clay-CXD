# 🧠 Clay MCP Server — Documentación funcional

Este documento describe el propósito, funcionamiento y arquitectura del archivo `server.js`, que actúa como servidor MCP (Model Context Protocol) para integrar Clay con herramientas LLM y sistemas externos.

---

## 🎯 Propósito

Este script implementa un **servidor MCP** que expone herramientas de memoria Clay como comandos ejecutables desde un protocolo estándar (`@modelcontextprotocol/sdk`). Sirve como **puente entre JavaScript y Python**, conectando módulos LLM con el sistema de memoria persistente de Clay.

---

## 🏗️ Estructura General

- **Librerías usadas**:
  - `@modelcontextprotocol/sdk/server`: implementación MCP
  - `child_process.spawn`: para ejecutar scripts Python desde Node
  - `path`, `fs`, `url`: gestión de rutas y archivos

- **Rutas clave**:
  - `CLAY_DIR`: ruta base del proyecto
  - `PYTHON_SCRIPTS_DIR`: carpeta donde están los scripts Clay (como `clay_remember.py`, `clay_reflect.py`, etc.)

---

## ⚙️ Funcionamiento

### Ejecución de herramientas

Cada vez que un LLM llama a una herramienta registrada, el servidor:
1. Busca el script Python correspondiente (`clay_<tool>.py`)
2. Lo ejecuta con `spawn` y argumentos serializados
3. Captura stdout/stderr
4. Devuelve el resultado al cliente MCP (como texto o JSON)

Incluye lógica de:
- **timeout automático** (30s)
- **control de errores** y trazas con timestamp

### Registro de herramientas

Las herramientas disponibles (como `remember`, `recall`, `reflect`) se definen en un objeto `CLAY_TOOLS` con:
- `name`
- `description`
- `inputSchema` para validación

Estas herramientas son listadas y llamadas vía los handlers de `ListToolsRequestSchema` y `CallToolRequestSchema`.

---

## 🔐 Entorno de ejecución Python

El script ejecuta Python desde una ruta fija (por ahora):
```
D:\claude\clay\venv\Scripts\python.exe
```

Esto se puede parametrizar en versiones futuras. Además, define variables de entorno como:
- `PYTHONPATH = CLAY_DIR`
- `PYTHONUTF8 = 1`

---

## 🚦 Inicialización y Transporte

- Se usa `StdioServerTransport` para comunicar con el protocolo MCP
- El servidor se inicia asincrónicamente y escucha en stdin/stdout
- Captura señales `SIGINT` y `SIGTERM` para apagado limpio

---

## ✅ Estado actual

El servidor es estable, rápido y funcional. Ya es capaz de:
- Ejecutar cualquier herramienta Python ubicada en `python_bridge/`
- Retornar respuestas JSON al sistema MCP
- Loguear errores, tiempos y rutas

---

## 🧩 Próximos pasos posibles

- Parametrizar ruta del intérprete Python vía `.env`
- Añadir más herramientas registradas (actualmente incompleto)
- Posibilidad de exponer herramientas con alias
- Añadir autenticación o validación opcional de llamadas

---

Este archivo es la base de la **interfaz operativa de Clay**. Todo lo que recuerde, reflexione, clasifique o modifique en su memoria, pasa por aquí cuando es invocado desde un entorno MCP o frontend externo.
