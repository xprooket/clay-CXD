
Este documento describe las herramientas ubicadas en el directorio `python_bridge/` del sistema Clay. Todas están diseñadas para gestionar operaciones sobre la memoria del asistente: lectura, escritura, limpieza, análisis o clasificación. Algunas usan CXD para tomar decisiones cognitivas.

---

## 📂 Herramientas Disponibles

### 📂 Inserción / Escritura

#### `clay_remember.py`

Guarda una nueva memoria textual con su tipo e intención.

- Usa CXD para clasificar el tipo si no se especifica.
    
- Ideal para guardar "golden memories" u otras reflexiones.
    

#### `clay_remember_without_cxd.py`

Versión básica que guarda sin pasar por el clasificador CXD.

- Requiere que el tipo sea especificado manualmente.
    
- Útil para test o entradas controladas.
    

#### `clay_direct_insert.py`

Inserta directamente una memoria en la base de datos.

- Permite sobrescribir ID y fecha (peligroso).
    
- Usado para restauraciones manuales o correcciones quirúrgicas.
    

---

### 📊 Lectura / Consulta

#### `clay_recall.py`

Consulta por texto o tipo de memoria en base de datos.

- Opcional: filtro por tipo (`--type`)
    
- Devuelve confianza, acceso y contenido.
    

#### `clay_recall_cxd.py`

Consulta inteligente guiada por CXD.

- Clasifica la intención del input.
    
- Decide si usar búsqueda léxica o semántica, y en qué memoria buscar.
    
- Usa `ClaySemanticSearchEngine`
    

---

### 🖊️ Actualización / Modificación

#### `clay_update_memory_guided.py`

Permite reescribir, cambiar tipo, o actualizar confianza de una memoria.

- Guía al LLM con el contenido original y una instrucción.
    
- Soporta modo interactivo (preguntas de confirmación).
    

#### `clay_bootstrap.py`

Herramienta de inyección masiva de "memorias doradas".

- Puede usarse para inicializar el sistema con sabiduría básica.
    
- Usa un YAML de ejemplos canonizados.
    

---

### 🗑️ Eliminación

#### `clay_delete_memory_guided.py`

Elimina una memoria específica con análisis semántico de impacto.

- Evalúa la conectividad semántica con otras memorias.
    
- Genera advertencias y preguntas socráticas antes de permitir eliminar.
    

---

### 🧬 Análisis / Reflexión

#### `clay_analyze_patterns.py`

Analiza patrones entre memorias seleccionadas.

- Extrae conceptos frecuentes, tipos y correlaciones.
    

#### `clay_reflect.py`

Reflexiona sobre memorias recientes.

- Usa LLM para detectar patrones o insights.
    
- Puede guardar nuevas memorias del tipo "reflection".
    

#### `clay_think.py`

Intenta generar un razonamiento contextual.

- Toma un input y lo procesa usando recuerdos relevantes.
    

#### `clay_socratic.py`

Versión alternativa centrada en preguntas.

- Devuelve preguntas socráticas derivadas del input y memoria activa.
    

---

### 🔢 Utilidades varias

#### `clay_status.py`

Resumen del estado de la base de datos:

- Número total de memorias, tipos, y accesos
    
- Tamaño del archivo y fecha de última escritura
    

#### `clay_classify_cxd.py`

Clasificador manual por CXD.

- Ideal para probar la ontología y analizar decisiones.
    

#### `clay_debug_python.py`

Script abierto para pruebas y debugging.

- Puede ser editado según necesidades.
    

---

## 🔄 Síntesis funcional

|Herramienta|Propósito principal|Usa CXD|Usa LLM|
|---|---|---|---|
|clay_remember|Guardar memoria con clasificación|✅|Opcional|
|clay_remember_without_cxd|Guardar memoria sin clasificación|❌|❌|
|clay_direct_insert|Insertar directamente en BD|❌|❌|
|clay_recall|Búsqueda rápida|❌|❌|
|clay_recall_cxd|Búsqueda guiada|✅|✅|
|clay_update_memory_guided|Modificación guiada por LLM|✅|✅|
|clay_delete_memory_guided|Eliminación guiada con impacto semántico|✅|✅|
|clay_analyze_patterns|Extracción de patrones semánticos|❌|✅|
|clay_reflect|Reflexión contextual|❌|✅|
|clay_think|Pensamiento dirigido|❌|✅|
|clay_socratic|Generación de preguntas|❌|✅|
|clay_status|Diagnóstico del sistema de memoria|❌|❌|
|clay_classify_cxd|Clasificación CXD manual|✅|❌|
|clay_debug_python|Debug libre|?|?|

---

## 🚀 Recomendaciones de uso

- Para cargas iniciales: `clay_bootstrap.py`
    
- Para curación de memorias: `clay_update_memory_guided.py` + `clay_delete_memory_guided.py`
    
- Para operativa habitual: `clay_remember.py`, `clay_recall_cxd.py`
    
- Para introspección: `clay_reflect.py` + `clay_think.py`
    

---

> Este conjunto de herramientas representa la interfaz "viva" entre los razonamientos LLM y la persistencia narrativa de Clay.