# 🧮 Manual Completo de Herramientas Clay-CXD

**Versión**: 2.0 (Diciembre 2025)  
**Estado**: Sistema completamente operativo con CRUD semántico

## 📋 Índice de Herramientas

### **🧠 Memoria Core**
1. [remember](#remember) - Guardar información en memoria persistente
2. [recall](#recall) - Buscar y recuperar memorias (SQL tradicional)
3. [recall_cxd](#recall_cxd) - Búsqueda semántica con clasificación CXD
4. [think_with_memory](#think_with_memory) - Procesar input con contexto de memoria

### **🔧 CRUD Avanzado** 
5. [update_memory_guided](#update_memory_guided) - Actualizar memoria con análisis contextual
6. [delete_memory_guided](#delete_memory_guided) - Eliminar memoria con análisis de impacto

### **🎯 Análisis y Clasificación**
7. [classify_cxd](#classify_cxd) - Clasificar texto con framework CXD
8. [status](#status) - Estado del sistema y estadísticas

### **🤔 Metacognición**
9. [socratic_dialogue](#socratic_dialogue) - Auto-cuestionamiento socrático
10. [reflect](#reflect) - Reflexión offline y análisis de patrones
11. [analyze_memory_patterns](#analyze_memory_patterns) - Análisis de patrones en memorias

### **⚙️ Configuración y Utilidades**
12. [bootstrap_synthetic_memories](#bootstrap_synthetic_memories) - Cargar memorias sintéticas
13. [debug_python](#debug_python) - Debug del intérprete Python

---

## 🧠 Memoria Core

### `remember`
**Descripción**: Almacenar información en memoria persistente  
**Archivo**: `clay_remember.py`

**Parámetros**:
```javascript
{
  content: "string",        // REQUERIDO - Contenido a recordar
  memory_type: "string"     // OPCIONAL - Tipo de memoria (default: "interaction")
}
```

**Tipos de memoria válidos**:
- `interaction` - Conversaciones y intercambios
- `golden` - Reglas empíricas para navegación personal
- `reflection` - Análisis y patrones identificados  
- `milestone` - Hitos importantes del proyecto
- `collaboration` - Información sobre colaboradores
- `experiment` - Experimentos y pruebas
- `synthetic` - Sabiduría destilada pre-cargada
- `socratic_dialogue` - Auto-cuestionamientos

**Ejemplo**:
```javascript
remember({
  content: "Siempre admitir incertidumbre en lugar de fingir certeza",
  memory_type: "golden"
})
```

---

### `recall`
**Descripción**: Buscar memorias usando SQL tradicional (keyword matching)  
**Archivo**: `clay_recall.py` (modificado con soporte de tipo)

**Parámetros**:
```javascript
{
  query: "string",          // REQUERIDO - Texto a buscar
  limit: number,            // OPCIONAL - Máximo resultados (default: 5)
  memory_type: "string"     // OPCIONAL - Filtrar por tipo específico
}
```

**Funcionalidades especiales**:
- `query: "all"` + `memory_type: "golden"` → Todas las memorias golden
- Búsqueda keyword exacta en contenido
- Incluye IDs de memoria para curaduría
- Ordenado por fecha de creación (más recientes primero)

**Ejemplos**:
```javascript
// Buscar cualquier memoria que contenga "sprooket"
recall({ query: "sprooket", limit: 10 })

// Obtener TODAS las memorias golden
recall({ query: "all", memory_type: "golden", limit: 50 })

// Buscar en memorias de reflexión específicamente
recall({ query: "pattern", memory_type: "reflection" })
```

---

### `recall_cxd`
**Descripción**: Búsqueda semántica híbrida con clasificación cognitiva CXD  
**Archivo**: `clay_recall_cxd.py`

**Parámetros**:
```javascript
{
  query: "string",           // REQUERIDO - Consulta semántica
  function_filter: "string", // OPCIONAL - Filtro CXD (default: "")
  limit: number              // OPCIONAL - Máximo resultados (default: 5)
}
```

**Filtros CXD disponibles**:
- `"ALL"` - Sin filtro (default)
- `"CONTROL"` - Funciones de búsqueda, filtrado, gestión
- `"CONTEXT"` - Relaciones, referencias, situacional
- `"DATA"` - Procesamiento, análisis, transformación

**Características**:
- 🧮 Búsqueda semántica usando embeddings
- 📚 Expansión con WordNet para sinónimos
- 🎯 Convergencia híbrida semántica + keyword
- ⭐ Bonificaciones por términos originales

**Ejemplo**:
```javascript
// Búsqueda semántica general
recall_cxd({ query: "machine learning concepts", limit: 8 })

// Buscar solo memorias de control/gestión
recall_cxd({ 
  query: "manage project tasks", 
  function_filter: "CONTROL", 
  limit: 5 
})
```

---

### `think_with_memory`
**Descripción**: Procesar input con contexto completo de memoria  
**Archivo**: `clay_think.py`

**Parámetros**:
```javascript
{
  input_text: "string"      // REQUERIDO - Texto a procesar con memoria
}
```

**Funcionalidad**:
- Busca memorias relevantes automáticamente
- Genera respuesta contextualizada
- Proporciona trace del proceso de razonamiento
- Cuenta memorias utilizadas

**Ejemplo**:
```javascript
think_with_memory({
  input_text: "¿Qué habíamos discutido sobre redes neuronales?"
})
```

---

## 🔧 CRUD Avanzado

### `update_memory_guided`
**Descripción**: Actualizar memoria con análisis contextual y cuestionamiento socrático  
**Archivo**: `clay_update_memory_guided.py`

**Parámetros**:
```javascript
{
  memory_id: number         // REQUERIDO - ID de la memoria a actualizar
}
```

**Proceso**:
1. **Análisis contextual semántico** - Encuentra ~8 memorias relacionadas
2. **Cuestionamiento socrático** - Preguntas adaptadas al tipo de memoria
3. **Interfaz interactiva** - Guía para actualización informada
4. **Invalidación de índice** - Regenera embeddings automáticamente

**Contexto semántico incluye**:
- same_type(3) - Memorias similares del mismo tipo
- temporal(3) - Relevancia temporal + semántica
- keyword_overlap(2) - Similitud conceptual directa
- references - Referencias explícitas + dependencias semánticas

**Ejemplo**:
```javascript
update_memory_guided({ memory_id: 189 })
```

---

### `delete_memory_guided`
**Descripción**: Eliminar memoria con análisis de impacto semántico  
**Archivo**: `clay_delete_memory_guided.py`

**Parámetros**:
```javascript
{
  memory_id: number,        // REQUERIDO - ID de la memoria a eliminar
  confirm: boolean          // OPCIONAL - Confirmación explícita (default: false)
}
```

**Análisis de impacto**:
- **Severidad automática**: LOW/MEDIUM/HIGH/CRITICAL
- **Conectividad semántica**: Score 0.0-1.0
- **Referencias semánticas**: Dependencias encontradas
- **Protección golden**: CRITICAL automático para type='golden'

**Proceso**:
1. **Sin confirm**: Solo análisis y advertencias
2. **Con confirm=true**: Ejecuta eliminación tras análisis

**Ejemplo**:
```javascript
// Solo análisis (recomendado primero)
delete_memory_guided({ memory_id: 189 })

// Ejecutar eliminación confirmada
delete_memory_guided({ memory_id: 189, confirm: true })
```

---

## 🎯 Análisis y Clasificación

### `classify_cxd`
**Descripción**: Clasificar texto usando framework cognitivo CXD  
**Archivo**: `clay_classify_cxd.py`

**Parámetros**:
```javascript
{
  text: "string"            // REQUERIDO - Texto a clasificar
}
```

**Framework CXD**:
- **C (Control)**: Búsqueda, filtrado, gestión, decisiones
- **X (Context)**: Relaciones, referencias, situacional  
- **D (Data)**: Procesamiento, análisis, transformación

**Salida**:
- Función cognitiva principal
- Score de confianza (0.0-1.0)
- Descripción de la función

**Ejemplo**:
```javascript
classify_cxd({ text: "Analizar los datos de ventas del último trimestre" })
// Resultado: "D (Data) - Processing, analysis, transformation"
```

---

### `status`
**Descripción**: Estado del sistema Clay y estadísticas de memoria  
**Archivo**: `clay_status.py`

**Parámetros**: Ninguno

**Información proporcionada**:
- Estado general del sistema Clay
- Total de memorias almacenadas
- Memorias recientes (24h)
- Distribución por tipo
- Estado del clasificador CXD
- Información del entorno Python

**Ejemplo**:
```javascript
status()
```

---

## 🤔 Metacognición

### `socratic_dialogue`
**Descripción**: Auto-cuestionamiento socrático para análisis profundo  
**Archivo**: `clay_socratic.py`

**Parámetros**:
```javascript
{
  query: "string",          // REQUERIDO - Tema para análisis socrático
  depth: number             // OPCIONAL - Profundidad 1-5 (default: 3)
}
```

**Proceso**:
- Genera preguntas internas progresivas
- Identifica asunciones y limitaciones
- Proporciona insights sintéticos
- Recomienda acciones basadas en reflexión

**Ejemplo**:
```javascript
socratic_dialogue({
  query: "¿Por qué sigo clasificando mal las golden memories?",
  depth: 4
})
```

---

### `reflect`
**Descripción**: Reflexión offline y análisis de patrones  
**Archivo**: `clay_reflect.py`

**Parámetros**: Ninguno

**Funcionalidad**:
- Identifica patrones cross-memorias
- Genera insights meta-cognitivos
- Análisis de errores recurrentes
- Síntesis de aprendizajes

**Estado**: ⚠️ En desarrollo (ReflectionEngine faltante)

---

### `analyze_memory_patterns`
**Descripción**: Análisis estadístico de patrones en memorias  
**Archivo**: `clay_analyze_patterns.py`

**Parámetros**: Ninguno

**Análisis incluye**:
- Distribución temporal de memorias
- Frecuencia de tipos
- Patrones de contenido
- Métricas de uso

---

## ⚙️ Configuración y Utilidades

### `bootstrap_synthetic_memories`
**Descripción**: Cargar memorias sintéticas fundamentales  
**Archivo**: `clay_bootstrap.py`

**Parámetros**: Ninguno

**Función**:
- Carga sabiduría destilada predefinida
- Inicializa golden memories básicas
- Establece principios fundamentales

---

### `debug_python`
**Descripción**: Debug del intérprete Python utilizado  
**Archivo**: `clay_debug_python.py`

**Parámetros**: Ninguno

**Información de debug**:
- Versión de Python
- Rutas del sistema
- Imports disponibles
- Variables de entorno

---

## 🔧 Patrones de Uso Recomendados

### **Curaduría de Memorias**
```javascript
// 1. Listar todas las golden memories
recall({ query: "all", memory_type: "golden", limit: 50 })

// 2. Analizar una memoria específica
update_memory_guided({ memory_id: 42 })

// 3. Eliminar si es necesario (con precaución)
delete_memory_guided({ memory_id: 42, confirm: true })
```

### **Búsqueda Inteligente**
```javascript
// Keyword exacta para términos específicos
recall({ query: "sprooket método martillo" })

// Semántica para conceptos
recall_cxd({ query: "machine learning deep neural networks" })

// Filtrada por tipo y función cognitiva
recall_cxd({ 
  query: "analysis data processing", 
  function_filter: "DATA" 
})
```

### **Análisis y Reflexión**
```javascript
// Auto-cuestionamiento sobre patrones
socratic_dialogue({ 
  query: "¿Por qué sigo cometiendo el mismo error de clasificación?" 
})

// Procesar con contexto completo
think_with_memory({ 
  input_text: "¿Cómo puedo mejorar mi curaduría de memorias?" 
})
```

---

## ⚠️ Consideraciones Importantes

### **Protecciones del Sistema**
- **Golden memories**: Protección CRITICAL automática
- **Análisis semántico**: Previene eliminaciones problemáticas
- **Invalidación de índice**: Regeneración automática tras cambios

### **Limitaciones Conocidas**
- `reflect()`: ReflectionEngine no implementado
- Clasificador CXD: Puede requerir calibración
- Búsqueda semántica: Ocasional contaminación en resultados

### **Mejores Prácticas**
- Usar `recall` para búsquedas exactas, `recall_cxd` para conceptuales
- Siempre analizar con `update_memory_guided` antes de modificar
- Confirmar eliminaciones solo tras revisión socrática completa
- Mantener golden memories enfocadas en reglas personales, no información sobre otros

---

**🧮 Clay-CXD v2.0** - *Sistema de memoria persistente y cognición contextual*

*Última actualización: 7 Junio 2025*