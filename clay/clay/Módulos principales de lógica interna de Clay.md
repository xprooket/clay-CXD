# 📁 clay/ — Módulos principales de lógica interna de Clay

Este directorio contiene los módulos fundamentales del sistema de memoria persistente Clay. Aquí se implementan los motores de reflexión, diálogo socrático, memoria sintética, y lógica básica de gestión de memorias. También vive aquí el "asistente" como coordinador general.

---

## 📚 Contenido del directorio

|Archivo|Propósito|
|---|---|
|`__init__.py`|Inicialización del módulo. Define versión (`0.1.0`).|
|`assistant.py`|Controlador principal del sistema: ejecuta acciones cognitivas y conecta memoria, reflexión y diálogo.|
|`bootstrap.py`|Carga memorias sintéticas desde JSON al iniciar el sistema.|
|`memory.py`|Define `Memory` y `MemoryStore`, la base del sistema de almacenamiento persistente.|
|`reflection.py`|`ReflectionEngine`: analiza patrones, gaps y temas en memorias recientes.|
|`socratic.py`|`SocraticEngine`: lanza diálogos de auto-cuestionamiento cuando se detecta ambigüedad o temas complejos.|
|`synthetic.py`|`SyntheticMemoryLoader`: carga sabiduría base, técnica e histórica.|

---

## 🧠 Componentes clave

### 🧱 `Memory` y `MemoryStore` (`memory.py`)

- Unidad básica: `Memory`
    
    - `content`, `type`, `confidence`
        
- Persistencia con SQLite (por ahora)
    
- Funciones de búsqueda, adición, actualización
    

### 🧠 `ReflectionEngine`

- Se ejecuta periódicamente o a demanda
    
- Detecta:
    
    - Temas recurrentes
        
    - Gaps de conocimiento
        
    - Tendencias de confianza
        
- Almacena reflexiones como memorias
    

### 🧠 `SocraticEngine`

- Se activa en:
    
    - Incertidumbre
        
    - Preguntas profundas
        
    - Asunciones no justificadas
        
- Formula preguntas + genera insights
    
- Almacena síntesis como memoria tipo `socratic_dialogue`
    

### 🧠 `SyntheticMemoryLoader`

- Carga 3 tipos de memoria sintética:
    
    - `synthetic_wisdom` (filosofía base del sistema)
        
    - `synthetic_technical` (arquitectura y uso)
        
    - `synthetic_history` (origen del proyecto)
        

---

## 🔁 Interacciones principales

El módulo `assistant.py` orquesta lo siguiente:

1. `remember(content)` → guarda memoria
    
2. `recall(query)` → busca memorias relevantes
    
3. `reflect()` → activa `ReflectionEngine`
    
4. `think_with_memory(input)` → recupera contexto + genera respuesta
    
5. `socratic_analysis(input)` → invoca `SocraticEngine`
    

---

## 🧩 Consideraciones futuras

- Separar tipos de memoria por base de datos (actualmente todo va al mismo `.db`)
    
- Añadir más heurísticas para activación de reflexión/socrático
    
- Posibilidad de combinar resultados con embeddings
    
- Modularización completa del asistente
    

---

Este directorio es el **corazón funcional de Clay**. Aquí reside la lógica cognitiva que transforma un almacén de memorias en un sistema con aprendizaje continuo, introspección y sentido contextual.