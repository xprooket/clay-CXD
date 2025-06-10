
Este documento explica exhaustivamente la estructura, validación y uso del archivo de ejemplos canónicos (`canonical_examples.yaml`), que forma parte fundamental del funcionamiento del clasificador semántico y meta del sistema CXD.

---

## 🧱 Estructura del archivo YAML

Un archivo de ejemplos canónicos tiene la siguiente estructura general:

```yaml
version: "1.2"
description: "Ejemplos canónicos para entrenamiento semántico CXD"
examples:
  CONTROL:
    - text: "Buscar información en la base de datos"
      id: "ctrl_001"
      tags: ["busqueda", "datos"]
      category: "search"
      quality_score: 0.9
      created_by: "equipo_a"
      last_modified: "2025-06-01"
  CONTEXT:
    - text: "Esto se refiere a una conversación anterior"
      id: "ctx_003"
      tags: ["referencia", "conversacion"]
      category: "relacional"
      quality_score: 0.85
```

---

## 🔍 Campo `examples`

Es un diccionario que mapea cada `CXDFunction` (CONTROL, CONTEXT, DATA) a una lista de ejemplos. El valor de cada entrada es una lista de `CanonicalExample`.

---

## 🧩 Estructura de un `CanonicalExample`

|Campo|Tipo|Requerido|Descripción|
|---|---|---|---|
|`text`|`str`|✅|Texto del ejemplo. Núcleo semántico.|
|`id`|`str`|⛔|Identificador único. Autogenerado si se omite.|
|`tags`|`List[str]`|⛔|Lista de etiquetas auxiliares.|
|`category`|`str`|⛔|Categoría lógica del ejemplo.|
|`quality_score`|`float`|⛔ (default: 1.0)|Indica confianza o calidad del ejemplo (0.0 a 1.0).|
|`created_by`|`str`|⛔|Autor original del ejemplo.|
|`last_modified`|`str (fecha)`|⛔|Última fecha de modificación.|
|`metadata`|`dict[str, Any]`|⛔|Información adicional extensible.|

---

## ✅ Validaciones realizadas

Durante la carga por `YamlExampleProvider`:

- Se comprueba que `examples` contenga funciones válidas (`CONTROL`, `CONTEXT`, `DATA`)
    
- Cada entrada debe contener `text`, no vacío.
    
- Si no hay `id`, se genera uno (`uuid4()` o hash)
    
- Si hay campos mal formateados, se lanza error con contexto detallado
    
- Se computa un `checksum` del archivo para invalidación de caché semántica
    

---

## 🔄 Ejecución en el sistema

Los ejemplos se convierten internamente en objetos `CanonicalExample`, luego son:

1. Convertidos a `CXDTag` cuando son seleccionados como vecinos semánticos
    
2. Normalizados y validados por `CanonicalExampleSet`
    
3. Embebidos y almacenados en el `VectorStore`
    

---

## 🔃 Uso combinado (CompositeExampleProvider)

El sistema permite combinar múltiples fuentes:

```python
provider = CompositeExampleProvider([
    YamlExampleProvider("set1.yaml"),
    InMemoryExampleProvider([...]),
])
```

---

## 📂 Ubicación esperada

Por defecto se espera el archivo `canonical_examples.yaml` en la ruta indicada por:

- `CXDConfig.models.examples_file`
    
- o variable `CXD_EXAMPLES_FILE`
    

---

## 🛠️ Generación y mantenimiento

- Puedes generar ejemplos a mano o mediante herramientas automáticas
    
- Se recomienda mantener ejemplos equilibrados por función y categoría
    
- Puedes versionar el archivo y mantener diferentes variantes por entorno (`dev`, `prod`)
    

---

## 🧪 Ejemplo completo mínimo

```yaml
version: "1.0"
examples:
  DATA:
    - text: "Generar un resumen del documento"
      tags: ["resumen", "procesamiento"]
      quality_score: 0.95
      created_by: "modelo_v1"
      last_modified: "2025-05-20"
```

---

## 📌 Notas adicionales

- El orden de los ejemplos no afecta la clasificación, pero puede influir si usas variantes de VectorStore con prioridades
    
- Si se usa FAISS, el índice se construye tras la carga del YAML, salvo que haya caché válida
    
- Cambiar el YAML invalida automáticamente la caché (por checksum)
    

---

Fin del documento