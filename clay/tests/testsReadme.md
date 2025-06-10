# 🧪 Carpeta `tests/` — Pruebas del sistema Clay

Este directorio contiene pruebas automatizadas para validar el comportamiento básico de Clay y su sistema de memoria. Las pruebas están diseñadas para simular interacciones reales y confirmar que la memoria, tanto persistente como sintética, se comporta como se espera.

---

## ✅ Pruebas incluidas

### `test_core.py`

**Propósito:** Verifica que el asistente recuerde el contexto entre preguntas.

* Usa una base de datos en memoria (`:memory:`) para testeo limpio.
* Simula una interacción sobre transformers.
* Evalúa si el modelo recuerda lo hablado cuando se le hace una segunda pregunta relacionada.

**Resultado esperado:** La respuesta debe hacer uso del contexto anterior. No debe comportarse como una consulta aislada.

---

### `test_synthetic.py`

**Propósito:** Verifica que las memorias sintéticas afectan el comportamiento del asistente.

* Inserta manualmente una entrada `synthetic` en la memoria.
* Simula dos entradas de usuario.
* Evalúa si el sistema hace referencia a una "conversación previa" o "recuerdo" aunque no exista historial real.

**Resultado esperado:** El sistema debería actuar como si "recordara" algo gracias a la memoria sintética precargada.

---

## ▶️ Cómo ejecutar los tests

Desde la raíz del proyecto:

```bash
pytest tests/
```

> Asegúrate de tener `pytest` instalado en el entorno virtual.

---

## 🛠️ Recomendaciones

* Añadir más casos de test para:

  * `golden` memories
  * herramientas `clay_reflect.py` o `clay_think.py`
  * interacciones adversas (baja confianza, borrado, override)

* Usar `pytest.mark.parametrize` para cubrir variaciones del mismo patrón

* Considerar mocks para módulos LLM si se desea aislar el test

---

Estos tests forman parte del ciclo de aseguramiento cognitivo de Clay. Son un buen punto de entrada para validar si el sistema está razonando con continuidad.
