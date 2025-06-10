# tests/test_core.py
"""
El test más importante: ¿podemos preservar contexto?
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clay.memory import Memory, MemoryStore
from clay.assistant import ContextualAssistant

def test_memory_basics():
    """Verificar que podemos guardar y recuperar memorias"""
    store = MemoryStore(":memory:")  # SQLite en memoria para tests
    
    # Guardar una memoria
    mem = Memory("El usuario preguntó sobre transformers", "interaction")
    mem_id = store.add(mem)
    assert mem_id > 0
    
    # Buscarla
    found = store.search("transformers")
    assert len(found) > 0
    assert "transformers" in found[0].content
    
def test_basic_memory_preservation():
    """El test principal: contexto preservado entre interacciones"""
    assistant = ContextualAssistant("test_assistant", db_path=":memory:")
    
    # Primera interacción
    r1 = assistant.think("Explícame qué es un transformer en IA")
    assert r1["response"] is not None
    
    # Segunda interacción - debe recordar
    r2 = assistant.think("¿Puedes dar más detalles sobre la atención?")
    assert r2["memories_used"] > 0  # Debe haber usado memorias
    assert r2["confidence"] > 0.5  # Debe tener cierta confianza
    
    print("✅ Test pasado: El asistente recuerda!")

if __name__ == "__main__":
    test_memory_basics()
    test_basic_memory_preservation()
    print("🎉 Todos los tests pasaron!")