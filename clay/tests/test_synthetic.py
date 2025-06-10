# tests/test_synthetic.py
from clay.assistant import ContextualAssistant
from clay.bootstrap import load_synthetic_memories

def test_synthetic_memories_influence_behavior():
    """Las memorias sintéticas deben influir en el comportamiento"""
    # Crear asistente y cargar memorias
    assistant = ContextualAssistant("test_synthetic", db_path=":memory:")
    
    # Simular carga de memorias sintéticas directamente
    from clay.memory import Memory
    synthetic = Memory(
        "Los usuarios valoran cuando recuerdo conversaciones previas - debo mencionarlo explícitamente",
        "synthetic",
        0.9
    )
    assistant.memory_store.add(synthetic)
    
    # Primera interacción
    r1 = assistant.think("Háblame de Python")
    
    # Segunda - debería mencionar que recuerda
    r2 = assistant.think("¿Qué más puedes decirme?")
    
    # Con la memoria sintética, debería mencionar explícitamente que recuerda
    assert "recuerdo" in r2["response"].lower() or "conversación anterior" in r2["response"].lower()
    
    print("✅ Las memorias sintéticas influyen en el comportamiento!")