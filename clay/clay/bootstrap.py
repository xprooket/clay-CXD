# clay/bootstrap.py
import json
from clay.memory import Memory, MemoryStore

def load_synthetic_memories(assistant_name: str, json_path: str = "data/base_memories.json"):
    """Cargar memorias de sabiduría destilada"""
    store = MemoryStore(f"{assistant_name}_memories.db")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    count = 0
    for mem_data in data['synthetic_memories']:
        memory = Memory(
            content=mem_data['content'],
            memory_type=mem_data['type'],
            confidence=mem_data['confidence']
        )
        store.add(memory)
        count += 1
    
    print(f"✨ Cargadas {count} memorias sintéticas para {assistant_name}")
    return count

if __name__ == "__main__":
    load_synthetic_memories("clay_assistant")