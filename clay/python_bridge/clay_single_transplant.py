#!/usr/bin/env python3
"""
Clay Memory Single Transplant Tool
Transplanta una memoria específica de temporal a ID original
"""

import sys
import os
import sqlite3

# FORCE UTF-8 I/O
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Add Clay to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from clay.assistant import ContextualAssistant
except ImportError as e:
    print(f"[ERROR] Error importing Clay: {e}", file=sys.stderr)
    sys.exit(1)

def single_transplant(assistant, original_id, temp_id, content, memory_type, original_date):
    """Transplanta UNA memoria específica"""
    try:
        # 1. Eliminar memoria original
        assistant.memory_store.conn.execute(
            "DELETE FROM memories WHERE id = ?", (original_id,)
        )
        
        # 2. Insertar en ID original con SQL directo
        assistant.memory_store.conn.execute(
            """INSERT INTO memories (id, content, type, confidence, created_at, access_count) 
               VALUES (?, ?, ?, ?, ?, ?)""",
            (original_id, content, memory_type, 0.80, original_date, 0)
        )
        
        # 3. Eliminar memoria temporal
        assistant.memory_store.conn.execute(
            "DELETE FROM memories WHERE id = ?", (temp_id,)
        )
        
        # 4. Commit cambios
        assistant.memory_store.conn.commit()
        
        print(f"✅ [SUCCESS] Transplantado: {temp_id} → {original_id}")
        return True
        
    except Exception as e:
        print(f"❌ [ERROR] Fallo transplante {temp_id} → {original_id}: {e}")
        return False

def main():
    if len(sys.argv) != 6:
        print("[ERROR] Uso: clay_single_transplant.py <original_id> <temp_id> <content> <type> <original_date>")
        sys.exit(1)
    
    try:
        original_id = int(sys.argv[1])
        temp_id = int(sys.argv[2])
        content = sys.argv[3]
        memory_type = sys.argv[4]
        original_date = sys.argv[5]
        
        assistant = ContextualAssistant("claude_mcp_enhanced")
        
        success = single_transplant(assistant, original_id, temp_id, content, memory_type, original_date)
        
        if success:
            print(f"🎉 Memoria {original_id} curada exitosamente")
        else:
            print(f"❌ Falló curaduría de memoria {original_id}")
            sys.exit(1)
            
    except Exception as e:
        print(f"[ERROR] Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
