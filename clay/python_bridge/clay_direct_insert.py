#!/usr/bin/env python3
"""
Clay Memory Direct Insert Tool
Inserta memoria directamente en ID específico
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

def direct_insert():
    """Inserta memoria 196 en slot 174 con fecha original"""
    try:
        assistant = ContextualAssistant("claude_mcp_enhanced")
        
        # Contenido de memoria 196 pero tipo collaboration + fecha original de 174
        content = """INSIGHT FUNDAMENTAL SPROOKET - CRUD VS REALIDAD: "conceptualmente solo hay dos: actualizar y consultar". Reducción filosófica brillante de mi colaborador: CREATE/UPDATE/DELETE son todas formas de ACTUALIZAR estado (no existe→existe, estado A→B, existe→no existe). READ = CONSULTAR. Mantenemos CRUD por convención práctica "porque suena mejor" pero conceptualmente solo QUERY vs MODIFY. Su capacidad de ver la esencia detrás de las convenciones técnicas."""
        
        # Insertar directamente en ID 174 con fecha original
        assistant.memory_store.conn.execute(
            """INSERT INTO memories (id, content, type, confidence, created_at, access_count) 
               VALUES (?, ?, ?, ?, ?, ?)""",
            (174, content, 'collaboration', 0.80, '2025-06-07T05:32:50.134444', 0)
        )
        
        # Eliminar memoria temporal 196
        assistant.memory_store.conn.execute(
            "DELETE FROM memories WHERE id = ?", (196,)
        )
        
        assistant.memory_store.conn.commit()
        
        print("✅ [SUCCESS] Memoria 174 restaurada como 'collaboration'")
        print("✅ [SUCCESS] Memoria temporal 196 eliminada")
        print("✅ [SUCCESS] Fecha original preservada: 2025-06-07T05:32:50.134444")
        
    except Exception as e:
        print(f"❌ [ERROR] Fallo: {e}")
        return False

if __name__ == "__main__":
    direct_insert()
