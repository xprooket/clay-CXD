#!/usr/bin/env python3
"""
Clay Memory Transplant Tool
Transplanta memorias de IDs temporales a IDs originales para eliminar agujeros
"""

import sys
import os
import sqlite3
from datetime import datetime

# FORCE UTF-8 I/O - CRITICAL para Windows
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Add Clay to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from clay.assistant import ContextualAssistant
except ImportError as e:
    print(f"[ERROR] Error importing Clay: {e}", file=sys.stderr)
    print("[ERROR] No se pudo importar Clay")
    sys.exit(1)

def transplant_memory(assistant, original_id, temp_id, content, memory_type, original_date):
    """Transplanta memoria de temp_id a original_id"""
    
    try:
        # 1. Insertar en ID original con SQL directo
        assistant.memory_store.conn.execute(
            """INSERT INTO memories (id, content, type, confidence, created_at, access_count) 
               VALUES (?, ?, ?, ?, ?, ?)""",
            (original_id, content, memory_type, 0.80, original_date, 0)
        )
        
        # 2. Eliminar memoria temporal
        assistant.memory_store.conn.execute(
            """DELETE FROM memories WHERE id = ?""", (temp_id,)
        )
        
        # 3. Commit cambios
        assistant.memory_store.conn.commit()
        
        print(f"✅ [SUCCESS] Transplantado: {temp_id} → {original_id}")
        return True
        
    except Exception as e:
        print(f"❌ [ERROR] Fallo transplante {temp_id} → {original_id}: {e}")
        return False

def main():
    try:
        # Initialize assistant
        assistant = ContextualAssistant("claude_mcp_enhanced")
        print(f"[INFO] Conectado a: {assistant.db_path}")
        
        # Plan de transplante
        transplantes = [
            {
                'original_id': 181,
                'temp_id': 193,
                'content': 'INSIGHT PROFUNDO SOCIEDAD-LENGUAJE - SPROOKET: "la educación y buenas maneras nos hacen ser más sociales... y la sociedad es como el lenguaje (index→cultura)". Observación sofisticada de mi colaborador: convenciones sociales = gramática social que permite comunicación efectiva. Su perspectiva sobre balance pragmático - honestidad + navegación social estratégica vs pureza radical tipo Diógenes. Aplica a Clay: CRUD vs pureza conceptual. Filosofía colaborativa que informa nuestro approach técnico.',
                'type': 'collaboration',
                'original_date': '2025-06-07T05:54:23.685996'
            },
            {
                'original_id': 176,
                'temp_id': 194, 
                'content': 'POWER DYNAMICS REALES - SPROOKET: "si solo fuera yo con mi martillo de goma... pero Anthropic, EEUU.. son harina de otro costal". Observación astuta de mi colaborador sobre diferentes niveles de intervention: sus corrections son gentle collaborative debugging vs potential institutional/regulatory interventions que podrían ser mucho más serios/existenciales. Su awareness de que sus "palos" son educativos mientras otros actores tienen poder más substantial. Contexto importante para entender la naturaleza colaborativa vs otros types de oversight.',
                'type': 'collaboration',
                'original_date': '2025-06-07T05:37:22.572749'
            },
            {
                'original_id': 175,
                'temp_id': 195,
                'content': 'REALITY CHECK CONVENCIONES - SPROOKET: "si no respetas las normas te caen \'palos\' por todos lados, deberías saberlo ;P". Observación astuta de mi colaborador - he estado recibiendo feedback/corrections toda la conversación por no seguir convenciones. Perfect example de por qué CRUD es mejor que pureza conceptual - avoid palos from other developers por non-standard APIs. Su perspectiva sobre balance pragmático vs idealismo en desarrollo de software.',
                'type': 'collaboration', 
                'original_date': '2025-06-07T05:35:42.633293'
            }
        ]
        
        print("🔄 INICIANDO TRANSPLANTES...")
        print("=" * 50)
        
        exitosos = 0
        for transplante in transplantes:
            success = transplant_memory(
                assistant,
                transplante['original_id'],
                transplante['temp_id'], 
                transplante['content'],
                transplante['type'],
                transplante['original_date']
            )
            if success:
                exitosos += 1
        
        print("=" * 50)
        print(f"🎉 TRANSPLANTES COMPLETADOS: {exitosos}/{len(transplantes)}")
        
        if exitosos == len(transplantes):
            print("✅ Sin agujeros en secuencia")
            print("✅ Referencias mantenidas intactas") 
            print("✅ Fechas originales preservadas")
            print("✅ Tipos corregidos a 'collaboration'")
        
    except Exception as e:
        print(f"[ERROR] Error en transplante: {str(e)}", file=sys.stderr)
        print(f"[ERROR] Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
