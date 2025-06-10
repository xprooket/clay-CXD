#!/usr/bin/env python3
"""
Clay Memory Bridge - Remember Tool (BACKUP VERSION - WITHOUT CXD)
Simple bridge script for JavaScript MCP server - CLEAN CLAY VERSION

This is the backup version without CXD integration.
Use this file to restore Clay to "clean" state without CXD dependencies.
"""

import sys
import os
import json


# FORCE UTF-8 I/O - CRITICAL para Windows
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    
# Add Clay to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from clay.memory import Memory, MemoryStore
    from clay.assistant import ContextualAssistant
except ImportError as e:
    print(f"[ERROR] Error importing Clay: {e}", file=sys.stderr)
    print("[ERROR] No se pudo importar Clay")
    sys.exit(1)

def main():
    if len(sys.argv) < 2:
        print("[ERROR] Faltan argumentos")
        sys.exit(1)
    
    try:
        content = sys.argv[1]
        memory_type = sys.argv[2] if len(sys.argv) > 2 else "interaction"
        
        # Initialize assistant (use same name as original server)
        assistant = ContextualAssistant("claude_mcp_enhanced")
        
        # Create and store memory
        memory = Memory(content, memory_type)
        memory_id = assistant.memory_store.add(memory)
        
        # Return success message
        result = f"[OK] Guardado en memoria (ID: {memory_id}, Type: {memory_type})"
        print(result)
        
    except Exception as e:
        print(f"[ERROR] Error al guardar memoria: {str(e)}", file=sys.stderr)
        print(f"[ERROR] Error al guardar: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
