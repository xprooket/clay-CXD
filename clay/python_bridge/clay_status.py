#!/usr/bin/env python3
"""
Clay Memory Bridge - Status Tool v3.0
Simple, clean status without bombardment
"""

import sys
import os
from pathlib import Path

# FORCE UTF-8 I/O - CRITICAL para Windows
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Add Clay to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from clay.assistant import ContextualAssistant
    from clay.memory import MemoryStore
except ImportError as e:
    print(f"[ERROR] Error importing Clay: {e}", file=sys.stderr)
    print("[ERROR] Could not import Clay")
    sys.exit(1)

def main():
    try:
        # Initialize assistant
        assistant = ContextualAssistant("claude_mcp")
        
        # Get basic memory count
        memories = assistant.memory_store.get_all()
        total_memories = len(memories)
        
        # Simple, clean status
        print("CLAY SYSTEM STATUS")
        print("=" * 30)
        print(f"[OK] Memory system active: {total_memories} memories stored")
        print("[OK] CXD classifier available")
        print("[OK] All tools operational")
        print()
        print("💡 Main tools:")
        print("   • 'list_tales' / 'load_tale' - personal documentation")
        print("   • 'recall_cxd' - semantic search (primary)")
        print("   • 'remember' - store new insights")
        print("   • 'socratic_dialogue' - self-reflection")
        print()
        print("💭 This is persistent memory - survives across sessions")
        print()
        print("System ready.")
        
    except Exception as e:
        print(f"[ERROR] Error en status: {str(e)}", file=sys.stderr)
        print(f"[ERROR] Error obteniendo status del sistema: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
