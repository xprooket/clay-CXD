#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clay Memory Bridge - Socratic Dialogue Tool
Simple bridge script for JavaScript MCP server
"""

import sys
import os
import json


# FORCE UTF-8 I/O - CRITICAL for Windows
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

    
# Add Clay to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from clay.memory import Memory, MemoryStore
    from clay.assistant import ContextualAssistant
    from clay.socratic import SocraticEngine
except ImportError as e:
    print(f"[ERROR] Error importing Clay: {e}", file=sys.stderr)
    print("[ERROR] Could not import Clay")
    sys.exit(1)

def main():
    if len(sys.argv) < 2:
        print("[ERROR] Missing arguments")
        sys.exit(1)
    
    try:
        query = sys.argv[1]
        depth = int(sys.argv[2]) if len(sys.argv) > 2 else 3
        
        # Initialize assistant
        assistant = ContextualAssistant("claude_mcp_enhanced")
        
        # Initialize socratic engine
        socratic_engine = SocraticEngine(assistant.memory_store)
        
        # Create mock initial response for self-questioning
        initial_response = f"Initial analysis of topic: {query}"
        
        # Get relevant memories
        relevant_memories = assistant.memory_store.search(query, limit=5)
        
        # Conduct socratic dialogue
        dialogue = socratic_engine.conduct_dialogue(
            user_input=query,
            initial_response=initial_response,
            memories_used=relevant_memories
        )
        
        # Save dialogue as memory
        dialogue_memory = dialogue.to_memory()
        memory_id = assistant.memory_store.add(dialogue_memory)
        
        # Format response
        response = f"""🤔 SOCRATIC DIALOGUE COMPLETED
==================================================
🎯 Query: {query}
📊 Depth: {depth}
❓ Questions generated: {len(dialogue.questions)}
💡 Insights discovered: {len(dialogue.insights)}
🎯 Synthesis: {dialogue.final_synthesis}

💾 Saved as memory ID: {memory_id}

📝 COMPLETE PROCESS:
{dialogue_memory.content}
"""
        
        print(response)
        
    except Exception as e:
        print(f"[ERROR] Error in socratic dialogue: {str(e)}", file=sys.stderr)
        print(f"[ERROR] Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()