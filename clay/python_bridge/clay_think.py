#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clay Memory Bridge - Think with Memory Tool
Process input with full contextual memory
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
    from clay.assistant import ContextualAssistant
except ImportError as e:
    print(f"? Error importing Clay: {e}", file=sys.stderr)
    print("? Error: No se pudo importar Clay")
    sys.exit(1)

def main():
    if len(sys.argv) < 2:
        print("? Error: Falta input text")
        sys.exit(1)

    try:
        input_text = sys.argv[1]

        # Initialize assistant
        assistant = ContextualAssistant("claude_mcp")

        # Process with memory
        result = assistant.think(input_text)

        # Format response
        response_parts = []

        # Main response
        response_parts.append("?? RESPUESTA CON MEMORIA:")
        response_parts.append("=" * 50)
        response_parts.append(f"?? Input: {input_text}")
        response_parts.append("")

        # Response
        if isinstance(result, dict) and 'response' in result:
            response_parts.append("?? Respuesta:")
            response_parts.append(result['response'])
            response_parts.append("")

            # Memory usage
            if result.get('memories_used', 0) > 0:
                response_parts.append(f"?? Memorias utilizadas: {result['memories_used']}")
                response_parts.append("")

            # Thought process
            if result.get('thought_process'):
                response_parts.append("?? Proceso de razonamiento:")
                if isinstance(result['thought_process'], dict):
                    for key, value in result['thought_process'].items():
                        response_parts.append(f"  • {key}: {value}")
                else:
                    response_parts.append(f"  {result['thought_process']}")
                response_parts.append("")

            # Socratic analysis (if available)
            if result.get('socratic_analysis'):
                socratic = result['socratic_analysis']
                response_parts.append("?? ANÁLISIS SOCRÁTICO:")
                response_parts.append(f"  • Preguntas internas: {socratic.get('questions_asked', 0)}")
                response_parts.append(f"  • Insights generados: {socratic.get('insights_generated', 0)}")
                if socratic.get('synthesis'):
                    response_parts.append(f"  • Síntesis: {socratic['synthesis'][:200]}...")
                response_parts.append("")
        else:
            # Fallback for simple string response
            response_parts.append("?? Respuesta:")
            response_parts.append(str(result))

        print("\n".join(response_parts))

    except Exception as e:
        print(f"? Error en procesamiento: {str(e)}", file=sys.stderr)
        print(f"? Error al procesar con memoria: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()