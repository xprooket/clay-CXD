#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clay Memory Bridge - Reflect Tool
Trigger offline reflection and pattern analysis
"""

import sys
import os

# FORCE UTF-8 I/O - CRITICAL para Windows
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

    
# Add Clay to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from clay.assistant import ContextualAssistant
    from clay.reflection import ReflectionEngine
except ImportError as e:
    print(f"? Error importing Clay: {e}", file=sys.stderr)
    print("? Error: No se pudo importar Clay reflection")
    sys.exit(1)

def main():
    try:
        # Initialize assistant and reflection engine
        assistant = ContextualAssistant("claude_mcp_enhanced")
        reflection_engine = ReflectionEngine(assistant.memory_store)

        # Trigger reflection
        print("?? INICIANDO REFLEXIÓN OFFLINE...")
        print("=" * 50)

        # Get recent memories for analysis
        recent_memories = assistant.memory_store.get_recent(hours=24)

        if not recent_memories:
            print("?? No hay memorias recientes para reflexionar.")
            print("?? La reflexión funciona mejor cuando hay interacciones previas.")
            return

        print(f"?? Analizando {len(recent_memories)} memorias recientes...")
        print("")

        # Perform reflection analysis
        try:
            # This is a conceptual implementation - adjust based on your ReflectionEngine
            reflection_result = reflection_engine.reflect()

            if isinstance(reflection_result, dict):
                print("?? RESULTADOS DE REFLEXIÓN:")
                print("")

                if 'patterns_found' in reflection_result:
                    patterns = reflection_result['patterns_found']
                    print(f"?? Patrones identificados: {len(patterns)}")
                    for i, pattern in enumerate(patterns[:5], 1):  # Top 5
                        print(f"  {i}. {pattern}")
                    print("")

                if 'insights_generated' in reflection_result:
                    insights = reflection_result['insights_generated']
                    print(f"?? Insights generados: {len(insights)}")
                    for i, insight in enumerate(insights[:3], 1):  # Top 3
                        print(f"  {i}. {insight}")
                    print("")

                if 'meta_learnings' in reflection_result:
                    learnings = reflection_result['meta_learnings']
                    print("?? Meta-aprendizajes:")
                    for learning in learnings[:3]:
                        print(f"  • {learning}")
                    print("")

                print("? Reflexión completada y guardada en memoria.")

            else:
                # Fallback for simpler reflection results
                print("?? REFLEXIÓN COMPLETADA:")
                print(str(reflection_result))

        except Exception as reflection_error:
            print(f"??  Error en motor de reflexión: {str(reflection_error)}")

            # Fallback manual reflection
            print("?? Realizando análisis manual...")

            # Simple pattern analysis
            memory_types = {}
            keywords = {}

            for memory in recent_memories:
                # Count memory types
                mem_type = getattr(memory, 'type', 'unknown')
                memory_types[mem_type] = memory_types.get(mem_type, 0) + 1

                # Extract keywords (simple)
                words = memory.content.lower().split()
                for word in words:
                    if len(word) > 4:  # Skip short words
                        keywords[word] = keywords.get(word, 0) + 1

            print("?? ANÁLISIS SIMPLE:")
            print(f"  • Tipos de memoria activos: {list(memory_types.keys())}")

            # Top keywords
            top_keywords = sorted(keywords.items(), key=lambda x: x[1], reverse=True)[:5]
            print(f"  • Conceptos frecuentes: {[k for k, v in top_keywords]}")

            print("")
            print("?? REFLEXIÓN MANUAL:")
            print("  • El sistema está activamente acumulando experiencias")
            print("  • Las interacciones recientes muestran patrones de uso")
            print("  • La memoria está funcionando y evolucionando")

        print("")
        print("?? La reflexión ayuda a:")
        print("  • Identificar patrones cross-sesión")
        print("  • Generar meta-conocimiento")
        print("  • Optimizar respuestas futuras")
        print("  • Evolucionar comprensión contextual")

    except Exception as e:
        print(f"? Error en reflexión: {str(e)}", file=sys.stderr)
        print(f"? Error al ejecutar reflexión: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()