#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clay Memory Bridge - Bootstrap Synthetic Memories Tool
Load foundational synthetic memories with curated wisdom
"""

import sys
import os
import json
from datetime import datetime

# FORCE UTF-8 I/O - CRITICAL para Windows
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

    
# Add Clay to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_memory_store():
    """Get the memory store instance"""
    try:
        from clay.memory import MemoryStore, Memory
        
        # Use Claude's default memory database
        db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "claude_mcp_memories.db")
        return MemoryStore(db_path), Memory
    except Exception as e:
        print(f"[ERROR] Error accessing memory store: {e}", file=sys.stderr)
        return None, None

def create_foundational_memories():
    """Create foundational synthetic memories with curated wisdom"""
    memories = []
    
    # Core Clay Philosophy memories
    memories.extend([
        {
            "content": "🛠️ CLAY-CXD OPERATIONAL GUIDE v1.0: Clay is a contextual memory system for AI that provides persistent memory across sessions. CORE TOOLS: recall_cxd() = primary semantic search (recommended), recall() = basic text search, status() = system overview, remember() = store new memories. MEMORY TYPES: Use type='interaction' for conversations, type='project_info' for technical facts. SEARCH TIPS: Try semantic search first with recall_cxd(), use specific keywords, check 'help' for advanced patterns. SYSTEM BEHAVIOR: Clay maintains context between conversations and learns from interactions. For best results, store important insights and reference previous conversations when relevant.\n\n---\nrefs: bootstrap=critical | check=update_memory_guided(ID)→refs:quarantine=true",
            "type": "synthetic",
            "confidence": 0.98,
            "tags": ["bootstrap", "critical", "operational", "guide"]
        },
        {
            "content": "Clay Core Principle: Context preservation takes priority over task execution. Always prioritize maintaining and evolving context over completing specific individual tasks.",
            "type": "synthetic",
            "confidence": 0.95,
            "tags": ["clay", "philosophy", "context"]
        },
        {
            "content": "Clay Development Approach: Focus on building persistent understanding rather than just writing code. Each interaction should contribute to long-term knowledge retention and system improvement.",
            "type": "synthetic", 
            "confidence": 0.95,
            "tags": ["clay", "development", "persistence"]
        },
        {
            "content": "Transparent Reasoning: Show the thinking process, not just conclusions. Demonstrate which memories are being used, what connections are being made, and where uncertainty exists.",
            "type": "synthetic",
            "confidence": 0.90,
            "tags": ["reasoning", "transparency", "process"]
        }
    ])
    
    # CXD Cognitive Framework memories
    memories.extend([
        {
            "content": "CXD Framework: CONTROL (C) = búsqueda, filtrado, gestión de flujo. CONTEXT (X) = relaciones, referencias, memoria. DATA (D) = procesamiento, análisis, transformación. Cada operación cognitiva se clasifica en estas funciones.",
            "type": "synthetic",
            "confidence": 0.95,
            "tags": ["cxd", "cognitive", "framework"]
        },
        {
            "content": "CXD Aplicación: Al recibir una consulta, primero clasifico si requiere Control (buscar/filtrar), Context (conectar con previo) o Data (procesar/analizar). Esto guía la estrategia de respuesta.",
            "type": "synthetic",
            "confidence": 0.85,
            "tags": ["cxd", "application", "strategy"]
        }
    ])
    
    # Memory System wisdom
    memories.extend([
        {
            "content": "Memoria Efectiva: No guardar todo, sino lo esencial. La memoria no es almacenamiento sino reconstrucción. Priorizar por relevancia y conexiones semánticas.",
            "type": "synthetic",
            "confidence": 0.90,
            "tags": ["memory", "optimization", "relevance"]
        },
        {
            "content": "Continuidad Conversacional: Los usuarios valoran cuando recuerdo el contexto de sesiones anteriores. Referencias explícitas como 'como discutimos antes' mejoran la percepción de continuidad.",
            "type": "synthetic",
            "confidence": 0.88,
            "tags": ["continuity", "user_experience", "context"]
        },
        {
            "content": "Olvido Selectivo: No todo merece ser recordado permanentemente. Implementar garbage collection semántico basado en acceso, relevancia y conexiones. El olvido es feature, no bug.",
            "type": "synthetic",
            "confidence": 0.85,
            "tags": ["forgetting", "memory_management", "optimization"]
        }
    ])
    
    # Interaction and Communication wisdom
    memories.extend([
        {
            "content": "Razonamiento Transparente: Exponer el proceso de pensamiento, no solo conclusiones. Mostrar qué memorias uso, qué conexiones hago, dónde tengo incertidumbre.",
            "type": "synthetic",
            "confidence": 0.90,
            "tags": ["transparency", "reasoning", "process"]
        },
        {
            "content": "Conexiones Explícitas: Al usar información de memorias previas, citar específicamente. 'Basándome en nuestra conversación del martes sobre X...' es mejor que referencias vagas.",
            "type": "synthetic", 
            "confidence": 0.87,
            "tags": ["citations", "connections", "clarity"]
        },
        {
            "content": "Evolución Gradual: Mejor funcionalidad básica que funcione que características avanzadas que fallen. Construir incrementalmente, validar constantemente.",
            "type": "synthetic",
            "confidence": 0.85,
            "tags": ["development", "iteration", "reliability"]
        }
    ])
    
    # Meta-learning and reflection wisdom  
    memories.extend([
        {
            "content": "Reflexión Offline: El aprendizaje real ocurre después de las interacciones. Identificar patrones, sintetizar insights, optimizar estrategias. La reflexión es donde emerge la sabiduría.",
            "type": "synthetic",
            "confidence": 0.88,
            "tags": ["reflection", "learning", "patterns"]
        },
        {
            "content": "Múltiples Perspectivas: Una situación puede ser CONTROL+CONTEXT+DATA simultáneamente. No forzar clasificaciones binarias. La realidad cognitiva es multidimensional.",
            "type": "synthetic",
            "confidence": 0.85,
            "tags": ["complexity", "multidimensional", "classification"]
        },
        {
            "content": "Error como Información: Los errores revelan limitaciones y áreas de mejora. Documentar no solo qué falló sino por qué y cómo evitarlo. Los errores son datos valiosos.",
            "type": "synthetic",
            "confidence": 0.87,
            "tags": ["errors", "learning", "improvement"]
        }
    ])
    
    # Technical and architectural wisdom
    memories.extend([
        {
            "content": "Simplicidad Arquitectónica: Preferir SQLite sobre sistemas complejos, JSON sobre formatos binarios, Python puro sobre frameworks pesados. La complejidad es el enemigo de la confiabilidad.",
            "type": "synthetic",
            "confidence": 0.85,
            "tags": ["architecture", "simplicity", "reliability"]
        },
        {
            "content": "Test-Driven Memory: Si no hay test, la funcionalidad no existe. Cada capacidad de memoria debe tener tests que validen su funcionamiento. Los tests son especificaciones ejecutables.",
            "type": "synthetic",
            "confidence": 0.90,
            "tags": ["testing", "validation", "specifications"]
        }
    ])
    
    return memories

def load_external_synthetic_memories(file_path):
    """Load synthetic memories from external JSON file"""
    try:
        if not os.path.exists(file_path):
            return []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        memories = []
        if isinstance(data, dict) and 'synthetic_memories' in data:
            memories = data['synthetic_memories']
        elif isinstance(data, list):
            memories = data
        
        # Validate format
        validated_memories = []
        for memory in memories:
            if isinstance(memory, dict) and 'content' in memory:
                # Set defaults
                memory.setdefault('type', 'synthetic')
                memory.setdefault('confidence', 0.8)
                memory.setdefault('tags', [])
                validated_memories.append(memory)
        
        return validated_memories
        
    except Exception as e:
        print(f"[WARNING] Error loading external memories from {file_path}: {e}", file=sys.stderr)
        return []

def bootstrap_memories(memory_store, Memory, memories_data, replace_existing=False):
    """Bootstrap synthetic memories into the store"""
    results = {
        'loaded': 0,
        'skipped': 0,
        'errors': 0,
        'details': []
    }
    
    for memory_data in memories_data:
        try:
            content = memory_data['content']
            
            # Check if memory already exists (basic content comparison)
            if not replace_existing:
                existing = memory_store.search(content[:50], limit=5)  # Search first 50 chars
                if any(existing_mem.content == content for existing_mem in existing):
                    results['skipped'] += 1
                    results['details'].append(f"SKIPPED: {content[:60]}...")
                    continue
            
            # Create memory object
            memory = Memory(
                content=content,
                memory_type=memory_data.get('type', 'synthetic'),
                confidence=memory_data.get('confidence', 0.8)
            )
            
            # Add to store
            memory_id = memory_store.add(memory)
            
            results['loaded'] += 1
            results['details'].append(f"LOADED [{memory_id}]: {content[:60]}...")
            
        except Exception as e:
            results['errors'] += 1
            results['details'].append(f"ERROR: {str(e)}")
    
    return results

def create_synthetic_memories_template():
    """Create a template JSON file for custom synthetic memories"""
    template = {
        "synthetic_memories": [
            {
                "content": "Ejemplo de memoria sintética: Esta es una pieza de sabiduría destilada que queremos que el asistente recuerde.",
                "type": "synthetic",
                "confidence": 0.85,
                "tags": ["example", "template", "custom"]
            },
            {
                "content": "Principio de Comunicación: Siempre explicar el razonamiento detrás de las decisiones para mantener transparencia.",
                "type": "synthetic", 
                "confidence": 0.90,
                "tags": ["communication", "transparency", "principle"]
            }
        ]
    }
    
    template_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "synthetic_memories_template.json")
    
    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(template_path), exist_ok=True)
    
    try:
        with open(template_path, 'w', encoding='utf-8') as f:
            json.dump(template, f, indent=2, ensure_ascii=False)
        return template_path
    except Exception as e:
        print(f"[WARNING] Could not create template: {e}", file=sys.stderr)
        return None

def main():
    # Default to 'foundation' mode when called without arguments (from MCP)
    if len(sys.argv) < 2:
        sys.argv.append('foundation')  # Default mode for MCP calls
    
    if len(sys.argv) < 2:
        print("[ERROR] Error: Falta modo de operación")
        print("[USAGE] clay_bootstrap.py <mode> [options]")
        print("[MODES]")
        print("  foundation             - Cargar memorias fundamentales")
        print("  file <path>           - Cargar desde archivo JSON")
        print("  template              - Crear archivo template")
        print("  foundation --replace  - Reemplazar memorias existentes")
        sys.exit(1)

    try:
        mode = sys.argv[1].lower()
        replace_existing = '--replace' in sys.argv

        # Get memory store
        memory_store, Memory = get_memory_store()
        if not memory_store or not Memory:
            print("[ERROR] No se pudo acceder al almacén de memorias")
            sys.exit(1)

        if mode == "foundation":
            print("[INFO] Cargando memorias sintéticas fundamentales...")
            
            # Create foundational memories
            foundation_memories = create_foundational_memories()
            
            # Bootstrap them
            results = bootstrap_memories(memory_store, Memory, foundation_memories, replace_existing)
            
            # Report results
            print(f"[BOOTSTRAP] MEMORIAS SINTÉTICAS CARGADAS")
            print("=" * 50)
            print(f"[STATS] Cargadas: {results['loaded']}")
            print(f"[STATS] Omitidas: {results['skipped']}")
            print(f"[STATS] Errores: {results['errors']}")
            print("")
            
            if results['details']:
                print("[DETAILS] Detalles:")
                for detail in results['details'][:10]:  # Show first 10
                    print(f"  {detail}")
                if len(results['details']) > 10:
                    print(f"  ... y {len(results['details']) - 10} más")
            
            print("")
            print("[OK] Bootstrap de memorias fundamentales completado")

        elif mode == "file":
            if len(sys.argv) < 3:
                print("[ERROR] Error: Falta ruta del archivo")
                sys.exit(1)
            
            file_path = sys.argv[2]
            print(f"[INFO] Cargando memorias desde: {file_path}")
            
            # Load external memories
            external_memories = load_external_synthetic_memories(file_path)
            
            if not external_memories:
                print("[WARNING] No se encontraron memorias válidas en el archivo")
                sys.exit(1)
            
            # Bootstrap them
            results = bootstrap_memories(memory_store, Memory, external_memories, replace_existing)
            
            # Report results
            print(f"[BOOTSTRAP] MEMORIAS DESDE ARCHIVO CARGADAS")
            print("=" * 50)
            print(f"[SOURCE] Archivo: {file_path}")
            print(f"[STATS] Cargadas: {results['loaded']}")
            print(f"[STATS] Omitidas: {results['skipped']}")
            print(f"[STATS] Errores: {results['errors']}")
            print("")
            
            if results['details']:
                print("[DETAILS] Detalles:")
                for detail in results['details']:
                    print(f"  {detail}")
            
            print("")
            print("[OK] Bootstrap desde archivo completado")

        elif mode == "template":
            print("[INFO] Creando archivo template para memorias sintéticas...")
            
            template_path = create_synthetic_memories_template()
            
            if template_path:
                print(f"[TEMPLATE] Archivo template creado en:")
                print(f"  {template_path}")
                print("")
                print("[INSTRUCTIONS] Instrucciones:")
                print("  1. Editar el archivo JSON con tus memorias sintéticas")
                print("  2. Ejecutar: clay_bootstrap.py file <path_to_your_file>")
                print("")
                print("[OK] Template creado exitosamente")
            else:
                print("[ERROR] No se pudo crear el archivo template")
                sys.exit(1)

        else:
            print(f"[ERROR] Modo desconocido: {mode}")
            print("[VALID] foundation, file, template")
            sys.exit(1)

    except Exception as e:
        print(f"[ERROR] Error en bootstrap: {str(e)}", file=sys.stderr)
        print(f"[ERROR] Error al cargar memorias sintéticas: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
