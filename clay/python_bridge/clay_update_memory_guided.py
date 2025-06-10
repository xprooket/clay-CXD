#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clay Memory Bridge - Update Memory Guided Tool v2.0
Context-aware memory updating with ADVANCED SEMANTIC SEARCH + Socratic questioning

UPGRADE: Uses ClaySemanticSearchEngine instead of primitive SQL search
- Semantic context discovery with embeddings
- WordNet-enhanced keyword expansion  
- Hybrid scoring for relevance
- CXD cognitive function awareness
"""

import sys
import os
import json
import sqlite3
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path

# FORCE UTF-8 I/O - CRITICAL para Windows
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Add Clay to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from clay.memory import Memory, MemoryStore
    from clay.assistant import ContextualAssistant
    
    # Import advanced semantic search engine
    from python_bridge.clay_recall_cxd import ClaySemanticSearchEngine, safe_decode_text, safe_encode_text
    
except ImportError as e:
    print(f"[ERROR] Error importing Clay: {e}", file=sys.stderr)
    print("[ERROR] No se pudo importar Clay")
    sys.exit(1)

class AdvancedGuidedMemoryManager:
    """Manager for context-aware memory operations using semantic search"""
    
    def __init__(self, assistant):
        self.assistant = assistant
        self.store = assistant.memory_store
        self.semantic_engine = ClaySemanticSearchEngine()
        
        # Initialize semantic index
        try:
            self.semantic_engine.index_memories(self.store, force_rebuild=False)
        except Exception as e:
            print(f"[WARNING] Semantic indexing failed: {e}", file=sys.stderr)
    
    def get_memory_by_id(self, memory_id: int) -> Optional[Memory]:
        """Get specific memory by ID"""
        cursor = self.store.conn.execute(
            "SELECT * FROM memories WHERE id = ?", (memory_id,)
        )
        row = cursor.fetchone()
        if row:
            memory = self.store._row_to_memory(row)
            memory.id = memory_id  # Ensure ID is set
            return memory
        return None
    
    def get_semantic_contextual_memories(self, target_memory: Memory) -> Dict[str, List[Memory]]:
        """
        Get SEMANTIC CONTEXT using advanced search engine:
        - same_type(3) via TYPE-filtered semantic search
        - temporal(3) via temporal window + semantic ranking  
        - keyword_overlap(2) via semantic similarity search
        - references via content-based semantic search
        """
        context = {
            'same_type': [],
            'temporal': [],
            'keyword_overlap': [],
            'references': []
        }
        
        memory_id = getattr(target_memory, 'id', 0)
        content = safe_decode_text(target_memory.content)
        memory_type = target_memory.type
        
        try:
            # 1. SAME TYPE - Use semantic search filtered by type
            all_memories = self.store.get_all()
            same_type_memories = [m for m in all_memories 
                                if m.type == memory_type and getattr(m, 'id', 0) != memory_id]
            
            if same_type_memories and len(content.strip()) > 10:
                # Use semantic similarity for same-type ranking
                semantic_results = self.semantic_engine.search_semantic(content, limit=5)
                for result in semantic_results[:3]:
                    result_memory = result['memory']
                    if (result_memory.type == memory_type and 
                        getattr(result_memory, 'id', 0) != memory_id):
                        context['same_type'].append(result_memory)
                        if len(context['same_type']) >= 3:
                            break
            
            # Fallback to recent same-type if semantic fails
            if len(context['same_type']) < 3:
                cursor = self.store.conn.execute(
                    """SELECT * FROM memories 
                       WHERE type = ? AND id != ?
                       ORDER BY created_at DESC 
                       LIMIT ?""",
                    (memory_type, memory_id, 3 - len(context['same_type']))
                )
                for row in cursor:
                    fallback_memory = self.store._row_to_memory(row)
                    fallback_memory.id = row['id']
                    context['same_type'].append(fallback_memory)
        
        except Exception as e:
            print(f"[WARNING] Same-type semantic search failed: {e}", file=sys.stderr)
            
        try:
            # 2. TEMPORAL - Recent memories ranked by semantic relevance
            recent_memories = self.store.get_recent(hours=168)  # 1 week window
            if recent_memories and len(content.strip()) > 10:
                # Extract key concepts for temporal search
                key_concepts = self._extract_semantic_keywords(content)
                temporal_query = " ".join(key_concepts[:3])  # Top 3 concepts
                
                if temporal_query.strip():
                    semantic_results = self.semantic_engine.search_semantic(temporal_query, limit=5)
                    for result in semantic_results[:3]:
                        result_memory = result['memory']
                        if getattr(result_memory, 'id', 0) != memory_id:
                            context['temporal'].append(result_memory)
                            if len(context['temporal']) >= 3:
                                break
            
            # Fallback to chronological recent
            if len(context['temporal']) < 3:
                cursor = self.store.conn.execute(
                    """SELECT * FROM memories 
                       WHERE id != ?
                       ORDER BY ABS(julianday(created_at) - julianday(?))
                       LIMIT ?""",
                    (memory_id, target_memory.created_at, 3 - len(context['temporal']))
                )
                for row in cursor:
                    fallback_memory = self.store._row_to_memory(row)
                    fallback_memory.id = row['id']
                    context['temporal'].append(fallback_memory)
                    
        except Exception as e:
            print(f"[WARNING] Temporal semantic search failed: {e}", file=sys.stderr)
            
        try:
            # 3. KEYWORD OVERLAP - Direct semantic similarity (most powerful)
            if len(content.strip()) > 10:
                semantic_results = self.semantic_engine.search_semantic(content, limit=7)
                added_count = 0
                for result in semantic_results:
                    result_memory = result['memory']
                    result_id = getattr(result_memory, 'id', 0)
                    
                    # Skip if it's the target or already in other contexts
                    if (result_id != memory_id and
                        result_id not in [getattr(m, 'id', 0) for m in context['same_type']] and
                        result_id not in [getattr(m, 'id', 0) for m in context['temporal']]):
                        
                        context['keyword_overlap'].append(result_memory)
                        added_count += 1
                        if added_count >= 2:
                            break
                            
        except Exception as e:
            print(f"[WARNING] Semantic overlap search failed: {e}", file=sys.stderr)
            
        try:
            # 4. REFERENCES - Search for explicit references + semantic mentions
            ref_patterns = [f"refs: {memory_id}", f"ID: {memory_id}", f"memoria {memory_id}"]
            
            # Direct reference search
            for pattern in ref_patterns:
                try:
                    semantic_results = self.semantic_engine.search_semantic(pattern, limit=3)
                    for result in semantic_results:
                        result_memory = result['memory']
                        if getattr(result_memory, 'id', 0) != memory_id:
                            context['references'].append(result_memory)
                            if len(context['references']) >= 2:
                                break
                    if len(context['references']) >= 2:
                        break
                except Exception:
                    continue
            
            # Fallback to SQL for explicit refs if semantic didn't find them
            if len(context['references']) < 2:
                ref_conditions = " OR ".join(["content LIKE ?" for _ in ref_patterns])
                ref_params = [f"%{pattern}%" for pattern in ref_patterns] + [memory_id]
                
                cursor = self.store.conn.execute(
                    f"""SELECT * FROM memories 
                       WHERE ({ref_conditions}) AND id != ?
                       ORDER BY created_at DESC 
                       LIMIT 2""",
                    ref_params
                )
                for row in cursor:
                    ref_memory = self.store._row_to_memory(row)
                    ref_memory.id = row['id']
                    context['references'].append(ref_memory)
                    
        except Exception as e:
            print(f"[WARNING] Reference search failed: {e}", file=sys.stderr)
            
        return context
    
    def _extract_semantic_keywords(self, content: str) -> List[str]:
        """Extract meaningful keywords optimized for semantic search"""
        # Enhanced keyword extraction for semantic context
        exclude = {
            'que', 'para', 'con', 'una', 'por', 'del', 'las', 'los', 'esto', 'esta', 'ese', 'esa',
            'más', 'como', 'muy', 'todo', 'bien', 'puede', 'hacer', 'ser', 'estar', 'tener',
            'desde', 'hasta', 'entre', 'sobre', 'bajo', 'ante', 'tras', 'durante', 'mediante'
        }
        
        # Split and clean words
        words = content.lower().split()
        
        # Filter for meaningful terms
        keywords = []
        for word in words:
            clean_word = word.strip('.,!?;:()[]"\'')
            if (len(clean_word) > 2 and 
                clean_word not in exclude and
                clean_word.replace('_', '').replace('-', '').isalpha()):
                keywords.append(clean_word)
        
        # Return top meaningful words for semantic search
        return keywords[:7]  # More keywords for better semantic context
    
    def generate_socratic_questions(self, target_memory: Memory, context: Dict) -> List[str]:
        """Generate Socratic questions adapted by memory type with semantic context awareness"""
        questions = []
        memory_id = getattr(target_memory, 'id', '?')
        
        # Universal base questions with semantic awareness
        questions.extend([
            f"¿Qué elementos específicos de esta memoria (ID: {memory_id}) necesitan actualización semántica?",
            "¿Qué información nueva contradice o complementa conceptualmente lo guardado?",
            "¿Cómo se relaciona esta memoria con las similares encontradas semánticamente?"
        ])
        
        # Type-specific questions
        if target_memory.type == "interaction":
            questions.extend([
                "¿Esta conversación representa una evolución conceptual desde la memoria original?",
                "¿Hay matices semánticos en la interacción que no se capturaron inicialmente?"
            ])
        elif target_memory.type == "reflection":
            questions.extend([
                "¿Los patrones conceptuales identificados siguen siendo válidos con nueva información?",
                "¿Qué insights semánticamente relacionados han emergido desde esta reflexión?"
            ])
        elif target_memory.type == "milestone":
            questions.extend([
                "¿Se ha completado realmente este hito o hay aspectos conceptualmente pendientes?",
                "¿El contexto semántico del logro ha cambiado significativamente?"
            ])
        elif target_memory.type == "golden":
            questions.extend([
                "¿Esta golden memory sigue siendo conceptualmente válida o necesita refinamiento semántico?",
                "¿Hay evidencia empírica que confirme o desafíe esta sabiduría fundamental?"
            ])
        elif target_memory.type.startswith("synthetic"):
            questions.extend([
                "¿Esta sabiduría sintética sigue siendo semánticamente relevante?",
                "¿Hay evidencia empírica que confirme o desafíe esta memoria sintética?"
            ])
        else:
            questions.extend([
                "¿Qué aspectos conceptuales de esta memoria han evolucionado con el tiempo?",
                "¿La categorización actual sigue siendo semánticamente apropiada?"
            ])
        
        # Context-aware questions based on semantic findings
        context_count = sum(len(memories) for memories in context.values())
        if context_count > 6:
            questions.append("¿El contexto semántico encontrado sugiere conexiones no obvias?")
        
        if context['same_type']:
            questions.append("¿Cómo se distingue conceptualmente esta memoria de otras del mismo tipo?")
        
        if context['keyword_overlap']:
            questions.append("¿Las memorias semánticamente similares ofrecen perspectivas adicionales?")
            
        if context['references']:
            questions.append("¿Las referencias encontradas mantienen coherencia semántica?")
        
        return questions
    
    def update_memory(self, memory_id: int, new_content: str, new_type: str = None) -> bool:
        """Update memory in database and invalidate semantic index"""
        try:
            if new_type:
                self.store.conn.execute(
                    "UPDATE memories SET content = ?, type = ? WHERE id = ?",
                    (new_content, new_type, memory_id)
                )
            else:
                self.store.conn.execute(
                    "UPDATE memories SET content = ? WHERE id = ?",
                    (new_content, memory_id)
                )
            self.store.conn.commit()
            
            # Invalidate semantic index for this memory
            if hasattr(self.semantic_engine, 'indexed_memory_ids'):
                self.semantic_engine.indexed_memory_ids.discard(memory_id)
            
            return True
        except Exception as e:
            print(f"[ERROR] Database update failed: {e}", file=sys.stderr)
            return False

def main():
    if len(sys.argv) < 2:
        print("[ERROR] Falta memory_id")
        sys.exit(1)
    
    try:
        memory_id = int(sys.argv[1])
        
        # Initialize assistant with advanced semantic search
        assistant = ContextualAssistant("claude_mcp_enhanced")
        manager = AdvancedGuidedMemoryManager(assistant)
        
        # Get target memory
        target_memory = manager.get_memory_by_id(memory_id)
        if not target_memory:
            print(f"[ERROR] Memoria {memory_id} no encontrada")
            sys.exit(1)
        
        # Get SEMANTIC contextual memories (advanced ~8 memories)
        context = manager.get_semantic_contextual_memories(target_memory)
        context_count = sum(len(memories) for memories in context.values())
        
        # Generate Socratic questions with semantic awareness
        questions = manager.generate_socratic_questions(target_memory, context)
        
        # Format guided update interface with semantic indicators
        response_parts = []
        response_parts.append("🔧 ACTUALIZACIÓN GUIADA DE MEMORIA v2.0 (SEMANTIC)")
        response_parts.append("=" * 60)
        response_parts.append(f"📝 MEMORIA OBJETIVO (ID: {memory_id})")
        response_parts.append(f"Tipo: {target_memory.type}")
        response_parts.append(f"Creada: {target_memory.created_at}")
        response_parts.append(f"Contenido actual:")
        content_preview = target_memory.content[:250] + "..." if len(target_memory.content) > 250 else target_memory.content
        response_parts.append(f'"{content_preview}"')
        response_parts.append("")
        
        response_parts.append(f"🧠 CONTEXTO SEMÁNTICO ({context_count} memorias relacionadas)")
        response_parts.append("-" * 40)
        
        context_descriptions = {
            'same_type': 'Mismo Tipo (semánticamente rankeadas)',
            'temporal': 'Temporales (relevancia semántica)', 
            'keyword_overlap': 'Similitud Semántica Directa',
            'references': 'Referencias (explícitas + semánticas)'
        }
        
        for context_type, memories in context.items():
            if memories:
                desc = context_descriptions.get(context_type, context_type)
                response_parts.append(f"🎯 {desc}: {len(memories)} memorias")
                for i, mem in enumerate(memories[:2]):  # Show first 2
                    preview = mem.content[:80] + "..." if len(mem.content) > 80 else mem.content
                    mem_id = getattr(mem, 'id', '?')
                    response_parts.append(f"  {i+1}. (ID: {mem_id}) {preview}")
        response_parts.append("")
        
        response_parts.append("❓ CUESTIONAMIENTO SOCRÁTICO (Semantic-Aware)")
        response_parts.append("-" * 40)
        for i, question in enumerate(questions, 1):
            response_parts.append(f"{i}. {question}")
        response_parts.append("")
        
        response_parts.append("💡 PRÓXIMOS PASOS AVANZADOS:")
        response_parts.append("1. 🧠 Analizar contexto semánticamente relacionado arriba")
        response_parts.append("2. 🤔 Reflexionar sobre preguntas socráticas contextuales")
        response_parts.append("3. 🎯 Formular nuevo contenido con consciencia semántica")
        response_parts.append("4. 🔄 Ejecutar actualización (invalidará índice semántico)")
        response_parts.append("")
        response_parts.append(f"🎯 REFERENCIAS MANUALES: refs: {memory_id}")
        response_parts.append("🧮 POWERED BY: ClaySemanticSearchEngine v2.0")
        response_parts.append("")
        response_parts.append("[INFO] Herramienta de actualización semántica lista")
        
        print("\n".join(response_parts))
        
    except ValueError:
        print("[ERROR] memory_id debe ser un número")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Error en update_memory_guided v2.0: {str(e)}", file=sys.stderr)
        print(f"[ERROR] Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
