#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clay Memory Bridge - Delete Memory Guided Tool v2.0
Context-aware memory deletion with ADVANCED SEMANTIC SEARCH + impact analysis

UPGRADE: Uses ClaySemanticSearchEngine instead of primitive SQL search
- Semantic impact analysis with embeddings
- WordNet-enhanced reference detection
- Hybrid scoring for contextual relevance
- Advanced dependency discovery
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

class AdvancedGuidedMemoryDeletion:
    """Manager for context-aware memory deletion using semantic analysis"""
    
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
        Get SEMANTIC CONTEXT for deletion impact analysis:
        - same_type(3) via semantic similarity within type
        - temporal(3) via chronological + semantic relevance
        - keyword_overlap(2) via direct semantic similarity  
        - references via advanced semantic + explicit reference detection
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
            # 1. SAME TYPE - Semantic similarity within same type
            all_memories = self.store.get_all()
            same_type_memories = [m for m in all_memories 
                                if m.type == memory_type and getattr(m, 'id', 0) != memory_id]
            
            if same_type_memories and len(content.strip()) > 10:
                # Use semantic search for intelligent same-type discovery
                semantic_results = self.semantic_engine.search_semantic(content, limit=8)
                for result in semantic_results[:5]:  # Check more for type filtering
                    result_memory = result['memory']
                    if (result_memory.type == memory_type and 
                        getattr(result_memory, 'id', 0) != memory_id):
                        context['same_type'].append(result_memory)
                        if len(context['same_type']) >= 3:
                            break
            
            # Fallback for same type
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
            print(f"[WARNING] Same-type semantic analysis failed: {e}", file=sys.stderr)
            
        try:
            # 2. TEMPORAL - Recent memories with semantic relevance scoring
            recent_memories = self.store.get_recent(hours=336)  # 2 weeks window
            if recent_memories and len(content.strip()) > 10:
                # Extract key concepts for temporal relevance
                key_concepts = self._extract_semantic_keywords(content)
                temporal_query = " ".join(key_concepts[:4])  # More concepts for deletion context
                
                if temporal_query.strip():
                    semantic_results = self.semantic_engine.search_semantic(temporal_query, limit=6)
                    for result in semantic_results[:4]:
                        result_memory = result['memory']
                        if getattr(result_memory, 'id', 0) != memory_id:
                            context['temporal'].append(result_memory)
                            if len(context['temporal']) >= 3:
                                break
            
            # Chronological fallback
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
            print(f"[WARNING] Temporal semantic analysis failed: {e}", file=sys.stderr)
            
        try:
            # 3. KEYWORD OVERLAP - Direct semantic similarity (deletion impact analysis)
            if len(content.strip()) > 10:
                semantic_results = self.semantic_engine.search_semantic(content, limit=10)
                added_count = 0
                for result in semantic_results:
                    result_memory = result['memory']
                    result_id = getattr(result_memory, 'id', 0)
                    
                    # Skip if already in other contexts
                    if (result_id != memory_id and
                        result_id not in [getattr(m, 'id', 0) for m in context['same_type']] and
                        result_id not in [getattr(m, 'id', 0) for m in context['temporal']]):
                        
                        context['keyword_overlap'].append(result_memory)
                        added_count += 1
                        if added_count >= 2:
                            break
                            
        except Exception as e:
            print(f"[WARNING] Semantic overlap analysis failed: {e}", file=sys.stderr)
            
        try:
            # 4. REFERENCES - Advanced semantic + explicit reference detection
            # Critical for deletion: find who depends on this memory
            
            # a) Explicit references
            ref_patterns = [f"refs: {memory_id}", f"ID: {memory_id}", f"memoria {memory_id}"]
            
            for pattern in ref_patterns[:2]:  # Check most common patterns
                try:
                    semantic_results = self.semantic_engine.search_semantic(pattern, limit=4)
                    for result in semantic_results:
                        result_memory = result['memory']
                        if getattr(result_memory, 'id', 0) != memory_id:
                            context['references'].append(result_memory)
                            if len(context['references']) >= 3:
                                break
                    if len(context['references']) >= 3:
                        break
                except Exception:
                    continue
            
            # b) Semantic content references - find memories that semantically reference this content
            if len(context['references']) < 3 and len(content.strip()) > 20:
                # Use abbreviated content as search to find related discussions
                content_summary = " ".join(content.split()[:10])  # First 10 words
                try:
                    semantic_results = self.semantic_engine.search_semantic(content_summary, limit=6)
                    for result in semantic_results:
                        result_memory = result['memory']
                        result_id = getattr(result_memory, 'id', 0)
                        
                        # Check if not already found and has high semantic similarity
                        if (result_id != memory_id and
                            result_id not in [getattr(m, 'id', 0) for m in context['references']] and
                            result.get('semantic_similarity', 0) > 0.4):  # High threshold for references
                            
                            context['references'].append(result_memory)
                            if len(context['references']) >= 3:
                                break
                except Exception:
                    pass
            
            # c) SQL fallback for explicit references
            if len(context['references']) < 2:
                ref_conditions = " OR ".join(["content LIKE ?" for _ in ref_patterns])
                ref_params = [f"%{pattern}%" for pattern in ref_patterns] + [memory_id]
                
                cursor = self.store.conn.execute(
                    f"""SELECT * FROM memories 
                       WHERE ({ref_conditions}) AND id != ?
                       ORDER BY created_at DESC 
                       LIMIT 3""",
                    ref_params
                )
                for row in cursor:
                    ref_memory = self.store._row_to_memory(row)
                    ref_memory.id = row['id']
                    context['references'].append(ref_memory)
                    
        except Exception as e:
            print(f"[WARNING] Reference analysis failed: {e}", file=sys.stderr)
            
        return context
    
    def _extract_semantic_keywords(self, content: str) -> List[str]:
        """Extract meaningful keywords for semantic deletion analysis"""
        exclude = {
            'que', 'para', 'con', 'una', 'por', 'del', 'las', 'los', 'esto', 'esta', 'ese', 'esa',
            'más', 'como', 'muy', 'todo', 'bien', 'puede', 'hacer', 'ser', 'estar', 'tener',
            'desde', 'hasta', 'entre', 'sobre', 'bajo', 'ante', 'tras', 'durante', 'mediante',
            'memoria', 'recordar', 'guardar', 'almacenar'  # Memory-specific terms
        }
        
        words = content.lower().split()
        keywords = []
        
        for word in words:
            clean_word = word.strip('.,!?;:()[]"\'')
            if (len(clean_word) > 2 and 
                clean_word not in exclude and
                clean_word.replace('_', '').replace('-', '').isalpha()):
                keywords.append(clean_word)
        
        return keywords[:8]  # More keywords for deletion impact analysis
    
    def analyze_semantic_deletion_impact(self, target_memory: Memory, context: Dict) -> Dict[str, any]:
        """Analyze semantic impact of deleting this memory with advanced scoring"""
        impact = {
            'severity': 'LOW',
            'semantic_references_count': len(context['references']),
            'semantic_similarity_count': len(context['keyword_overlap']),
            'type_isolation': len(context['same_type']) == 0,
            'temporal_isolation': len(context['temporal']) < 2,
            'content_uniqueness': 'unknown',
            'semantic_connectivity': 0.0,
            'warnings': [],
            'recommendations': []
        }
        
        warnings = []
        
        # Enhanced severity analysis using semantic insights
        if impact['semantic_references_count'] > 0:
            impact['severity'] = 'HIGH'
            warnings.append(f"Esta memoria tiene {impact['semantic_references_count']} referencias semánticas")
        
        if target_memory.type == 'golden':
            impact['severity'] = 'CRITICAL'
            warnings.append("Las GOLDEN MEMORIES son fundamentales del sistema")
        elif target_memory.type.startswith('synthetic'):
            impact['severity'] = 'HIGH'
            warnings.append("Las memorias sintéticas son componentes arquitecturales")
        elif target_memory.type == 'milestone':
            impact['severity'] = 'MEDIUM'
            warnings.append("Los hitos documentan progreso histórico importante")
        
        # Semantic uniqueness analysis
        if impact['semantic_similarity_count'] == 0:
            if impact['severity'] == 'LOW':
                impact['severity'] = 'MEDIUM'
            warnings.append("No hay memorias semánticamente similares - contenido único")
        
        if impact['type_isolation']:
            warnings.append("Es la única memoria de este tipo - posible pérdida de categoría")
        
        # Calculate semantic connectivity score
        total_connections = sum(len(memories) for memories in context.values())
        if total_connections > 0:
            impact['semantic_connectivity'] = min(total_connections / 8.0, 1.0)  # Normalize to 0-1
        
        if impact['semantic_connectivity'] > 0.75:
            warnings.append("Alta conectividad semántica - eliminar puede fragmentar conocimiento")
        
        # Generate enhanced recommendations
        recommendations = []
        if impact['severity'] == 'CRITICAL':
            recommendations.append("⛔ CRITICAL: NO eliminar bajo ninguna circunstancia")
        elif impact['severity'] == 'HIGH':
            recommendations.append("⚠️ HIGH: Solo eliminar con análisis exhaustivo y backup")
        elif impact['severity'] == 'MEDIUM':
            recommendations.append("⚡ MEDIUM: Revisar impacto semántico cuidadosamente")
        else:
            recommendations.append("✅ LOW: Eliminación relativamente segura")
        
        if impact['semantic_references_count'] > 0:
            recommendations.append("🔗 Actualizar referencias semánticas antes de eliminar")
        
        if target_memory.type == 'golden':
            recommendations.append("🏆 Golden memories requieren consensus explícito para eliminación")
        
        if impact['semantic_similarity_count'] == 0:
            recommendations.append("💾 Considerar backup - contenido conceptualmente único")
        
        if impact['semantic_connectivity'] > 0.5:
            recommendations.append("🕸️ Verificar integridad de red semántica post-eliminación")
        
        impact['warnings'] = warnings
        impact['recommendations'] = recommendations
        
        return impact
    
    def generate_semantic_socratic_confirmation(self, target_memory: Memory, context: Dict, impact: Dict) -> List[str]:
        """Generate semantic-aware Socratic questions for deletion confirmation"""
        questions = []
        memory_id = getattr(target_memory, 'id', '?')
        
        # Universal semantic confirmation questions
        questions.extend([
            f"¿Es realmente necesario eliminar la memoria {memory_id} o existe alternativa semántica?",
            "¿Qué conocimiento conceptual se perdería permanentemente?",
            "¿Hay dependencias semánticas ocultas que no son obvias?"
        ])
        
        # Type-specific semantic questions
        if target_memory.type == "golden":
            questions.extend([
                "🏆 ¿Esta golden memory ha sido empíricamente invalidada?",
                "¿Existe consensus explícito para eliminar esta regla fundamental?",
                "¿Se ha documentado la razón de invalidación de esta sabiduría golden?"
            ])
        elif target_memory.type == "interaction":
            questions.extend([
                "¿Esta interacción contiene insights únicos no capturados en otras memorias?",
                "¿La eliminación afectará la continuidad narrativa del diálogo?"
            ])
        elif target_memory.type == "reflection":
            questions.extend([
                "¿Los insights de esta reflexión son únicos o redundantes semánticamente?",
                "¿Otros análisis cubren conceptualmente el mismo territorio?"
            ])
        elif target_memory.type == "milestone":
            questions.extend([
                "¿Este hito es parte integral de la narrativa histórica del proyecto?",
                "¿Su eliminación creará gaps en la comprensión del progreso?"
            ])
        elif target_memory.type.startswith("synthetic"):
            questions.extend([
                "¿Esta sabiduría sintética ha sido completamente superada conceptualmente?",
                "¿Existe una versión mejorada que reemplace completamente este conocimiento?"
            ])
        
        # Semantic impact-based questions
        if impact['severity'] == 'CRITICAL':
            questions.extend([
                "🚨 CRITICAL SEVERITY: ¿Has agotado todas las alternativas a la eliminación?",
                "¿Los riesgos semánticos justifican la eliminación irreversible?"
            ])
        elif impact['severity'] == 'HIGH':
            questions.extend([
                "⚠️ HIGH SEVERITY: ¿Se ha analizado completamente el impacto semántico?",
                "¿Existe estrategia de mitigación para dependencias semánticas?"
            ])
        
        if impact['semantic_references_count'] > 0:
            questions.append(f"¿Cómo afectará a las {impact['semantic_references_count']} memorias semánticamente dependientes?")
        
        if impact['semantic_similarity_count'] == 0:
            questions.append("⚠️ Contenido único: ¿Estás seguro de que este conocimiento no será necesario?")
        
        if impact['semantic_connectivity'] > 0.7:
            questions.append("🕸️ Alta conectividad: ¿Cómo afectará a la integridad de la red semántica?")
        
        if target_memory.type == 'golden':
            questions.extend([
                "🏆 ¿Se ha documentado empíricamente por qué esta golden memory ya no aplica?",
                "¿Existe nueva golden memory que reemplace explícitamente esta sabiduría?"
            ])
        
        return questions
    
    def delete_memory(self, memory_id: int) -> bool:
        """DELETE memory from database and invalidate semantic index - EXTREME CAUTION"""
        try:
            cursor = self.store.conn.execute(
                "DELETE FROM memories WHERE id = ?", (memory_id,)
            )
            self.store.conn.commit()
            
            # Invalidate semantic index
            if hasattr(self.semantic_engine, 'indexed_memory_ids'):
                self.semantic_engine.indexed_memory_ids.discard(memory_id)
            
            return cursor.rowcount > 0
        except Exception as e:
            print(f"[ERROR] Database deletion failed: {e}", file=sys.stderr)
            return False

def main():
    if len(sys.argv) < 2:
        print("[ERROR] Falta memory_id")
        print("[USAGE] clay_delete_memory_guided.py <memory_id> [--confirm]")
        sys.exit(1)
    
    try:
        memory_id = int(sys.argv[1])
        confirm_delete = len(sys.argv) > 2 and sys.argv[2] == "--confirm"
        
        # Initialize assistant with advanced semantic analysis
        assistant = ContextualAssistant("claude_mcp_enhanced")
        manager = AdvancedGuidedMemoryDeletion(assistant)
        
        # Get target memory
        target_memory = manager.get_memory_by_id(memory_id)
        if not target_memory:
            print(f"[ERROR] Memoria {memory_id} no encontrada")
            sys.exit(1)
        
        # Get SEMANTIC contextual memories for impact analysis
        context = manager.get_semantic_contextual_memories(target_memory)
        context_count = sum(len(memories) for memories in context.values())
        
        # Analyze semantic deletion impact
        impact = manager.analyze_semantic_deletion_impact(target_memory, context)
        
        # Generate semantic Socratic confirmation questions
        questions = manager.generate_semantic_socratic_confirmation(target_memory, context, impact)
        
        # Format guided deletion interface with semantic analysis
        response_parts = []
        response_parts.append("🗑️ ELIMINACIÓN GUIADA DE MEMORIA v2.0 (SEMANTIC)")
        response_parts.append("=" * 65)
        
        # Enhanced severity indicator
        severity_emojis = {
            'LOW': '🟩',
            'MEDIUM': '🟨', 
            'HIGH': '🟥',
            'CRITICAL': '🚨'
        }
        severity_emoji = severity_emojis.get(impact['severity'], '❓')
        response_parts.append(f"{severity_emoji} ANÁLISIS DE IMPACTO SEMÁNTICO: {impact['severity']} SEVERITY")
        response_parts.append("")
        
        response_parts.append(f"📝 MEMORIA OBJETIVO (ID: {memory_id})")
        response_parts.append(f"Tipo: {target_memory.type}")
        response_parts.append(f"Creada: {target_memory.created_at}")
        response_parts.append(f"Conectividad semántica: {impact['semantic_connectivity']:.2f}")
        response_parts.append(f"Contenido:")
        content_preview = target_memory.content[:350] + "..." if len(target_memory.content) > 350 else target_memory.content
        response_parts.append(f'"{content_preview}"')
        response_parts.append("")
        
        response_parts.append(f"🧠 CONTEXTO SEMÁNTICO ({context_count} memorias relacionadas)")
        response_parts.append("-" * 45)
        
        context_descriptions = {
            'same_type': 'Mismo Tipo (similitud semántica)',
            'temporal': 'Temporales (relevancia semántica)', 
            'keyword_overlap': 'Similitud Conceptual Directa',
            'references': 'Referencias (explícitas + semánticas)'
        }
        
        for context_type, memories in context.items():
            if memories:
                desc = context_descriptions.get(context_type, context_type)
                response_parts.append(f"🎯 {desc}: {len(memories)} memorias")
                for i, mem in enumerate(memories[:2]):
                    preview = mem.content[:80] + "..." if len(mem.content) > 80 else mem.content
                    mem_id = getattr(mem, 'id', '?')
                    response_parts.append(f"  {i+1}. (ID: {mem_id}) {preview}")
        response_parts.append("")
        
        response_parts.append("⚠️ ADVERTENCIAS SEMÁNTICAS Y RECOMENDACIONES")
        response_parts.append("-" * 45)
        for warning in impact['warnings']:
            response_parts.append(f"⚠️ {warning}")
        response_parts.append("")
        for recommendation in impact['recommendations']:
            response_parts.append(f"💡 {recommendation}")
        response_parts.append("")
        
        response_parts.append("❓ CONFIRMACIÓN SOCRÁTICA SEMÁNTICA")
        response_parts.append("-" * 45)
        for i, question in enumerate(questions, 1):
            response_parts.append(f"{i}. {question}")
        response_parts.append("")
        
        response_parts.append("🚨 PRÓXIMOS PASOS CRÍTICOS (SEMANTIC-AWARE):")
        response_parts.append("1. ⚠️ REVISAR todas las advertencias semánticas arriba")
        response_parts.append("2. 🤔 REFLEXIONAR sobre cada pregunta socrática contextual")
        response_parts.append("3. 🔍 ANALIZAR impacto en red semántica de memorias")
        response_parts.append("4. 🧠 VERIFICAR dependencias conceptuales ocultas")
        response_parts.append("5. ⚖️ EVALUAR si actualización semántica es mejor alternativa")
        response_parts.append("6. 🗑️ Solo entonces considerar eliminación IRREVERSIBLE")
        response_parts.append("")
        
        response_parts.append(f"{severity_emoji} SEVERIDAD DE ELIMINACIÓN: {impact['severity']}")
        response_parts.append(f"🕸️ CONECTIVIDAD SEMÁNTICA: {impact['semantic_connectivity']:.2f}")
        response_parts.append("🧮 POWERED BY: ClaySemanticSearchEngine v2.0")
        response_parts.append("")
        response_parts.append("[WARNING] La eliminación es IRREVERSIBLE e invalida índice semántico")
        
        print("\n".join(response_parts))
        
        # CONFIRMACIÓN FINAL
        if confirm_delete:
            print("\n" + "="*50)
            print("🗑️ EJECUTANDO ELIMINACIÓN CONFIRMADA...")
            print("="*50)
            
            if manager.delete_memory(memory_id):
                print(f"✅ [SUCCESS] Memoria {memory_id} eliminada exitosamente")
                print("🔄 [INFO] Índice semántico invalidado - se regenerará automáticamente")
                print(f"📊 [INFO] Severidad de eliminación: {impact['severity']}")
                print(f"🕸️ [INFO] Conectividad semántica previa: {impact['semantic_connectivity']:.2f}")
            else:
                print(f"❌ [ERROR] No se pudo eliminar la memoria {memory_id}")
                sys.exit(1)
        else:
            print("\n" + "="*50)
            print("ℹ️ ANÁLISIS COMPLETADO - NO SE EJECUTÓ ELIMINACIÓN")
            print("="*50)
            print("Para confirmar eliminación usar: --confirm")
            print(f"Comando: python clay_delete_memory_guided.py {memory_id} --confirm")
            print("⚠️ [WARNING] Solo confirmar tras revisar TODAS las advertencias socráticas arriba")
        
    except ValueError:
        print("[ERROR] memory_id debe ser un número")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Error en delete_memory_guided v2.0: {str(e)}", file=sys.stderr)
        print(f"[ERROR] Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
