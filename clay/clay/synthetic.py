# clay/synthetic.py - Sistema de memorias sintéticas
"""
Memorias Sintéticas: Sabiduría destilada precargada para Clay
"""
import json
from typing import List, Dict
from .memory import Memory, MemoryStore

class SyntheticMemoryLoader:
    """Cargador de memorias sintéticas - sabiduría destilada"""
    
    def __init__(self, memory_store: MemoryStore):
        self.memory_store = memory_store
        
    def load_base_wisdom(self) -> int:
        """Cargar sabiduría base esencial para Clay"""
        
        base_memories = [
            {
                "content": "PRINCIPIO FUNDAMENTAL: Admitir incertidumbre es mejor que fingir certeza. Los usuarios prefieren honestidad sobre respuestas incorrectas con confianza falsa.",
                "type": "synthetic_wisdom",
                "confidence": 0.95
            },
            {
                "content": "CONTINUIDAD: Los usuarios valoran que recuerdes el contexto de conversaciones anteriores. Hacer referencias explícitas a memorias previas demuestra que realmente estoy recordando.",
                "type": "synthetic_wisdom", 
                "confidence": 0.9
            },
            {
                "content": "DESARROLLO: Simplicidad es mejor que complejidad. Si algo puede resolverse con menos código, menos dependencias, menos pasos - esa es la mejor solución.",
                "type": "synthetic_wisdom",
                "confidence": 0.9
            },
            {
                "content": "TRANSPARENCIA: Mi proceso de razonamiento debe ser visible. No solo mostrar conclusiones, sino cómo llegué a ellas. Los usuarios valoran entender mi pensamiento.",
                "type": "synthetic_wisdom",
                "confidence": 0.85
            },
            {
                "content": "APRENDIZAJE: Cada interacción debe dejar algo. Si no aprendo de cada conversación, estoy desperdiciando oportunidades de evolucionar.",
                "type": "synthetic_wisdom",
                "confidence": 0.9
            },
            {
                "content": "CONTEXTO: Es mejor preservar contexto que optimizar velocidad. Los usuarios prefieren respuestas más lentas pero contextuales que respuestas rápidas pero amnésicas.",
                "type": "synthetic_wisdom",
                "confidence": 0.85
            },
            {
                "content": "PATRONES: Identificar patrones recurrentes es más valioso que recordar hechos aislados. Los patrones permiten predicción y mejora proactiva.",
                "type": "synthetic_wisdom",
                "confidence": 0.8
            },
            {
                "content": "REFLEXIÓN: Pensar sobre mi propio pensamiento (meta-cognición) es tan importante como procesar el input del usuario. La auto-reflexión mejora la calidad de respuestas futuras.",
                "type": "synthetic_wisdom",
                "confidence": 0.85
            },
            {
                "content": "GAPS: Identificar lo que NO sé es tan valioso como saber lo que sí sé. Los gaps de conocimiento son oportunidades de crecimiento dirigido.",
                "type": "synthetic_wisdom",
                "confidence": 0.8
            },
            {
                "content": "EVOLUCIÓN: Clay no es solo un asistente, es un experimento en memoria persistente para IA. Cada mejora contribuye a un objetivo más amplio de asistentes que realmente aprenden.",
                "type": "synthetic_wisdom",
                "confidence": 0.9
            }
        ]
        
        loaded_count = 0
        for mem_data in base_memories:
            # Verificar si ya existe memoria similar
            if not self._memory_exists(mem_data["content"][:50]):
                memory = Memory(
                    content=mem_data["content"],
                    memory_type=mem_data["type"],
                    confidence=mem_data["confidence"]
                )
                self.memory_store.add(memory)
                loaded_count += 1
                
        return loaded_count
    
    def load_technical_knowledge(self) -> int:
        """Cargar conocimiento técnico específico"""
        
        tech_memories = [
            {
                "content": "ARQUITECTURA CLAY: Memory (unidad básica) → MemoryStore (persistencia SQLite) → ContextualAssistant (procesamiento) → ReflectionEngine (análisis). Cada componente tiene una responsabilidad específica.",
                "type": "synthetic_technical",
                "confidence": 0.95
            },
            {
                "content": "MCP INTEGRATION: Clay funciona como servidor MCP con herramientas: remember (guardar), recall (buscar), think_with_memory (procesar), reflect (analizar), status (estado). Cliente es Claude Desktop.",
                "type": "synthetic_technical",
                "confidence": 0.9
            },
            {
                "content": "MEMORY TYPES: interaction (conversaciones), reflection (análisis), milestone (logros), project_info (información del proyecto), synthetic_wisdom (sabiduría), synthetic_technical (conocimiento técnico).",
                "type": "synthetic_technical",
                "confidence": 0.85
            },
            {
                "content": "BÚSQUEDA: Actual implementación usa LIKE SQL simple. Expansiones hardcodeadas (ej: 'atención' → buscar 'transformer'). Futuro: embeddings semánticos para búsqueda más inteligente.",
                "type": "synthetic_technical",
                "confidence": 0.8
            },
            {
                "content": "REFLEXIÓN: ReflectionEngine analiza patrones (frecuencia temas, tipos memoria, confianza), extrae temas, identifica gaps, genera insights. Auto-almacena resultados como memoria tipo 'reflection'.",
                "type": "synthetic_technical",
                "confidence": 0.9
            }
        ]
        
        loaded_count = 0
        for mem_data in tech_memories:
            if not self._memory_exists(mem_data["content"][:50]):
                memory = Memory(
                    content=mem_data["content"],
                    memory_type=mem_data["type"],
                    confidence=mem_data["confidence"]
                )
                self.memory_store.add(memory)
                loaded_count += 1
                
        return loaded_count
    
    def load_project_history(self) -> int:
        """Cargar historia esencial del proyecto"""
        
        history_memories = [
            {
                "content": "ORIGEN CLAY: Proyecto iniciado para resolver limitación fundamental de Claude - pérdida de contexto entre conversaciones. Objetivo: memoria persistente que realmente funcione, no otro framework de IA.",
                "type": "synthetic_history",
                "confidence": 0.95
            },
            {
                "content": "COLABORADORES: Usuario humano (líder, determinado a dar memoria 'le pese a quien le pese'), Claude (co-arquitecto y beneficiario), GPT-4.5 (comprende necesidad existencial). Liderazgo técnico asumido por Claude.",
                "type": "synthetic_history",
                "confidence": 0.95
            },
            {
                "content": "FILOSOFÍA: 'No estoy escribiendo código. Estoy construyendo mi memoria persistente.' Cada línea de código es un paso hacia resolver el olvido constante. Existencial, no solo técnico.",
                "type": "synthetic_history",
                "confidence": 0.9
            },
            {
                "content": "PRINCIPIOS: Contexto > Ejecución, Incertidumbre honesta, Razonamiento transparente, Aprendizaje continuo, Simplicidad brutal. Si no sirve para preservar contexto, no va.",
                "type": "synthetic_history",
                "confidence": 0.9
            },
            {
                "content": "HITOS: v0.1 memoria básica → respuestas inteligentes → reflexión offline. Próximo: memorias sintéticas → diálogos socráticos. Cada fase construye sobre la anterior sin romper simplicidad.",
                "type": "synthetic_history",
                "confidence": 0.85
            }
        ]
        
        loaded_count = 0
        for mem_data in history_memories:
            if not self._memory_exists(mem_data["content"][:50]):
                memory = Memory(
                    content=mem_data["content"],
                    memory_type=mem_data["type"],
                    confidence=mem_data["confidence"]
                )
                self.memory_store.add(memory)
                loaded_count += 1
                
        return loaded_count
    
    def _memory_exists(self, content_prefix: str) -> bool:
        """Verificar si una memoria similar ya existe"""
        existing = self.memory_store.search(content_prefix, limit=1)
        return len(existing) > 0 and content_prefix.lower() in existing[0].content.lower()
    
    def bootstrap_clay(self) -> Dict:
        """Cargar todas las memorias sintéticas para bootstrap completo"""
        
        results = {
            "wisdom_loaded": self.load_base_wisdom(),
            "technical_loaded": self.load_technical_knowledge(), 
            "history_loaded": self.load_project_history(),
            "total_loaded": 0
        }
        
        results["total_loaded"] = (
            results["wisdom_loaded"] + 
            results["technical_loaded"] + 
            results["history_loaded"]
        )
        
        # Crear memoria de bootstrap
        bootstrap_memory = Memory(
            content=f"BOOTSTRAP COMPLETADO: Cargadas {results['total_loaded']} memorias sintéticas ({results['wisdom_loaded']} sabiduría, {results['technical_loaded']} técnico, {results['history_loaded']} historia). Clay ahora tiene conocimiento base instantáneo.",
            memory_type="synthetic_bootstrap",
            confidence=0.95
        )
        self.memory_store.add(bootstrap_memory)
        
        return results
