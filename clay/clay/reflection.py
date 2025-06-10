# clay/reflection.py - Sistema de reflexión offline
"""
Reflexión Offline: Clay piensa cuando no hay usuario
"""
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from .memory import Memory, MemoryStore

class ReflectionEngine:
    """Motor de reflexión para análisis offline de memorias"""
    
    def __init__(self, memory_store: MemoryStore):
        self.memory_store = memory_store
        self.reflection_history = []
        
    def reflect_on_recent_interactions(self, hours: int = 24) -> Dict:
        """
        Analizar interacciones recientes y generar insights
        """
        # 1. Obtener memorias recientes
        recent_memories = self.memory_store.get_recent(hours)
        
        if len(recent_memories) < 3:
            return {"status": "insufficient_data", "message": "Necesito más interacciones para reflexionar"}
            
        # 2. Análisis de patrones
        patterns = self._analyze_patterns(recent_memories)
        
        # 3. Detectar temas emergentes
        themes = self._extract_themes(recent_memories)
        
        # 4. Identificar gaps de conocimiento
        knowledge_gaps = self._identify_knowledge_gaps(recent_memories)
        
        # 5. Generar insights
        insights = self._generate_insights(patterns, themes, knowledge_gaps)
        
        # 6. Crear memoria de reflexión
        reflection_memory = self._create_reflection_memory(insights)
        
        # 7. Guardar reflexión
        self.memory_store.add(reflection_memory)
        
        return {
            "status": "completed",
            "insights": insights,
            "patterns_found": len(patterns),
            "themes_discovered": len(themes),
            "knowledge_gaps": len(knowledge_gaps),
            "reflection_id": reflection_memory.content[:50] + "..."
        }
    
    def _analyze_patterns(self, memories: List[Memory]) -> List[Dict]:
        """Analizar patrones en las memorias"""
        patterns = []
        
        # Patrón 1: Frecuencia de temas
        topic_frequency = {}
        for memory in memories:
            words = memory.content.lower().split()
            for word in words:
                if len(word) > 4:  # Solo palabras significativas
                    topic_frequency[word] = topic_frequency.get(word, 0) + 1
        
        # Temas recurrentes
        frequent_topics = [(topic, freq) for topic, freq in topic_frequency.items() if freq >= 2]
        if frequent_topics:
            patterns.append({
                "type": "recurring_topics",
                "data": frequent_topics[:5],  # Top 5
                "insight": "Detecté temas que aparecen repetidamente en nuestras conversaciones"
            })
        
        # Patrón 2: Tipos de memoria más comunes
        memory_types = {}
        for memory in memories:
            memory_types[memory.type] = memory_types.get(memory.type, 0) + 1
            
        patterns.append({
            "type": "memory_distribution",
            "data": memory_types,
            "insight": f"La mayoría de memorias son de tipo: {max(memory_types, key=memory_types.get)}"
        })
        
        # Patrón 3: Evolución de confianza
        confidence_trend = [m.confidence for m in memories[-5:]]  # Últimas 5
        if len(confidence_trend) >= 3:
            avg_confidence = sum(confidence_trend) / len(confidence_trend)
            patterns.append({
                "type": "confidence_trend",
                "data": confidence_trend,
                "insight": f"Confianza promedio reciente: {avg_confidence:.2f}"
            })
        
        return patterns
    
    def _extract_themes(self, memories: List[Memory]) -> List[str]:
        """Extraer temas principales de las memorias"""
        themes = []
        
        # Temas técnicos
        technical_terms = ["clay", "memoria", "asistente", "proyecto", "sistema", "base", "datos", "mcp", "servidor"]
        for term in technical_terms:
            if any(term in m.content.lower() for m in memories):
                themes.append(f"Tema técnico: {term}")
        
        # Temas de desarrollo
        dev_terms = ["implementar", "completar", "error", "funcionar", "probar", "restart", "status"]
        for term in dev_terms:
            if any(term in m.content.lower() for m in memories):
                themes.append(f"Desarrollo: {term}")
        
        return themes
    
    def _identify_knowledge_gaps(self, memories: List[Memory]) -> List[str]:
        """Identificar áreas donde falta conocimiento"""
        gaps = []
        
        # Gap 1: Si hay muchas preguntas sin respuestas completas
        question_markers = ["¿", "cómo", "qué", "por qué", "cuál"]
        questions = [m for m in memories if any(marker in m.content.lower() for marker in question_markers)]
        
        if len(questions) > len(memories) * 0.3:  # Más del 30% son preguntas
            gaps.append("Hay muchas preguntas sin respuestas completas")
        
        # Gap 2: Temas mencionados pero no desarrollados
        mentioned_topics = set()
        developed_topics = set()
        
        for memory in memories:
            words = memory.content.lower().split()
            for word in words:
                if len(word) > 5:
                    mentioned_topics.add(word)
                    if len([w for w in words if w == word]) > 1:  # Aparece más de una vez
                        developed_topics.add(word)
        
        underdeveloped = mentioned_topics - developed_topics
        if underdeveloped:
            gaps.append(f"Temas mencionados pero no desarrollados: {list(underdeveloped)[:3]}")
        
        return gaps
    
    def _generate_insights(self, patterns: List[Dict], themes: List[str], gaps: List[str]) -> Dict:
        """Generar insights de alto nivel"""
        insights = {
            "summary": f"Analicé {len(patterns)} patrones, identifiqué {len(themes)} temas principales y {len(gaps)} gaps de conocimiento.",
            "key_observations": [],
            "recommendations": []
        }
        
        # Observaciones de patrones
        for pattern in patterns:
            insights["key_observations"].append(pattern["insight"])
        
        # Observaciones de temas
        if themes:
            insights["key_observations"].append(f"Temas principales: {', '.join(themes[:3])}")
        
        # Recomendaciones basadas en gaps
        for gap in gaps:
            insights["recommendations"].append(f"Mejorar: {gap}")
        
        # Insight meta sobre la reflexión
        insights["meta_insight"] = "Esta reflexión me permite aprender de patrones en lugar de solo almacenar datos"
        
        return insights
    
    def _create_reflection_memory(self, insights: Dict) -> Memory:
        """Crear memoria de reflexión"""
        reflection_content = f"""
        REFLEXIÓN OFFLINE - {datetime.now().strftime('%Y-%m-%d %H:%M')}
        
        {insights['summary']}
        
        OBSERVACIONES CLAVE:
        {chr(10).join(f'• {obs}' for obs in insights['key_observations'])}
        
        RECOMENDACIONES:
        {chr(10).join(f'• {rec}' for rec in insights['recommendations'])}
        
        META-INSIGHT: {insights['meta_insight']}
        """
        
        return Memory(
            content=reflection_content.strip(),
            memory_type="reflection",
            confidence=0.9  # Alta confianza en auto-reflexión
        )
    
    def schedule_reflection(self, hours_interval: int = 24) -> str:
        """Programar próxima reflexión (placeholder para futuro)"""
        next_reflection = datetime.now() + timedelta(hours=hours_interval)
        return f"Próxima reflexión programada para: {next_reflection.strftime('%Y-%m-%d %H:%M')}"
