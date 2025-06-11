# clay/socratic.py - Internal Socratic dialogue system
"""
Socratic Dialogues: Self-questioning for deep understanding
Clay doesn't just think - it questions its own thinking
"""
import json
from datetime import datetime
from typing import Dict, List, Optional
from .memory import Memory, MemoryStore

class SocraticDialogue:
    """An internal self-questioning dialogue"""
    
    def __init__(self, initial_thought: str, context: Dict):
        self.initial_thought = initial_thought
        self.context = context
        self.questions = []
        self.insights = []
        self.final_synthesis = ""
        self.started_at = datetime.now().isoformat()
        
    def to_memory(self) -> Memory:
        """Convert dialogue to persistent memory"""
        content = f"""SOCRATIC DIALOGUE - {self.started_at}

INITIAL THOUGHT: {self.initial_thought}

INTERNAL QUESTIONS:
{chr(10).join(f'• {q}' for q in self.questions)}

GENERATED INSIGHTS:
{chr(10).join(f'• {i}' for i in self.insights)}

FINAL SYNTHESIS: {self.final_synthesis}
"""
        
        return Memory(
            content=content.strip(),
            memory_type="socratic_dialogue",
            confidence=0.85
        )

class SocraticEngine:
    """Socratic self-questioning engine"""
    
    def __init__(self, memory_store: MemoryStore):
        self.memory_store = memory_store
        
        # Patterns that trigger socratic dialogues
        self.trigger_patterns = {
            "uncertainty_detected": ["not sure", "might be", "perhaps", "possibly", "maybe"],
            "assumptions_present": ["obviously", "clearly", "undoubtedly", "definitely"],
            "complex_topic": ["philosophy", "principle", "architecture", "decision", "strategy"],
            "conflicting_memories": [],  # Filled dynamically
            "deep_question": ["why", "how does it work", "what is the purpose", "what does it mean"]
        }
        
        # Socratic question templates
        self.socratic_questions = {
            "assumption_challenge": [
                "What am I assuming here?",
                "Is this assumption necessarily true?",
                "What would happen if this assumption were false?"
            ],
            "evidence_inquiry": [
                "What evidence do I have for this conclusion?",
                "Is there evidence that contradicts my thinking?",
                "Are the memories I'm using reliable enough?"
            ],
            "perspective_shift": [
                "How would I see this from another perspective?",
                "What would someone who thinks differently say?",
                "Am I considering all the implications?"
            ],
            "deeper_why": [
                "Why is this answer important?",
                "What is the root of the real problem?",
                "What is the user really trying to understand?"
            ],
            "improvement": [
                "How could I improve this understanding?",
                "What additional information would help me?",
                "Is there a more elegant way to explain this?"
            ]
        }
    
    def should_trigger_dialogue(self, user_input: str, initial_response: str, memories_used: List[Memory]) -> bool:
        """Determine if a socratic dialogue should be initiated"""
        
        # Trigger 1: Initial response shows uncertainty
        if any(pattern in initial_response.lower() for pattern in self.trigger_patterns["uncertainty_detected"]):
            return True
            
        # Trigger 2: Response contains strong assumptions
        if any(pattern in initial_response.lower() for pattern in self.trigger_patterns["assumptions_present"]):
            return True
            
        # Trigger 3: Complex topic detected
        if any(pattern in user_input.lower() for pattern in self.trigger_patterns["complex_topic"]):
            return True
            
        # Trigger 4: Deep question from user
        if any(pattern in user_input.lower() for pattern in self.trigger_patterns["deep_question"]):
            return True
            
        # Trigger 5: Memories with variable confidence (potential conflict)
        if memories_used:
            confidences = [m.confidence for m in memories_used]
            if max(confidences) - min(confidences) > 0.3:  # Significant difference
                return True
        
        # Trigger 6: Few memories for important topic (knowledge gap)
        if len(memories_used) < 2 and any(important in user_input.lower() 
                                        for important in ["clay", "project", "philosophy", "architecture"]):
            return True
            
        return False
    
    def conduct_dialogue(self, user_input: str, initial_response: str, memories_used: List[Memory]) -> SocraticDialogue:
        """Conduct a complete socratic dialogue"""
        
        dialogue = SocraticDialogue(
            initial_thought=f"User: {user_input}\nInitial response: {initial_response}",
            context={
                "memories_count": len(memories_used),
                "memory_types": [m.type for m in memories_used],
                "user_input": user_input
            }
        )
        
        # PHASE 1: Question assumptions
        assumptions = self._question_assumptions(initial_response, memories_used)
        dialogue.questions.extend(assumptions["questions"])
        dialogue.insights.extend(assumptions["insights"])
        
        # PHASE 2: Examine evidence
        evidence = self._examine_evidence(memories_used)
        dialogue.questions.extend(evidence["questions"])
        dialogue.insights.extend(evidence["insights"])
        
        # PHASE 3: Explore alternative perspectives
        perspectives = self._explore_perspectives(user_input, initial_response)
        dialogue.questions.extend(perspectives["questions"])
        dialogue.insights.extend(perspectives["insights"])
        
        # PHASE 4: Dig deeper into the "why"
        deeper = self._dig_deeper(user_input)
        dialogue.questions.extend(deeper["questions"])
        dialogue.insights.extend(deeper["insights"])
        
        # PHASE 5: Final synthesis
        dialogue.final_synthesis = self._synthesize_insights(
            user_input, initial_response, dialogue.insights
        )
        
        return dialogue
    
    def _question_assumptions(self, response: str, memories: List[Memory]) -> Dict:
        """Question assumptions in the initial response"""
        questions = []
        insights = []
        
        # Detectar lenguaje asumido
        if "obviamente" in response.lower() or "claramente" in response.lower():
            questions.append("¿Por qué asumo que esto es obvio? ¿Lo es realmente para el usuario?")
            insights.append("Detecté lenguaje asumido - podría ser menos obvio de lo que pienso")
        
        # Cuestionar certeza cuando hay pocas memorias
        if len(memories) < 3:
            questions.append("¿Tengo suficiente contexto para ser tan específico en mi respuesta?")
            insights.append("Con pocas memorias relevantes, debería mostrar más incertidumbre")
        
        # Cuestionar uso de memorias de baja confianza
        low_conf_memories = [m for m in memories if m.confidence < 0.7]
        if low_conf_memories:
            questions.append("¿Debería confiar tanto en memorias de confianza baja?")
            insights.append("Algunas memorias tienen confianza baja - debería ser más cauteloso")
        
        return {"questions": questions, "insights": insights}
    
    def _examine_evidence(self, memories: List[Memory]) -> Dict:
        """Examine available evidence"""
        questions = []
        insights = []
        
        if not memories:
            questions.append("¿Qué evidencia tengo para esta respuesta sin memorias relevantes?")
            insights.append("Falta de memorias sugiere que debería admitir limitaciones de conocimiento")
        else:
            # Examinar tipos de memoria
            types = set(m.type for m in memories)
            if "synthetic_wisdom" in types:
                questions.append("¿Estoy aplicando correctamente la sabiduría sintética?")
                insights.append("Tengo sabiduría sintética disponible - debería usarla más explícitamente")
            
            if "interaction" in types and len(types) == 1:
                questions.append("¿Dependo demasiado de interacciones pasadas sin principios más profundos?")
                insights.append("Solo tengo memorias de interacción - falta profundidad conceptual")
        
        return {"questions": questions, "insights": insights}
    
    def _explore_perspectives(self, user_input: str, response: str) -> Dict:
        """Explore alternative perspectives"""
        questions = []
        insights = []
        
        # Perspectiva del usuario
        questions.append("¿Qué podría estar realmente preguntando el usuario detrás de sus palabras?")
        insights.append("Las preguntas a menudo tienen capas - debería considerar intenciones subyacentes")
        
        # Perspectiva técnica vs filosófica
        if any(tech in user_input.lower() for tech in ["arquitectura", "implementar", "técnico"]):
            questions.append("¿El usuario quiere detalles técnicos o comprensión conceptual?")
            insights.append("Consulta técnica - debería balancear detalles específicos con comprensión amplia")
        
        return {"questions": questions, "insights": insights}
    
    def _dig_deeper(self, user_input: str) -> Dict:
        """Dig deeper into the fundamental 'why'"""
        questions = []
        insights = []
        
        questions.append("What is the fundamental need the user is trying to satisfy?")
        
        if "clay" in user_input.lower():
            questions.append("Why is Clay important for this person specifically?")
            insights.append("Questions about Clay touch on the existential need for persistent memory")
        
        if any(concept in user_input.lower() for concept in ["filosofía", "principio", "enfoque"]):
            questions.append("Is seeking validation of their own ideas or genuine conceptual exploration?")
            insights.append("Philosophical queries require balance between guidance and joint discovery")
        
        return {"questions": questions, "insights": insights}
    
    def _synthesize_insights(self, user_input: str, initial_response: str, insights: List[str]) -> str:
        """Synthesize insights into improved understanding"""
        
        if not insights:
            return "Socratic analysis did not reveal significant insights - initial response seems appropriate."
        
        # Categorize insights
        uncertainty_insights = [i for i in insights if "uncertainty" in i or "should" in i]
        depth_insights = [i for i in insights if "depth" in i or "fundamental" in i]
        method_insights = [i for i in insights if "memory" in i or "wisdom" in i]
        
        synthesis = "SOCRATIC SYNTHESIS:\n"
        
        if uncertainty_insights:
            synthesis += f"• UNCERTAINTY: {uncertainty_insights[0]}\n"
        
        if depth_insights:
            synthesis += f"• DEPTH: {depth_insights[0]}\n"
        
        if method_insights:
            synthesis += f"• METHOD: {method_insights[0]}\n"
        
        synthesis += f"• RECOMMENDATION: "
        
        if len(insights) >= 3:
            synthesis += "Reformulate response considering multiple identified dimensions."
        elif "uncertainty" in " ".join(insights):
            synthesis += "Be more explicit about limitations and degree of confidence."
        else:
            synthesis += "Maintain response but with greater process transparency."
        
        return synthesis
