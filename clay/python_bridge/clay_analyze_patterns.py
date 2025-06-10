#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clay Memory Bridge - Memory Pattern Analysis Tool (Updated v2.0)
Analyze patterns in memory usage and content evolution with CXD v2.0
"""

import sys
import os
import json
from collections import defaultdict, Counter
from datetime import datetime, timedelta


# FORCE UTF-8 I/O - CRITICAL para Windows
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    
# Add Clay to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Add new CXD classifier path
cxd_path = r"D:\claude\cxd-classifier\src"
sys.path.insert(0, cxd_path)

def get_memory_store():
    """Get the memory store instance"""
    try:
        from clay.memory import MemoryStore
        
        # Use Claude's default memory database
        db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "claude_mcp_memories.db")
        return MemoryStore(db_path)
    except Exception as e:
        print(f"[ERROR] Error accessing memory store: {e}", file=sys.stderr)
        return None

def init_cxd_classifier():
    """Initialize CXD classifier v2.0 for pattern analysis"""
    try:
        from cxd_classifier.classifiers.meta import MetaCXDClassifier
        from cxd_classifier.core.config import CXDConfig
        
        config = CXDConfig()
        classifier = MetaCXDClassifier(config=config)
        return classifier
    except Exception as e:
        print(f"[WARNING] Error inicializando CXD v2.0: {e}", file=sys.stderr)
        return None

def classify_memory_batch(classifier, memories):
    """Classify multiple memories for pattern analysis"""
    if not classifier:
        return {}
    
    classifications = {}
    for memory in memories:
        try:
            result = classifier.classify_detailed(text=memory.content)
            classifications[memory.id] = {
                'function': result.final_sequence.dominant_function.value,
                'confidence': result.confidence_scores.get('final', 0.0),
                'pattern': result.final_sequence.pattern,
                'processing_time': result.processing_time_ms
            }
        except Exception as e:
            classifications[memory.id] = {
                'function': 'UNKNOWN',
                'confidence': 0.0,
                'pattern': 'UNKNOWN',
                'processing_time': 0.0
            }
    
    return classifications

def analyze_temporal_patterns(memories, classifications):
    """Analyze how memory patterns change over time"""
    # Group memories by time periods
    time_buckets = defaultdict(list)
    
    for memory in memories:
        try:
            # Parse creation time
            created = datetime.fromisoformat(memory.created_at.replace('Z', '+00:00'))
            
            # Group by hour for recent memories (last 24h)
            now = datetime.now().replace(tzinfo=created.tzinfo)
            hours_ago = (now - created).total_seconds() / 3600
            
            if hours_ago <= 24:
                bucket = f"last_{int(hours_ago)}h"
            elif hours_ago <= 24 * 7:
                bucket = f"last_{int(hours_ago / 24)}d"
            else:
                bucket = "older"
            
            memory_data = {
                'memory': memory,
                'classification': classifications.get(memory.id, {'function': 'UNKNOWN'})
            }
            time_buckets[bucket].append(memory_data)
            
        except Exception as e:
            continue
    
    return time_buckets

def analyze_content_patterns(memories, classifications):
    """Analyze content patterns and themes"""
    patterns = {
        'functions': Counter(),
        'types': Counter(),
        'confidence_distribution': [],
        'content_themes': defaultdict(list),
        'complexity_scores': []
    }
    
    for memory in memories:
        # Memory type distribution
        patterns['types'][memory.memory_type] += 1
        
        # CXD function distribution
        classification = classifications.get(memory.id, {'function': 'UNKNOWN'})
        patterns['functions'][classification['function']] += 1
        
        # Confidence distribution
        patterns['confidence_distribution'].append(classification.get('confidence', 0.0))
        
        # Content analysis
        content_length = len(memory.content)
        patterns['complexity_scores'].append(content_length)
        
        # Theme extraction (simple keyword analysis)
        content_lower = memory.content.lower()
        themes = []
        
        # Technical themes
        if any(word in content_lower for word in ['codigo', 'function', 'class', 'import']):
            themes.append('programming')
        if any(word in content_lower for word in ['memory', 'memoria', 'remember', 'recordar']):
            themes.append('memory_system')
        if any(word in content_lower for word in ['clay', 'cxd', 'clasificador']):
            themes.append('clay_development')
        if any(word in content_lower for word in ['error', 'bug', 'problema', 'arreglar']):
            themes.append('troubleshooting')
        if any(word in content_lower for word in ['test', 'testing', 'prueba', 'probar']):
            themes.append('testing')
        
        if not themes:
            themes.append('general')
        
        for theme in themes:
            patterns['content_themes'][theme].append(memory.id)
    
    return patterns

def analyze_usage_patterns(memories):
    """Analyze memory access and usage patterns"""
    patterns = {
        'access_frequency': defaultdict(int),
        'memory_age_distribution': [],
        'type_evolution': defaultdict(list)
    }
    
    now = datetime.now()
    
    for memory in memories:
        try:
            created = datetime.fromisoformat(memory.created_at.replace('Z', '+00:00'))
            age_hours = (now.replace(tzinfo=created.tzinfo) - created).total_seconds() / 3600
            
            patterns['memory_age_distribution'].append(age_hours)
            patterns['access_frequency'][memory.access_count] += 1
            patterns['type_evolution'][memory.memory_type].append(age_hours)
            
        except Exception as e:
            continue
    
    return patterns

def generate_insights(memories, classifications, temporal_patterns, content_patterns, usage_patterns):
    """Generate insights from pattern analysis"""
    insights = []
    
    total_memories = len(memories)
    
    # Function distribution insights
    func_dist = content_patterns['functions']
    if func_dist:
        dominant_function = func_dist.most_common(1)[0]
        insights.append(f"Funcion cognitiva dominante: {dominant_function[0]} ({dominant_function[1]}/{total_memories} memorias)")
        
        if len(func_dist) > 1:
            second_function = func_dist.most_common(2)[1]
            insights.append(f"Segunda funcion mas comun: {second_function[0]} ({second_function[1]} memorias)")
    
    # Temporal insights
    recent_memories = sum(len(bucket) for bucket_name, bucket in temporal_patterns.items() 
                         if 'h' in bucket_name)
    if recent_memories > 0:
        recent_percentage = (recent_memories / total_memories) * 100
        insights.append(f"Actividad reciente: {recent_memories} memorias en las ultimas 24h ({recent_percentage:.1f}%)")
    
    # Content complexity insights
    if content_patterns['complexity_scores']:
        avg_length = sum(content_patterns['complexity_scores']) / len(content_patterns['complexity_scores'])
        insights.append(f"Longitud promedio de memoria: {avg_length:.0f} caracteres")
    
    # Confidence insights
    if content_patterns['confidence_distribution']:
        avg_confidence = sum(content_patterns['confidence_distribution']) / len(content_patterns['confidence_distribution'])
        insights.append(f"Confianza promedio en clasificacion: {avg_confidence:.2f}")
    
    # Theme insights
    themes = content_patterns['content_themes']
    if themes:
        top_theme = max(themes.items(), key=lambda x: len(x[1]))
        insights.append(f"Tema dominante: {top_theme[0]} ({len(top_theme[1])} memorias)")
    
    # Usage patterns
    if usage_patterns['access_frequency']:
        total_accesses = sum(count * freq for count, freq in usage_patterns['access_frequency'].items())
        avg_access = total_accesses / total_memories if total_memories > 0 else 0
        insights.append(f"Promedio de accesos por memoria: {avg_access:.1f}")
    
    return insights

def format_pattern_analysis(memories, classifications, temporal_patterns, content_patterns, usage_patterns, insights):
    """Format the complete pattern analysis"""
    response_parts = []
    
    # Header
    response_parts.append("[PATTERNS] ANALISIS DE PATRONES DE MEMORIA")
    response_parts.append("=" * 60)
    response_parts.append(f"[DATASET] Total memorias analizadas: {len(memories)}")
    response_parts.append("")
    
    # Key insights
    response_parts.append("[INSIGHTS] INSIGHTS PRINCIPALES:")
    for insight in insights:
        response_parts.append(f"  - {insight}")
    response_parts.append("")
    
    # Function distribution
    response_parts.append("[CXD] DISTRIBUCION POR FUNCION COGNITIVA:")
    func_dist = content_patterns['functions']
    total = sum(func_dist.values())
    
    for func, count in func_dist.most_common():
        percentage = (count / total) * 100 if total > 0 else 0
        bar = "#" * int(percentage / 5) + "-" * (20 - int(percentage / 5))
        func_label = {"C": "CONTROL", "X": "CONTEXT", "D": "DATA"}.get(func, func)
        response_parts.append(f"  {func_label:<10} [{bar}] {count:3d} ({percentage:5.1f}%)")
    response_parts.append("")
    
    # Memory type distribution
    response_parts.append("[TYPES] DISTRIBUCION POR TIPO:")
    type_dist = content_patterns['types']
    for memory_type, count in type_dist.most_common():
        percentage = (count / len(memories)) * 100 if len(memories) > 0 else 0
        response_parts.append(f"  {memory_type:<15} {count:3d} ({percentage:5.1f}%)")
    response_parts.append("")
    
    # Temporal patterns
    response_parts.append("[TEMPORAL] PATRONES TEMPORALES:")
    for time_bucket in sorted(temporal_patterns.keys()):
        bucket_memories = temporal_patterns[time_bucket]
        if bucket_memories:
            bucket_functions = Counter(item['classification']['function'] for item in bucket_memories)
            response_parts.append(f"  {time_bucket:<10} {len(bucket_memories):3d} memorias - {dict(bucket_functions)}")
    response_parts.append("")
    
    # Content themes
    response_parts.append("[THEMES] TEMAS DE CONTENIDO:")
    themes = content_patterns['content_themes']
    for theme, memory_ids in sorted(themes.items(), key=lambda x: len(x[1]), reverse=True):
        count = len(memory_ids)
        percentage = (count / len(memories)) * 100 if len(memories) > 0 else 0
        response_parts.append(f"  {theme:<20} {count:3d} memorias ({percentage:5.1f}%)")
    response_parts.append("")
    
    # Quality metrics
    response_parts.append("[QUALITY] METRICAS DE CALIDAD:")
    if content_patterns['confidence_distribution']:
        confidences = content_patterns['confidence_distribution']
        avg_conf = sum(confidences) / len(confidences)
        high_conf = sum(1 for c in confidences if c > 0.7) / len(confidences) * 100
        response_parts.append(f"  Confianza promedio:     {avg_conf:.3f}")
        response_parts.append(f"  Memorias alta confianza: {high_conf:.1f}%")
    
    if content_patterns['complexity_scores']:
        lengths = content_patterns['complexity_scores']
        avg_length = sum(lengths) / len(lengths)
        response_parts.append(f"  Longitud promedio:      {avg_length:.0f} caracteres")
    response_parts.append("")
    
    # Recommendations
    response_parts.append("[RECOMMENDATIONS] RECOMENDACIONES:")
    
    # Based on function distribution
    if func_dist:
        dominant = func_dist.most_common(1)[0]
        if dominant[1] / total > 0.6:
            response_parts.append(f"  - Alto enfoque en {dominant[0]} - considera diversificar operaciones cognitivas")
    
    # Based on temporal patterns
    recent_activity = sum(len(bucket) for name, bucket in temporal_patterns.items() if 'h' in name)
    if recent_activity < len(memories) * 0.3:
        response_parts.append("  - Baja actividad reciente - considera usar memoria mas frecuentemente")
    
    # Based on confidence
    if content_patterns['confidence_distribution']:
        low_conf_count = sum(1 for c in content_patterns['confidence_distribution'] if c < 0.5)
        if low_conf_count > len(memories) * 0.3:
            response_parts.append("  - Muchas memorias con baja confianza - revisar calidad de contenido")
    
    response_parts.append("")
    response_parts.append("[OK] Analisis de patrones completado")
    
    return response_parts

def main():
    try:
        # Get memory store
        memory_store = get_memory_store()
        if not memory_store:
            print("[ERROR] No se pudo acceder al almacen de memorias")
            sys.exit(1)

        # Get all memories for analysis (sample for performance)
        print("[INFO] Obteniendo memorias para analisis...", file=sys.stderr)
        all_memories = memory_store.get_all()
        
        if not all_memories:
            print("[INFO] No hay memorias para analizar.")
            sys.exit(0)

        # Sample for performance (max 50 memories)
        memories = all_memories[:50] if len(all_memories) > 50 else all_memories
        print(f"[INFO] Analizando {len(memories)} memorias...", file=sys.stderr)

        # Initialize CXD classifier v2.0
        cxd_classifier = init_cxd_classifier()
        
        # Classify memories for pattern analysis
        classifications = {}
        if cxd_classifier:
            print("[INFO] Clasificando memorias con CXD v2.0...", file=sys.stderr)
            classifications = classify_memory_batch(cxd_classifier, memories)
        else:
            print("[WARNING] Analisis sin clasificacion CXD", file=sys.stderr)

        # Perform pattern analyses
        print("[INFO] Analizando patrones temporales...", file=sys.stderr)
        temporal_patterns = analyze_temporal_patterns(memories, classifications)
        
        print("[INFO] Analizando patrones de contenido...", file=sys.stderr)
        content_patterns = analyze_content_patterns(memories, classifications)
        
        print("[INFO] Analizando patrones de uso...", file=sys.stderr)
        usage_patterns = analyze_usage_patterns(memories)
        
        print("[INFO] Generando insights...", file=sys.stderr)
        insights = generate_insights(memories, classifications, temporal_patterns, 
                                   content_patterns, usage_patterns)

        # Format and output results
        output = format_pattern_analysis(memories, classifications, temporal_patterns,
                                       content_patterns, usage_patterns, insights)
        
        print("\n".join(output))

    except Exception as e:
        print(f"[ERROR] Error en analisis de patrones: {str(e)}", file=sys.stderr)
        print(f"[ERROR] Error al analizar patrones de memoria: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
