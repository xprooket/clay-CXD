#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clay Memory Bridge - CXD Classify Tool (Updated v2.0)
Classify text using CXD cognitive framework with new classifier
"""

import sys
import os


# FORCE UTF-8 I/O - CRITICAL para Windows
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

    
# Add Clay to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Add CXD classifier path - use relative path within project
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cxd_path = os.path.join(os.path.dirname(project_root), "cxd-classifier", "src")
if os.path.exists(cxd_path):
    sys.path.insert(0, cxd_path)

def init_cxd_classifier():
    """Initialize CXD classifier v2.0"""
    try:
        from cxd_classifier.classifiers.meta import MetaCXDClassifier
        from cxd_classifier.core.config import CXDConfig
        
        config = CXDConfig()
        classifier = MetaCXDClassifier(config=config)
        return classifier
    except Exception as e:
        print(f"[ERROR] Error inicializando CXD v2.0: {e}", file=sys.stderr)
        return None

def format_cxd_sequence(sequence, label):
    """Format a CXD sequence for display"""
    if not sequence or not sequence.tags:
        return f"  {label}: Sin clasificación"

    # Get dominant function
    dominant = sequence.dominant_function

    # Format tags
    tag_strings = []
    for tag in sequence.tags:
        confidence_bar = "#" * int(tag.confidence * 5) + "-" * (5 - int(tag.confidence * 5))
        tag_strings.append(f"{tag.function.value}({tag.confidence:.2f})")

    return f"  {label}: {dominant.value} | {' + '.join(tag_strings)}"

def main():
    if len(sys.argv) < 2:
        print("[ERROR] Error: Falta texto para clasificar")
        sys.exit(1)

    try:
        text = sys.argv[1]

        # Initialize CXD classifier v2.0
        cxd_classifier = init_cxd_classifier()
        if not cxd_classifier:
            print("[ERROR] CXD Classifier v2.0 no disponible")
            sys.exit(1)

        # Classify text using detailed classification
        result = cxd_classifier.classify_detailed(text)

        # Format response
        response_parts = []

        response_parts.append("[CXD] ANÁLISIS CXD COGNITIVO v2.0")
        response_parts.append("=" * 50)
        response_parts.append(f"[INPUT] Texto: {text}")
        response_parts.append("")

        response_parts.append("[RESULTS] CLASIFICACIONES:")
        response_parts.append(format_cxd_sequence(result.lexical_sequence, "Léxico   "))
        response_parts.append(format_cxd_sequence(result.semantic_sequence, "Semántico"))
        response_parts.append(format_cxd_sequence(result.final_sequence, "[FINAL] "))
        response_parts.append("")

        # Confidence scores
        response_parts.append("[METRICS] MÉTRICAS DE CONFIANZA:")
        scores = result.confidence_scores
        response_parts.append(f"  • Léxico: {scores.get('lexical', 0):.2f}")
        response_parts.append(f"  • Semántico: {scores.get('semantic', 0):.2f}")
        response_parts.append(f"  • Concordancia: {scores.get('concordance', 0):.2f}")
        response_parts.append(f"  • Final: {scores.get('final', 0):.2f}")
        response_parts.append("")

        # Processing info
        response_parts.append("[INFO] INFORMACIÓN DE PROCESAMIENTO:")
        response_parts.append(f"  [TIME] Tiempo: {result.processing_time_ms:.1f}ms")

        if result.corrections_made:
            response_parts.append("  [CORRECTIONS] Correcciones aplicadas:")
            for correction in result.corrections_made:
                response_parts.append(f"    • {correction}")
        response_parts.append("")

        # Interpretation
        final_function = result.final_sequence.dominant_function
        response_parts.append("[INTERPRETATION] INTERPRETACIÓN:")

        if final_function.value == "C":
            response_parts.append("  [C] CONTROL: Búsqueda, filtrado, gestión de flujo")
            response_parts.append("       -> Enfocado en encontrar, seleccionar o dirigir")
        elif final_function.value == "X":
            response_parts.append("  [X] CONTEXT: Conexiones, referencias, memoria")
            response_parts.append("       -> Relacionando con información previa")
        elif final_function.value == "D":
            response_parts.append("  [D] DATA: Procesamiento, análisis, transformación")
            response_parts.append("       -> Trabajando con información directamente")

        response_parts.append("")
        response_parts.append("[OK] Análisis CXD v2.0 completado")

        print("\n".join(response_parts))

    except Exception as e:
        print(f"[ERROR] Error en clasificación CXD v2.0: {str(e)}", file=sys.stderr)
        print(f"[ERROR] Error al clasificar texto: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
