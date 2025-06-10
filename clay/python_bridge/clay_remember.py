#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clay Memory Bridge - Remember Tool (WITH CXD INTEGRATION)
Bridge script with automatic CXD classification for new memories
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

# Add CXD classifier path
cxd_path = r"D:\claude\cxd-classifier\src"
sys.path.insert(0, cxd_path)

try:
    from clay.memory import Memory, MemoryStore
    from clay.assistant import ContextualAssistant
except ImportError as e:
    print(f"[ERROR] Error importing Clay: {e}", file=sys.stderr)
    print("[ERROR] No se pudo importar Clay")
    sys.exit(1)

def init_cxd_classifier():
    """Initialize CXD classifier for automatic classification"""
    try:
        from cxd_classifier.classifiers.meta import MetaCXDClassifier
        from cxd_classifier.core.config import CXDConfig
        
        config = CXDConfig()
        classifier = MetaCXDClassifier(config=config)
        return classifier
    except ImportError as e:
        print(f"[WARNING] CXD not available: {e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"[WARNING] CXD initialization failed: {e}", file=sys.stderr)
        return None

def classify_content_with_cxd(classifier, content):
    """Classify content and return CXD metadata"""
    if not classifier:
        return {}
    
    try:
        # Use basic classify method for speed
        result = classifier.classify(content)
        
        cxd_metadata = {
            'cxd_function': result.dominant_function.value if result.dominant_function else 'UNKNOWN',
            'cxd_confidence': result.average_confidence,
            'cxd_pattern': result.pattern,
            'cxd_execution_pattern': result.execution_pattern,
            'cxd_timestamp': result.timestamp if hasattr(result, 'timestamp') else None,
            'cxd_version': '2.0'
        }
        
        return cxd_metadata
        
    except Exception as e:
        print(f"[WARNING] CXD classification failed: {e}", file=sys.stderr)
        return {'cxd_error': str(e)}

def main():
    if len(sys.argv) < 2:
        print("[ERROR] Faltan argumentos")
        sys.exit(1)
    
    try:
        content = sys.argv[1]
        memory_type = sys.argv[2] if len(sys.argv) > 2 else "interaction"
        
        # Initialize CXD classifier
        cxd_classifier = init_cxd_classifier()
        cxd_status = "[CXD] CXD v2.0 activo" if cxd_classifier else "[WARNING] sin CXD"
        
        # Initialize assistant (use same name as original server)
        assistant = ContextualAssistant("claude_mcp_enhanced")
        
        # Classify content with CXD (if available)
        cxd_metadata = classify_content_with_cxd(cxd_classifier, content)
        
        # Create memory with CXD metadata
        memory = Memory(content, memory_type)
        
        # Add CXD metadata to memory if classification succeeded
        if cxd_metadata and 'cxd_function' in cxd_metadata:
            if not hasattr(memory, 'metadata') or memory.metadata is None:
                memory.metadata = {}
            memory.metadata.update(cxd_metadata)
        
        # Store memory
        memory_id = assistant.memory_store.add(memory)
        
        # Create enhanced success message (NO EMOJIS)
        result_parts = [f"[OK] Guardado en memoria (ID: {memory_id}, Type: {memory_type})"]
        
        if cxd_metadata and 'cxd_function' in cxd_metadata:
            cxd_func = cxd_metadata['cxd_function']
            cxd_conf = cxd_metadata['cxd_confidence']
            result_parts.append(f"[CXD] Clasificado como {cxd_func} (confianza: {cxd_conf:.2f})")
            
            # Add interpretation
            interpretations = {
                'C': 'Control - busqueda, filtrado, gestion',
                'X': 'Context - referencias, relaciones, memoria', 
                'D': 'Data - procesamiento, analisis, transformacion'
            }
            if cxd_func in interpretations:
                result_parts.append(f"[CXD] {interpretations[cxd_func]}")
        
        result_parts.append(f"[STATUS] {cxd_status}")
        
        print("\n".join(result_parts))
        
    except Exception as e:
        print(f"[ERROR] Error al guardar memoria: {str(e)}", file=sys.stderr)
        print(f"[ERROR] Error al guardar: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
