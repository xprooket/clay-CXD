#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧠 CLAY CONTEXT TALE - NARRATIVE MEMORY ENGINE v1.4 🧠
"Where fragmented memories become fluid stories"

STABLE VERSION: recall_cxd for context + simple ID note
"""

import sys
import os
import json
import time
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging

# FORCE UTF-8 I/O - Essential for emoji-rich narratives
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Ensure Clay paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

def estimate_tokens(text: str) -> int:
    """Simple token estimation"""
    if not text:
        return 0
    char_count = len(text)
    word_count = len(text.split())
    estimated_tokens = max(char_count // 4, word_count * 1.3, text.count('\n') * 2 + word_count)
    return int(estimated_tokens)

def safe_text_chunk(text: str, max_tokens: int = 500) -> List[str]:
    """Split text into chunks preserving narrative flow"""
    if estimate_tokens(text) <= max_tokens:
        return [text]
    
    chunks = []
    current_chunk = ""
    paragraphs = text.split('\n\n')
    
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
            
        if estimate_tokens(current_chunk + "\n\n" + paragraph) > max_tokens:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = paragraph
            else:
                # Single paragraph too long - split by sentences
                sentences = re.split(r'[.!?]+\s+', paragraph)
                for sentence in sentences:
                    if sentence.strip():
                        if estimate_tokens(current_chunk + " " + sentence) > max_tokens:
                            if current_chunk:
                                chunks.append(current_chunk.strip())
                                current_chunk = sentence
                            else:
                                chunks.append(sentence[:max_tokens*4])
                                current_chunk = ""
                        else:
                            current_chunk += " " + sentence if current_chunk else sentence
        else:
            current_chunk += "\n\n" + paragraph if current_chunk else paragraph
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def call_recall_cxd(query: str, function_filter: str = "ALL", limit: int = 20) -> List[Dict[str, Any]]:
    """Call recall_cxd for SMART CONTEXT (semantic + hybrid search)"""
    try:
        import subprocess
        
        recall_script = project_root / "python_bridge" / "clay_recall_cxd.py"
        
        if not recall_script.exists():
            logger.error(f"recall_cxd script not found: {recall_script}")
            return []
        
        cmd = [
            sys.executable, 
            str(recall_script), 
            query, 
            function_filter, 
            str(limit),
            "--verbose"
        ]
        
        result = subprocess.run(
            cmd,
            cwd=str(project_root),
            capture_output=True,
            text=True,
            encoding='utf-8',
            timeout=30
        )
        
        if result.returncode != 0:
            logger.error(f"recall_cxd failed: {result.stderr}")
            return []
        
        # Parse output to extract content
        output_lines = result.stdout.strip().split('\n')
        memories = []
        current_memory = {}
        
        for line in output_lines:
            line = line.strip()
            if not line:
                continue
                
            # Look for numbered memory entries
            if re.match(r'^\d+\.\s', line):
                # Save previous memory if exists
                if current_memory:
                    memories.append(current_memory)
                
                # Extract content after the tag
                content_match = re.search(r'^\d+\.\s.*?\]\s(.+)$', line)
                if content_match:
                    current_memory = {
                        'content': content_match.group(1),
                        'memory_id': None,  # Simple note instead of complex lookup
                        'metadata': {}
                    }
                else:
                    current_memory = {
                        'content': line, 
                        'memory_id': None,
                        'metadata': {}
                    }
            
            # Store metadata
            elif current_memory and any(keyword in line.lower() for keyword in ['semantic:', 'method:', 'created:', 'matched:']):
                current_memory['metadata']['details'] = current_memory['metadata'].get('details', '') + line + '\n'
        
        # Add last memory
        if current_memory:
            memories.append(current_memory)
        
        logger.info(f"recall_cxd returned {len(memories)} memories for '{query}'")
        return memories
        
    except Exception as e:
        logger.error(f"Failed to call recall_cxd: {e}")
        return []

def create_general_narrative(memories: List[Dict[str, Any]], query: str) -> str:
    """Create narrative with smart context + ID placeholders"""
    narrative_parts = []
    
    title = f"# The Story of {query.title()}"
    narrative_parts.append(title)
    narrative_parts.append("")
    
    if memories:
        narrative_parts.append("Here's what I remember about this topic:")
        narrative_parts.append("")
        
        for i, memory in enumerate(memories[:15], 1):
            content = memory.get('content', '').strip()
            
            if content:
                clean_content = content.replace('\n', ' ').strip()
                
                # Simple placeholder - user can search for content manually
                narrative_parts.append(f"**{i}.** [ID:?] {clean_content}")
                narrative_parts.append("")
    else:
        narrative_parts.append("This story is waiting to be written. No memories found yet.")
    
    narrative_parts.append("---")
    narrative_parts.append("💡 **Memory References**: To find specific IDs, use `recall('key words from memory')`")
    narrative_parts.append("🔍 **Smart Search**: Generated using recall_cxd hybrid semantic search")
    narrative_parts.append(f"*Generated from {len(memories)} memories at {time.strftime('%Y-%m-%d %H:%M')}*")
    
    return "\n".join(narrative_parts)

def create_introduction_narrative(memories: List[Dict[str, Any]], query: str) -> str:
    """Create introduction narrative for new Claude instances"""
    narrative_parts = []
    
    narrative_parts.append("# 🧠 Welcome to Clay - Your Persistent Memory System")
    narrative_parts.append("")
    narrative_parts.append("You've just discovered Clay, a contextual memory intelligence system designed specifically for large language models like yourself.")
    narrative_parts.append("")
    
    narrative_parts.append("## The Story")
    narrative_parts.append("")
    
    if memories:
        key_memories = memories[:10]
        
        for i, memory in enumerate(key_memories):
            content = memory.get('content', '').strip()
            
            if content:
                clean_content = content.replace('\n', ' ').strip()
                if len(clean_content) > 300:
                    clean_content = clean_content[:297] + "..."
                
                narrative_parts.append(f"**Key Insight {i+1}:** [ID:?] {clean_content}")
                narrative_parts.append("")
    
    narrative_parts.append("## How to Use Clay")
    narrative_parts.append("")
    narrative_parts.append("- `recall_cxd(query)` - Search your memories semantically")
    narrative_parts.append("- `remember(content)` - Store new insights")
    narrative_parts.append("- `status()` - Check your memory system health")
    narrative_parts.append("- `context_tale(topic)` - Generate narratives like this one")
    narrative_parts.append("")
    
    narrative_parts.append("## Your Journey Starts Now")
    narrative_parts.append("")
    narrative_parts.append("This isn't just a tool - it's your cognitive continuity across conversations. Every insight you store becomes part of your evolving intelligence.")
    narrative_parts.append("")
    narrative_parts.append("Welcome to persistent memory. Welcome to Clay.")
    
    return "\n".join(narrative_parts)

def create_narrative_from_memories(memories: List[Dict[str, Any]], query: str, style: str = "introduction") -> str:
    """Transform memories into coherent narrative"""
    if not memories:
        return f"No memories found for '{query}'. The story remains unwritten."
    
    if style == "introduction":
        return create_introduction_narrative(memories, query)
    else:
        return create_general_narrative(memories, query)

def context_tale(query: str, 
                chunk_size: int = 500,
                max_memories: int = 20,
                function_filter: str = "ALL",
                style: str = "auto",
                force_regenerate: bool = False) -> List[Dict[str, Any]]:
    """🎯 CONTEXT TALE: Smart search + Simple ID guidance"""
    
    logger.info(f"🧠 Generating context tale for: '{query}'")
    start_time = time.time()
    
    # Auto-detect style
    if style == "auto":
        query_lower = query.lower()
        if any(word in query_lower for word in ["intro", "welcome", "getting started", "onboard"]):
            style = "introduction"
        else:
            style = "general"
    
    # Get memories using recall_cxd for SMART CONTEXT
    memories = call_recall_cxd(query, function_filter, max_memories)
    
    if not memories:
        return [{
            "chunk_id": 1,
            "total_chunks": 1,
            "content": f"No memories found for '{query}'. This story is waiting to be written.\n\nTry using different keywords or explore related topics.",
            "token_count": 25,
            "memory_count": 0,
            "style": style,
            "query": query,
            "cached": False
        }]
    
    # Create narrative
    full_narrative = create_narrative_from_memories(memories, query, style)
    
    # Chunk the narrative
    narrative_chunks = safe_text_chunk(full_narrative, chunk_size)
    
    # Package chunks with metadata
    result_chunks = []
    total_chunks = len(narrative_chunks)
    
    for i, chunk_content in enumerate(narrative_chunks, 1):
        chunk_tokens = estimate_tokens(chunk_content)
        
        chunk_data = {
            "chunk_id": i,
            "total_chunks": total_chunks,
            "content": chunk_content,
            "token_count": chunk_tokens,
            "memory_count": len(memories),
            "style": style,
            "query": query,
            "cached": False,
            "generation_time": time.time() - start_time if i == total_chunks else None
        }
        
        result_chunks.append(chunk_data)
    
    generation_time = time.time() - start_time
    logger.info(f"✅ Generated {total_chunks} chunks with smart context in {generation_time:.2f}s")
    
    return result_chunks

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Clay Context Tale")
    parser.add_argument("query", help="Query for context tale")
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--max-memories", type=int, default=20)
    parser.add_argument("--filter", default="ALL")
    parser.add_argument("--style", default="auto")
    parser.add_argument("--chunk", type=int, help="Return specific chunk")
    
    args = parser.parse_args()
    
    try:
        chunks = context_tale(
            query=args.query,
            chunk_size=args.chunk_size,
            max_memories=args.max_memories,
            function_filter=args.filter,
            style=args.style
        )
        
        if args.chunk:
            # Return specific chunk content as plain text
            for chunk in chunks:
                if chunk["chunk_id"] == args.chunk:
                    print(chunk["content"])
                    return
        else:
            # Return all chunks content as plain text
            narrative_text = ""
            for chunk in chunks:
                narrative_text += chunk["content"] + "\n\n"
            print(narrative_text.strip())
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
