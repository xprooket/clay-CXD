#!/usr/bin/env python3
"""
Clay Memory Bridge - Recall Tool v2.0
Enhanced search with refs patterns, help system, and smart discoverability
"""

import sys
import os
import time
from pathlib import Path

# FORCE UTF-8 I/O - CRITICAL para Windows
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

    
# Add Clay to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from clay.assistant import ContextualAssistant
except ImportError as e:
    print(f"[ERROR] Error importing Clay: {e}", file=sys.stderr)
    print("[ERROR] No se pudo importar Clay")
    sys.exit(1)


# =============================================================================
# SMART DISCOVERABILITY & FIRST-RUN BOOTSTRAP
# =============================================================================

def get_project_root():
    """Get Clay project root directory."""
    return Path(__file__).parent.parent

def check_first_recall_session():
    """Check if this is first recall in session and return bootstrap hint."""
    try:
        flag_file = get_project_root() / ".first_recall_shown"
        
        # Check if flag exists and is recent (same session/day)
        if flag_file.exists():
            import datetime
            flag_time = datetime.datetime.fromtimestamp(flag_file.stat().st_mtime)
            now = datetime.datetime.now()
            
            # If flag is from today, first recall already happened
            if flag_time.date() == now.date():
                return False
        
        # Create/update flag file - this IS first recall
        flag_file.touch()
        return True
        
    except Exception as e:
        print(f"[DEBUG] First recall check failed: {e}", file=sys.stderr)
        return False  # Default to not showing if unsure


def get_bootstrap_count(assistant):
    """Get count of available bootstrap=critical memories."""
    try:
        cursor = assistant.memory_store.conn.execute(
            "SELECT COUNT(*) FROM memories WHERE content LIKE ?",
            ("%refs: %bootstrap=critical%",)
        )
        count = cursor.fetchone()[0]
        return count
    except Exception:
        return 0


def show_first_recall_bootstrap(assistant):
    """Show first-recall bootstrap with essential startup memories."""
    try:
        bootstrap_count = get_bootstrap_count(assistant)
        
        if bootstrap_count == 0:
            return "ℹ️ No bootstrap memories configured yet."
        
        # Get bootstrap memories
        cursor = assistant.memory_store.conn.execute(
            "SELECT * FROM memories WHERE content LIKE ? ORDER BY created_at DESC LIMIT 5",
            ("%refs: %bootstrap=critical%",)
        )
        
        rows = cursor.fetchall()
        if not rows:
            return "ℹ️ No bootstrap memories found."
        
        # Format bootstrap display
        lines = []
        lines.append("🚀 CLAY MEMORY SYSTEM - FIRST RECALL AUTO-BOOTSTRAP")
        lines.append(f"📋 Loading {len(rows)} essential startup memories:")
        lines.append("")
        
        for i, row in enumerate(rows, 1):
            memory = assistant.memory_store._row_to_memory(row)
            # Show just first line of content for overview
            first_line = memory.content.split('\n')[0].strip()
            if len(first_line) > 80:
                first_line = first_line[:77] + "..."
            lines.append(f"  {i}. [ID: {row['id']}] {first_line}")
        
        lines.append("")
        lines.append("💡 QUICK REFS SYSTEM GUIDE:")
        lines.append("  • recall('bootstrap=critical') - Essential startup memories")
        lines.append("  • recall('help') - Complete refs patterns manual")
        lines.append("  • recall('quarantine=true') - Quarantined/ignored memories")
        lines.append("")
        lines.append("🔧 MEMORY SELF-MANAGEMENT:")
        lines.append("  • Each memory has check=update_memory_guided(ID)→refs:quarantine=true")
        lines.append("  • Copy-paste that command to quarantine memories you don't need")
        lines.append("  • Quarantined memories auto-ignored in future bootstraps")
        lines.append("")
        lines.append("=" * 60)
        lines.append("")
        
        return "\n".join(lines)
        
    except Exception as e:
        print(f"[DEBUG] Bootstrap display failed: {e}", file=sys.stderr)
        return "ℹ️ Bootstrap display failed."


# =============================================================================
# HELP SYSTEM - COMPLETE REFS PATTERNS MANUAL
# =============================================================================

def show_help_manual(assistant):
    """Show complete help manual for refs patterns and system usage."""
    bootstrap_count = get_bootstrap_count(assistant)
    
    # Get quarantine count
    quarantine_count = 0
    try:
        cursor = assistant.memory_store.conn.execute(
            "SELECT COUNT(*) FROM memories WHERE content LIKE ? AND content NOT LIKE ?",
            ("%refs: %quarantine=true%", "%refs: %bootstrap=critical%")
        )
        quarantine_count = cursor.fetchone()[0]
    except Exception:
        pass
    
    # Get total memories count
    total_count = 0
    try:
        cursor = assistant.memory_store.conn.execute("SELECT COUNT(*) FROM memories")
        total_count = cursor.fetchone()[0]
    except Exception:
        pass
    
    help_text = f"""
📚 CLAY MEMORY SYSTEM - COMPLETE REFS PATTERNS MANUAL

🎯 CURRENT STATUS:
  • Total memories: {total_count}
  • Bootstrap critical: {bootstrap_count}
  • Quarantined: {quarantine_count}
  • Active: {total_count - quarantine_count}

🔍 CORE SEARCH COMMANDS:
  recall("query")                    - Standard text search
  recall("bootstrap=critical")       - Essential startup memories  
  recall("quarantine=true")          - Quarantined/ignored memories
  recall("help")                     - This manual
  recall("*", --type golden)         - All memories of specific type

🏷️ REFS PATTERNS (search in refs: section):
  bootstrap=critical     - Memories loaded in auto-bootstrap
  bootstrap=medium       - Extended context memories (if configured)
  quarantine=true        - Quarantined memories (auto-ignored)
  relates_to=ID,ID,ID    - Cross-references to other memories
  supersedes=ID          - This memory replaces older memory ID
  context=session_name   - Memories from specific conversation/session
  check=command          - Auto-quarantine commands (copy-paste to execute)
  validated_today=true   - Recently validated/reviewed memories

🔧 MEMORY SELF-MANAGEMENT WORKFLOW:
  1. Find memory that doesn't serve you
  2. Look for its refs: section with check=update_memory_guided(ID)→refs:quarantine=true
  3. Copy-paste that command to quarantine it
  4. Memory will be auto-ignored in future bootstraps
  5. Use recall("quarantine=true") to review quarantined memories

🚀 QUICK START PATTERNS:
  recall("bootstrap=critical")       - Your essential 5 startup memories
  recall("relates_to=248")           - Memories connected to memory #248
  recall("context=june_8_session")   - Memories from specific conversation
  recall("check=")                   - All memories with auto-quarantine commands

📋 MEMORY TYPES:
  golden        - Personal rules and principles for you
  collaboration - Dynamics and patterns with partners
  project_info  - Technical facts about Clay/CXD systems
  reflection    - Analysis and insights patterns
  milestone     - Important achievements or decisions
  experiment    - Tests and experimental findings

🔄 REFS CROSS-REFERENCING:
  refs: relates_to=247,189 | supersedes=234 | context=june_8_session
  
  This creates navigable memory graphs - follow relates_to chains to explore
  connected ideas and see how memories build on each other.

💡 TIPS:
  • Use refs patterns when regular search doesn't find what you need
  • Bootstrap memories are your "identity backup" - keep them current
  • Quarantine aggressively - better 5 perfect memories than 50 mediocre ones
  • relates_to patterns help you navigate memory relationships
  • check= commands enable one-click memory curation

🛠️ TECHNICAL NOTES:
  • Refs patterns search the "refs:" section at end of memory content
  • Standard queries search full memory content with LIKE patterns
  • Memory IDs are permanent - safe to use in relates_to references
  • Quarantine is non-destructive - memories remain but are auto-ignored

═══════════════════════════════════════════════════════════════════════════
🧮 Clay Memory System v2.0 - Where memory meets meaning
"""
    
    return help_text.strip()


# =============================================================================
# ENHANCED REFS PATTERN MATCHING
# =============================================================================

def search_refs_pattern(assistant, pattern, limit, filter_type=None):
    """Enhanced refs pattern search with support for cross-references."""
    
    # Handle relates_to patterns (relates_to=ID or relates_to=ID,ID,ID)
    if pattern.startswith("relates_to="):
        memory_ids = pattern.replace("relates_to=", "").split(",")
        # Build query for any of the specified IDs
        id_patterns = [f"%relates_to={mid.strip()}%" for mid in memory_ids]
        id_patterns.extend([f"%relates_to={mid.strip()},%"  for mid in memory_ids])  # ID in list
        id_patterns.extend([f"%,{mid.strip()},%"  for mid in memory_ids])  # ID in middle
        id_patterns.extend([f"%,{mid.strip()}%"  for mid in memory_ids])  # ID at end
        
        # Combine with OR conditions
        where_conditions = " OR ".join(["content LIKE ?" for _ in id_patterns])
        
        if filter_type:
            query = f"SELECT * FROM memories WHERE type = ? AND ({where_conditions}) ORDER BY created_at DESC LIMIT ?"
            params = [filter_type] + id_patterns + [limit]
        else:
            query = f"SELECT * FROM memories WHERE ({where_conditions}) ORDER BY created_at DESC LIMIT ?"
            params = id_patterns + [limit]
            
        cursor = assistant.memory_store.conn.execute(query, params)
        
    else:
        # Standard refs pattern search
        if filter_type:
            cursor = assistant.memory_store.conn.execute(
                "SELECT * FROM memories WHERE type = ? AND content LIKE ? ORDER BY created_at DESC LIMIT ?",
                (filter_type, f"%refs: %{pattern}%", limit)
            )
        else:
            cursor = assistant.memory_store.conn.execute(
                "SELECT * FROM memories WHERE content LIKE ? ORDER BY created_at DESC LIMIT ?",
                (f"%refs: %{pattern}%", limit)
            )
    
    rows = cursor.fetchall()
    memories = []
    for row in rows:
        memory = assistant.memory_store._row_to_memory(row)
        memory.id = row['id']  # Ensure ID is set
        memories.append(memory)
    
    return memories


# =============================================================================
# SMART NO-RESULTS DISCOVERABILITY
# =============================================================================

def show_no_results_help(query, is_refs_query, filter_type, assistant):
    """Show smart suggestions when no results found."""
    bootstrap_count = get_bootstrap_count(assistant)
    
    lines = []
    
    if is_refs_query:
        lines.append(f"ℹ️ No memories found with refs pattern '{query}'")
        lines.append("")
        lines.append("💡 Available refs patterns:")
        lines.append("  • bootstrap=critical - Essential startup memories")
        lines.append("  • quarantine=true - Quarantined memories")
        lines.append("  • relates_to=ID - Cross-referenced memories")
        lines.append("  • check= - Memories with auto-quarantine commands")
        lines.append("")
        lines.append("🔍 Try: recall('help') for complete patterns manual")
        
    elif filter_type:
        lines.append(f"ℹ️ No memories of type '{filter_type}' found for '{query}'")
        lines.append("")
        lines.append("💡 Available memory types: golden, collaboration, project_info, reflection, milestone")
        
    else:
        lines.append(f"ℹ️ No memories found for '{query}'")
        lines.append("")
        
        if bootstrap_count > 0:
            lines.append(f"💡 Quick suggestions:")
            lines.append(f"  • recall('bootstrap=critical') - {bootstrap_count} essential startup memories")
            lines.append(f"  • recall('help') - Complete refs patterns manual")
            lines.append(f"  • recall_cxd('{query}') - Try semantic + WordNet search")
        else:
            lines.append("💡 Try:")
            lines.append("  • recall('help') - Complete refs patterns manual")
            lines.append(f"  • recall_cxd('{query}') - Semantic + WordNet search")
            lines.append("  • Different keywords or phrasing")
    
    return "\n".join(lines)


# =============================================================================
# ENHANCED RESULTS FORMATTING
# =============================================================================

def format_memory_result(memory, index, show_refs_highlights=False):
    """Format memory result with optional refs pattern highlighting."""
    lines = []
    
    confidence_bar = "#" * int(memory.confidence * 10) + "-" * (10 - int(memory.confidence * 10))
    memory_id = getattr(memory, 'id', '?')
    
    # Main content
    lines.append(f"{index}. [ID: {memory_id}] [{memory.type.upper()}] {memory.content}")
    lines.append(f"   Confianza: [{confidence_bar}] {memory.confidence:.2f}")
    lines.append(f"   Creado: {memory.created_at}")
    
    if hasattr(memory, 'access_count') and memory.access_count > 0:
        lines.append(f"   Accesos: {memory.access_count}")
    
    # Extract and highlight refs section if present
    if show_refs_highlights and "refs:" in memory.content.lower():
        content_lines = memory.content.split('\n')
        refs_lines = [line for line in content_lines if line.strip().lower().startswith('refs:')]
        if refs_lines:
            lines.append(f"   🏷️ Refs: {refs_lines[0].replace('refs:', '').strip()}")
    
    return lines


# =============================================================================
# MAIN FUNCTION - ENHANCED WITH ALL FEATURES
# =============================================================================

def main():
    if len(sys.argv) < 2:
        print("[ERROR] Falta query de busqueda")
        print("[USAGE] clay_recall.py <query> [limit] [--type TYPE]")
        print("[REFS]  clay_recall.py bootstrap=critical 5")
        print("[REFS]  clay_recall.py relates_to=248,189 10")
        print("[HELP]  clay_recall.py help")
        sys.exit(1)
    
    try:
        # Parse arguments
        query = sys.argv[1]
        limit = 5
        filter_type = None
        
        # Parse optional arguments
        i = 2
        while i < len(sys.argv):
            if sys.argv[i] == '--type' and i + 1 < len(sys.argv):
                filter_type = sys.argv[i + 1]
                i += 2
            elif sys.argv[i].startswith('--'):
                i += 1  # Skip unknown flags
            else:
                # Must be limit
                limit = int(sys.argv[i])
                i += 1
        
        # Initialize assistant (uses claude_mcp_enhanced_memories.db)
        assistant = ContextualAssistant("claude_mcp_enhanced")
        print(f"[DEBUG] Using memory DB: {assistant.db_path}", file=sys.stderr)
        
        # === HELP SYSTEM ===
        if query.lower() in ['help', '--help', '-h']:
            help_content = show_help_manual(assistant)
            print(help_content)
            return
        
        # === FIRST RECALL AUTO-BOOTSTRAP ===
        if check_first_recall_session():
            bootstrap_display = show_first_recall_bootstrap(assistant)
            print(bootstrap_display)
            # Continue with regular search after bootstrap display
        
        # === SEARCH LOGIC ===
        memories = []
        
        # Enhanced refs pattern detection
        refs_patterns = ['bootstrap', 'quarantine', 'check', 'refs', 'relates_to', 'supersedes', 'context', 'validated_today']
        is_refs_query = '=' in query and any(pattern in query for pattern in refs_patterns)
        
        if is_refs_query:
            # ENHANCED REFS PATTERN SEARCH
            print(f"[DEBUG] Detected refs pattern search: '{query}'", file=sys.stderr)
            memories = search_refs_pattern(assistant, query, limit, filter_type)
            print(f"[DEBUG] Found {len(memories)} memories with refs pattern '{query}'", file=sys.stderr)
            
        elif filter_type:
            # TYPE FILTER SEARCH
            if query == "*" or query == "all":
                cursor = assistant.memory_store.conn.execute(
                    "SELECT * FROM memories WHERE type = ? ORDER BY created_at DESC LIMIT ?",
                    (filter_type, limit)
                )
            else:
                cursor = assistant.memory_store.conn.execute(
                    "SELECT * FROM memories WHERE type = ? AND content LIKE ? ORDER BY created_at DESC LIMIT ?",
                    (filter_type, f"%{query}%", limit)
                )
            
            rows = cursor.fetchall()
            for row in rows:
                memory = assistant.memory_store._row_to_memory(row)
                memory.id = row['id']
                memories.append(memory)
                
            print(f"[DEBUG] Found {len(memories)} memories of type '{filter_type}'", file=sys.stderr)
            
        else:
            # STANDARD TEXT SEARCH
            memories = assistant.memory_store.search(query, limit)
            print(f"[DEBUG] Found {len(memories) if memories else 0} memories", file=sys.stderr)
        
        # === RESULTS DISPLAY ===
        if not memories:
            no_results_help = show_no_results_help(query, is_refs_query, filter_type, assistant)
            print(no_results_help)
            return
        
        # Format results header
        results = []
        if is_refs_query:
            results.append(f"🏷️ MEMORIAS CON REFS PATTERN '{query}' ({len(memories)} resultados):\n")
        elif filter_type:
            results.append(f"📁 MEMORIAS DE TIPO '{filter_type.upper()}' ({len(memories)} resultados):\n")
        else:
            results.append(f"🔍 MEMORIAS ENCONTRADAS ({len(memories)} resultados):\n")
        
        # Format individual results
        for i, memory in enumerate(memories, 1):
            memory_lines = format_memory_result(memory, i, show_refs_highlights=is_refs_query)
            results.extend(memory_lines)
            results.append("")  # Empty line
        
        # Add helpful footer for refs queries
        if is_refs_query and memories:
            results.append("💡 Use recall('help') for complete refs patterns manual")
            results.append("")
        
        print("\n".join(results))
        
    except ValueError as ve:
        print(f"[ERROR] Error en argumentos: {str(ve)}", file=sys.stderr)
        print(f"[ERROR] Error en argumentos: {str(ve)}")
        sys.exit(1)
        
    except Exception as e:
        print(f"[ERROR] Error en busqueda: {str(e)}", file=sys.stderr)
        print(f"[ERROR] Error al buscar memorias: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
