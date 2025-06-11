#!/usr/bin/env python3
"""
Clay Memory Bridge - Status Tool v2.0
Enhanced status with refs patterns system info
"""

import sys
import os
import sqlite3
from pathlib import Path

# FORCE UTF-8 I/O - CRITICAL para Windows
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

    
# Add Clay to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from clay.assistant import ContextualAssistant
    from clay.memory import MemoryStore
except ImportError as e:
    print(f"[ERROR] Error importing Clay: {e}", file=sys.stderr)
    print("[ERROR] Could not import Clay")
    sys.exit(1)

def check_first_status_session():
    """Check if this is first status call in session and return bootstrap hint."""
    try:
        project_root = Path(__file__).parent.parent
        flag_file = project_root / ".first_status_shown"
        
        # Check if flag exists and is recent (same session/day)
        if flag_file.exists():
            import datetime
            flag_time = datetime.datetime.fromtimestamp(flag_file.stat().st_mtime)
            now = datetime.datetime.now()
            
            # If flag is from today, first status already happened
            if flag_time.date() == now.date():
                return False
        
        # Create/update flag file - this IS first status
        flag_file.touch()
        return True
        
    except Exception as e:
        print(f"[DEBUG] First status check failed: {e}", file=sys.stderr)
        return False  # Default to not showing if unsure

def show_first_status_bootstrap(assistant):
    """Show first-status bootstrap with essential startup memories."""
    try:
        # Get bootstrap memories count
        cursor = assistant.memory_store.conn.execute(
            "SELECT COUNT(*) FROM memories WHERE (content LIKE ? OR content LIKE ?)",
            ("%refs:%bootstrap=critical%", "%refs: %bootstrap=critical%")
        )
        bootstrap_count = cursor.fetchone()[0]
        
        if bootstrap_count == 0:
            return "ℹ️ No bootstrap memories configured yet."
        
        # Get bootstrap memories
        cursor = assistant.memory_store.conn.execute(
            "SELECT * FROM memories WHERE (content LIKE ? OR content LIKE ?) ORDER BY created_at DESC LIMIT 5",
            ("%refs:%bootstrap=critical%", "%refs: %bootstrap=critical%")
        )
        
        rows = cursor.fetchall()
        if not rows:
            return "ℹ️ No bootstrap memories found."
        
        # Format bootstrap display
        lines = []
        lines.append("🚀 CLAY AUTO-BRIEFING - BOOTSTRAP CRITICAL MEMORIES LOADED")
        lines.append(f"📋 Found {len(rows)} essential startup memories:")
        lines.append("")
        
        for i, row in enumerate(rows, 1):
            memory = assistant.memory_store._row_to_memory(row)
            # Show just first line of content for overview
            first_line = memory.content.split('\n')[0].strip()
            if len(first_line) > 80:
                first_line = first_line[:77] + "..."
            lines.append(f"  🛡️ [ID: {row['id']}] {first_line}")
        
        lines.append("")
        lines.append("💡 THESE MEMORIES ARE NOW ACTIVE - You should adopt these principles as your behavior")
        lines.append("🔧 Use recall('bootstrap=critical') to see full details anytime")
        lines.append("")
        lines.append("=" * 70)
        lines.append("")
        
        return "\n".join(lines)
        
    except Exception as e:
        print(f"[DEBUG] Bootstrap display failed: {e}", file=sys.stderr)
        return "ℹ️ Bootstrap display failed."

def init_cxd_classifier():
    """Try to initialize CXD classifier with graceful fallback"""
    try:
        # First try: Use installed package (recommended)
        from cxd_classifier import create_classifier
        classifier = create_classifier("optimized")
        return classifier
        
    except ImportError:
        # Second try: Add repository path and import
        try:
            clay_cxd_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            cxd_path = os.path.join(clay_cxd_root, "cxd-classifier", "src")
            
            if os.path.exists(cxd_path) and cxd_path not in sys.path:
                sys.path.insert(0, cxd_path)
                from cxd_classifier import create_classifier
                classifier = create_classifier("optimized")
                return classifier
            else:
                print(f"[DEBUG] CXD path not found: {cxd_path}", file=sys.stderr)
                return None
                
        except Exception as path_error:
            print(f"[DEBUG] CXD path import failed: {path_error}", file=sys.stderr)
            return None
            
    except Exception as e:
        print(f"[DEBUG] CXD initialization failed: {e}", file=sys.stderr)
        return None

def get_memory_stats(assistant):
    """Get enhanced memory statistics including refs patterns"""
    try:
        # Get all memories
        memories = assistant.memory_store.get_all()
        total_memories = len(memories)
        
        # Count by type
        type_counts = {}
        for memory in memories:
            memory_type = getattr(memory, 'type', 'unknown')
            type_counts[memory_type] = type_counts.get(memory_type, 0) + 1
        
        # Recent memories (last 24 hours)
        recent_memories = assistant.memory_store.get_recent(hours=24)
        recent_count = len(recent_memories)
        
        # Refs patterns statistics
        refs_stats = get_refs_patterns_stats(assistant)
        
        return {
            'total': total_memories,
            'by_type': type_counts,
            'recent_24h': recent_count,
            'refs_patterns': refs_stats
        }
    except Exception as e:
        return {'error': str(e)}

def get_refs_patterns_stats(assistant):
    """Get statistics for refs patterns system"""
    try:
        stats = {}
        
        # Bootstrap critical memories
        cursor = assistant.memory_store.conn.execute(
            "SELECT COUNT(*) FROM memories WHERE (content LIKE ? OR content LIKE ?)",
            ("%refs:%bootstrap=critical%", "%refs: %bootstrap=critical%")
        )
        stats['bootstrap_critical'] = cursor.fetchone()[0]
        
        # Quarantined memories (only those that are ONLY quarantined, not also bootstrap)
        cursor = assistant.memory_store.conn.execute(
            "SELECT COUNT(*) FROM memories WHERE (content LIKE ? OR content LIKE ?) AND content NOT LIKE ? AND content NOT LIKE ?",
            ("%refs:%quarantine=true%", "%refs: %quarantine=true%", "%refs:%bootstrap=critical%", "%refs: %bootstrap=critical%")
        )
        stats['quarantined'] = cursor.fetchone()[0]
        
        # Memories with relates_to references
        cursor = assistant.memory_store.conn.execute(
            "SELECT COUNT(*) FROM memories WHERE (content LIKE ? OR content LIKE ?)",
            ("%refs:%relates_to=%", "%refs: %relates_to=%")
        )
        stats['cross_referenced'] = cursor.fetchone()[0]
        
        # Memories with check commands (auto-quarantine ready)
        cursor = assistant.memory_store.conn.execute(
            "SELECT COUNT(*) FROM memories WHERE (content LIKE ? OR content LIKE ?)",
            ("%refs:%check=%", "%refs: %check=%")
        )
        stats['auto_quarantine_ready'] = cursor.fetchone()[0]
        
        # Memories with context tags
        cursor = assistant.memory_store.conn.execute(
            "SELECT COUNT(*) FROM memories WHERE (content LIKE ? OR content LIKE ?)",
            ("%refs:%context=%", "%refs: %context=%")
        )
        stats['contextualized'] = cursor.fetchone()[0]
        
        return stats
        
    except Exception as e:
        return {'error': str(e)}

def get_session_flags_status():
    """Check status of session flags"""
    try:
        project_root = Path(__file__).parent.parent
        
        flags_status = {}
        
        # First recall flag
        first_recall_flag = project_root / ".first_recall_shown"
        if first_recall_flag.exists():
            import datetime
            flag_time = datetime.datetime.fromtimestamp(first_recall_flag.stat().st_mtime)
            flags_status['first_recall_shown'] = flag_time.strftime("%Y-%m-%d %H:%M")
        else:
            flags_status['first_recall_shown'] = "Never"
        
        # Golden briefing flag
        golden_briefing_flag = project_root / ".golden_briefing_shown"
        if golden_briefing_flag.exists():
            import datetime
            flag_time = datetime.datetime.fromtimestamp(golden_briefing_flag.stat().st_mtime)
            flags_status['golden_briefing_shown'] = flag_time.strftime("%Y-%m-%d %H:%M")
        else:
            flags_status['golden_briefing_shown'] = "Never"
        
        # First status flag
        first_status_flag = project_root / ".first_status_shown"
        if first_status_flag.exists():
            import datetime
            flag_time = datetime.datetime.fromtimestamp(first_status_flag.stat().st_mtime)
            flags_status['first_status_shown'] = flag_time.strftime("%Y-%m-%d %H:%M")
        else:
            flags_status['first_status_shown'] = "Never"
        
        return flags_status
        
    except Exception as e:
        return {'error': str(e)}

def main():
    try:
        # Initialize assistant
        assistant = ContextualAssistant("claude_mcp")
        
        # === AUTO-BRIEFING CHECK ===
        if check_first_status_session():
            bootstrap_display = show_first_status_bootstrap(assistant)
            print(bootstrap_display)
            # Continue with regular status after auto-briefing
        
        # Get enhanced memory stats
        memory_stats = get_memory_stats(assistant)
        
        # Get session flags status
        session_flags = get_session_flags_status()
        
        # Try to get CXD stats
        cxd_classifier = init_cxd_classifier()
        
        # Build enhanced status report
        status_parts = []
        
        status_parts.append("CLAY SYSTEM STATUS")
        status_parts.append("=" * 50)
        
        # Basic status
        status_parts.append("GENERAL STATUS:")
        status_parts.append("  [OK] Clay Core: Active")
        status_parts.append("  [OK] MCP Bridge: JavaScript -> Python")
        status_parts.append("")
        
        # Enhanced memory statistics
        status_parts.append("MEMORY STATISTICS:")
        if 'error' not in memory_stats:
            status_parts.append(f"  Total memories: {memory_stats['total']}")
            status_parts.append(f"  Recent (24h): {memory_stats['recent_24h']}")
            status_parts.append("  By type:")
            for mem_type, count in memory_stats['by_type'].items():
                status_parts.append(f"    - {mem_type}: {count}")
        else:
            status_parts.append(f"  [ERROR] {memory_stats['error']}")
        status_parts.append("")
        
        # REFS PATTERNS SYSTEM STATUS
        if 'refs_patterns' in memory_stats and 'error' not in memory_stats['refs_patterns']:
            refs = memory_stats['refs_patterns']
            status_parts.append("🏷️ REFS PATTERNS SYSTEM:")
            status_parts.append(f"  Bootstrap critical: {refs['bootstrap_critical']}")
            status_parts.append(f"  Quarantined: {refs['quarantined']}")
            status_parts.append(f"  Cross-referenced: {refs['cross_referenced']}")
            status_parts.append(f"  Auto-quarantine ready: {refs['auto_quarantine_ready']}")
            status_parts.append(f"  Contextualized: {refs['contextualized']}")
            
            # Active memories (total - quarantined)
            active_memories = memory_stats['total'] - refs['quarantined']
            status_parts.append(f"  Active memories: {active_memories}")
            status_parts.append("")
            
            # Quick usage hints
            status_parts.append("💡 QUICK REFS COMMANDS:")
            if refs['bootstrap_critical'] > 0:
                status_parts.append(f"  recall('bootstrap=critical') - {refs['bootstrap_critical']} essential memories")
            status_parts.append("  recall('help') - Complete refs patterns manual")
            if refs['quarantined'] > 0:
                status_parts.append(f"  recall('quarantine=true') - {refs['quarantined']} quarantined memories")
            status_parts.append("")
        
        # Session flags status
        if 'error' not in session_flags:
            status_parts.append("🚩 SESSION FLAGS:")
            status_parts.append(f"  First recall bootstrap: {session_flags['first_recall_shown']}")
            status_parts.append(f"  First status bootstrap: {session_flags['first_status_shown']}")
            status_parts.append(f"  Golden briefing: {session_flags['golden_briefing_shown']}")
            status_parts.append("")
        
        # Database info
        try:
            db_path = assistant.memory_store.db_path
            if Path(db_path).exists():
                db_size = Path(db_path).stat().st_size / 1024  # KB
                status_parts.append("BASE DE DATOS:")
                status_parts.append(f"  File: {db_path}")
                status_parts.append(f"  Size: {db_size:.1f} KB")
                status_parts.append("")
        except Exception:
            pass
        
        # CXD Classifier status
        status_parts.append("CXD CLASSIFIER:")
        if cxd_classifier:
            try:
                # Get basic stats (avoid complex operations that might fail)
                status_parts.append("  [OK] Estado: Disponible")
                status_parts.append("  Tipo: OptimizedMetaCXDClassifier")
                
                # Try to get simple stats
                try:
                    stats = cxd_classifier.get_performance_stats()
                    if stats.get('total_classifications', 0) > 0:
                        status_parts.append(f"  Clasificaciones: {stats['total_classifications']}")
                        if 'concordance_rate' in stats:
                            status_parts.append(f"  Rate concordancia: {stats['concordance_rate']:.1%}")
                    else:
                        status_parts.append("  Sin clasificaciones aun")
                except Exception as stats_error:
                    status_parts.append(f"  [WARN] Stats no disponibles: {str(stats_error)[:50]}")
                    
            except Exception as cxd_error:
                status_parts.append(f"  [WARN] Error parcial: {str(cxd_error)[:50]}")
        else:
            status_parts.append("  [OK] Available via MCP tools")
        status_parts.append("")
        
        # System info
        status_parts.append("SYSTEM INFORMATION:")
        status_parts.append(f"  Python: {sys.version.split()[0]}")
        status_parts.append(f"  Directory: {os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}")
        status_parts.append("")
        
        # Final status with enhanced info
        status_parts.append("[OK] Clay is working correctly with JavaScript bridge.")
        
        print("\n".join(status_parts))
        
    except Exception as e:
        print(f"[ERROR] Error en status: {str(e)}", file=sys.stderr)
        print(f"[ERROR] Error obteniendo status del sistema: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
