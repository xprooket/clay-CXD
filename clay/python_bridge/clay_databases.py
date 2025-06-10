#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clay Memory Bridge - Database Discovery Tool
Discover and analyze all available Clay databases
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

def discover_clay_databases(search_dir=None):
    """Discover all Clay databases in directory"""
    if search_dir is None:
        search_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    search_path = Path(search_dir)
    databases = []
    
    # Find all .db files
    for db_file in search_path.glob("*.db"):
        try:
            db_info = analyze_database(db_file)
            if db_info:
                databases.append(db_info)
        except Exception as e:
            print(f"[WARNING] Error analyzing {db_file.name}: {e}", file=sys.stderr)
    
    return databases

def analyze_database(db_path):
    """Analyze a database file to see if it's a Clay database"""
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        
        # Check if it has a memories table
        cursor = conn.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='memories'
        """)
        
        if not cursor.fetchone():
            conn.close()
            return None  # Not a Clay database
        
        # Get memory count and types
        cursor = conn.execute("SELECT COUNT(*) as total FROM memories")
        total_memories = cursor.fetchone()['total']
        
        cursor = conn.execute("""
            SELECT type, COUNT(*) as count 
            FROM memories 
            GROUP BY type 
            ORDER BY count DESC
        """)
        memory_types = {row['type']: row['count'] for row in cursor}
        
        # Get last modified memory
        cursor = conn.execute("""
            SELECT created_at 
            FROM memories 
            ORDER BY created_at DESC 
            LIMIT 1
        """)
        last_memory_row = cursor.fetchone()
        last_memory = last_memory_row['created_at'] if last_memory_row else "Never"
        
        # Get file stats
        file_stats = db_path.stat()
        file_size_mb = file_stats.st_size / (1024 * 1024)
        
        conn.close()
        
        # Determine database type/purpose
        db_name = db_path.stem
        if "enhanced" in db_name:
            db_type = "enhanced"
            status = "active" if total_memories > 0 else "empty"
        elif "memories" in db_name and "enhanced" not in db_name:
            db_type = "legacy"
            status = "archive"
        else:
            db_type = "custom"
            status = "unknown"
        
        return {
            "name": db_name,
            "file": db_path.name,
            "path": str(db_path),
            "type": db_type,
            "status": status,
            "size_mb": round(file_size_mb, 2),
            "memory_count": total_memories,
            "memory_types": memory_types,
            "last_memory": last_memory,
            "file_modified": file_stats.st_mtime
        }
        
    except Exception as e:
        print(f"[ERROR] Failed to analyze {db_path}: {e}", file=sys.stderr)
        return None

def format_database_report(databases):
    """Format databases into a nice report (no emojis, encoding-safe)"""
    if not databases:
        return "[INFO] No Clay databases found in current directory."
    
    # Sort by file modification time (most recent first)
    databases.sort(key=lambda x: x['file_modified'], reverse=True)
    
    report = []
    report.append("CLAY DATABASES DISCOVERED")
    report.append("=" * 50)
    
    # Summary
    total_memories = sum(db['memory_count'] for db in databases)
    total_size = sum(db['size_mb'] for db in databases)
    
    report.append(f"SUMMARY:")
    report.append(f"  - Databases found: {len(databases)}")
    report.append(f"  - Total memories: {total_memories:,}")
    report.append(f"  - Total size: {total_size:.1f} MB")
    report.append("")
    
    # Recommended default
    if databases:
        primary_db = databases[0]  # Most recently modified
        report.append(f"RECOMMENDED DEFAULT: {primary_db['name']}")
        report.append(f"   ({primary_db['memory_count']} memories, most recent)")
        report.append("")
    
    # Detailed list
    report.append("DETAILED LIST:")
    report.append("")
    
    for i, db in enumerate(databases, 1):
        # Status indicator (text-based, no emojis)
        if db['status'] == 'active':
            indicator = "[ACTIVE]"
        elif db['status'] == 'archive':
            indicator = "[ARCHIVE]"
        else:
            indicator = "[UNKNOWN]"
        
        report.append(f"{i}. {indicator} {db['name']}")
        report.append(f"   File: {db['file']}")
        report.append(f"   Size: {db['size_mb']} MB | Memories: {db['memory_count']:,}")
        
        if db['memory_types']:
            types_str = ", ".join([f"{t}({c})" for t, c in list(db['memory_types'].items())[:3]])
            if len(db['memory_types']) > 3:
                types_str += f", +{len(db['memory_types'])-3} more"
            report.append(f"   Types: {types_str}")
        
        report.append(f"   Last memory: {db['last_memory'][:19] if db['last_memory'] != 'Never' else 'Never'}")
        report.append("")
    
    # Usage examples
    report.append("USAGE EXAMPLES:")
    if databases:
        example_db = databases[0]['name']
        report.append(f"  clay_recall_cxd \"query\" --db={example_db}")
        report.append(f"  clay_recall \"query\" --db={example_db}")
    report.append("  clay_databases --path=/other/directory")
    
    return "\n".join(report)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Discover Clay databases")
    parser.add_argument("--path", help="Directory to search (default: current Clay directory)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    
    args = parser.parse_args()
    
    try:
        databases = discover_clay_databases(args.path)
        
        if args.json:
            import json
            result = {
                "databases_found": len(databases),
                "total_memories": sum(db['memory_count'] for db in databases),
                "databases": databases
            }
            print(json.dumps(result, indent=2))
        else:
            report = format_database_report(databases)
            print(report)
            
    except Exception as e:
        print(f"[ERROR] Discovery failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
