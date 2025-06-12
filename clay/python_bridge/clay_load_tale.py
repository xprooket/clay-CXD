#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clay Load Tale - Personal Narrative Loading Tool
Load my autobiographical tales for identity recovery
"""

import sys
import os
import json
import argparse

# FORCE UTF-8 I/O
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Add Clay to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from clay.tale_manager import TaleManager
except ImportError as e:
    print(f"[ERROR] Error importing TaleManager: {e}", file=sys.stderr)
    print("[ERROR] Could not import Clay TaleManager")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Load a personal tale")
    parser.add_argument("name", help="Name of the tale to load")
    parser.add_argument("--category", "-c", help="Specific category to search in")
    parser.add_argument("--json", action="store_true", help="Output JSON format")
    parser.add_argument("--metadata", "-m", action="store_true", help="Show metadata only")
    parser.add_argument("--preview", "-p", type=int, help="Show only first N characters")
    
    # Handle MCP calls
    if len(sys.argv) == 1:
        print("[ERROR] Tale name is required")
        print("[USAGE] clay_load_tale <name> [options]")
        sys.exit(1)
    
    args = parser.parse_args()
    
    try:
        # Initialize TaleManager
        tale_manager = TaleManager()
        
        # Load tale
        tale = tale_manager.load_tale(args.name, args.category)
        
        if not tale:
            if args.json:
                error_result = {
                    "status": "error",
                    "error": "tale_not_found",
                    "message": f"Tale '{args.name}' not found"
                }
                print(json.dumps(error_result, indent=2, ensure_ascii=False))
            else:
                print(f"❌ ERROR: Tale '{args.name}' not found", file=sys.stderr)
                
                # Suggest similar tales
                all_tales = tale_manager.list_tales()
                similar = [t for t in all_tales if args.name.lower() in t['name'].lower()]
                
                if similar:
                    print("\n💡 Similar tales found:", file=sys.stderr)
                    for t in similar[:3]:
                        print(f"   - {t['name']} ({t['category']})", file=sys.stderr)
            sys.exit(1)
        
        if args.json:
            # JSON output for programmatic use
            content = tale.content
            if args.preview:
                content = content[:args.preview]
                if len(tale.content) > args.preview:
                    content += "..."
            
            result = {
                "status": "success",
                "tale": {
                    "name": tale.name,
                    "category": tale.category,
                    "content": content,
                    "metadata": tale.metadata,
                    "tags": tale.tags,
                    "filename": tale.get_filename()
                }
            }
            print(json.dumps(result, indent=2, ensure_ascii=False))
        
        elif args.metadata:
            # Show only metadata
            print(f"📖 TALE METADATA: {tale.name}")
            print("=" * 50)
            print(f"📂 Category: {tale.category}")
            print(f"📄 Filename: {tale.get_filename()}")
            print(f"📏 Size: {tale.metadata.get('size_chars', 0)} characters")
            print(f"🆔 Version: {tale.metadata.get('version', 1)}")
            print(f"📊 Usage Count: {tale.metadata.get('usage_count', 0)}")
            print(f"📅 Created: {tale.metadata.get('created', '')}")
            print(f"🔄 Updated: {tale.metadata.get('updated', '')}")
            if tale.tags:
                print(f"🏷️  Tags: {', '.join(tale.tags)}")
            
            # Additional metadata
            for key, value in tale.metadata.items():
                if key not in ['size_chars', 'version', 'usage_count', 'created', 'updated']:
                    print(f"📋 {key.title()}: {value}")
        
        else:
            # Full human-readable output
            print(f"📖 TALE LOADED: {tale.name}")
            print("=" * 60)
            print(f"📂 Category: {tale.category}")
            print(f"📏 Size: {tale.metadata.get('size_chars', 0)} characters")
            print(f"🆔 Version: {tale.metadata.get('version', 1)}")
            if tale.tags:
                print(f"🏷️  Tags: {', '.join(tale.tags)}")
            print(f"📅 Created: {tale.metadata.get('created', '')}")
            print("")
            
            # Content
            content = tale.content
            if args.preview:
                content = content[:args.preview]
                if len(tale.content) > args.preview:
                    content += f"\n\n[...truncated, showing first {args.preview} characters]"
            
            print("📝 CONTENT:")
            print("-" * 40)
            print(content)
            print("-" * 40)
            print("")
            
            print(f"💡 Use 'clay_update_tale {tale.name}' to modify this tale")
            print(f"💡 Use 'clay_search_tales \"{tale.name}\"' to find related tales")
    
    except Exception as e:
        if args.json:
            error_result = {
                "status": "error",
                "error": str(e),
                "message": f"Failed to load tale: {str(e)}"
            }
            print(json.dumps(error_result, indent=2, ensure_ascii=False))
        else:
            print(f"❌ ERROR: Failed to load tale: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
