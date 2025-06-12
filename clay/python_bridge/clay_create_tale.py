#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clay Create Tale - Personal Narrative Creation Tool
Create new autobiographical tales for identity continuity
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
    parser = argparse.ArgumentParser(description="Create a new personal tale")
    parser.add_argument("name", help="Name of the tale")
    parser.add_argument("content", nargs='?', default="", help="Initial content (optional)")
    parser.add_argument("--category", "-c", default="core", 
                       choices=['core', 'contexts', 'insights', 'current', 'archive'],
                       help="Tale category")
    parser.add_argument("--tags", "-t", help="Comma-separated tags")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing tale")
    parser.add_argument("--json", action="store_true", help="Output JSON format")
    
    # For MCP calls without args, provide interactive mode
    if len(sys.argv) == 1:
        try:
            # Interactive mode for MCP
            name = input("Tale name: ").strip()
            if not name:
                print("[ERROR] Tale name is required")
                sys.exit(1)
            
            content = input("Initial content (optional): ").strip()
            category = input("Category [core]: ").strip() or "core"
            tags_input = input("Tags (comma-separated, optional): ").strip()
            
            args = type('Args', (), {
                'name': name,
                'content': content,
                'category': category,
                'tags': tags_input,
                'overwrite': False,
                'json': False
            })()
            
        except (EOFError, KeyboardInterrupt):
            print("\n[CANCELLED] Tale creation cancelled")
            sys.exit(1)
    else:
        args = parser.parse_args()
    
    try:
        # Initialize TaleManager
        tale_manager = TaleManager()
        
        # Parse tags
        tags = []
        if args.tags:
            tags = [tag.strip() for tag in args.tags.split(',') if tag.strip()]
        
        # Create tale
        tale = tale_manager.create_tale(
            name=args.name,
            content=args.content,
            category=args.category,
            tags=tags,
            overwrite=args.overwrite
        )
        
        if args.json:
            # JSON output for programmatic use
            result = {
                "status": "success",
                "tale": {
                    "name": tale.name,
                    "category": tale.category,
                    "filename": tale.get_filename(),
                    "size": tale.metadata.get('size_chars', 0),
                    "created": tale.metadata.get('created', ''),
                    "version": tale.metadata.get('version', 1),
                    "tags": tale.tags
                },
                "message": f"Created tale '{tale.name}' in {tale.category}"
            }
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            # Human-readable output
            print(f"✅ TALE CREATED: {tale.name}")
            print("=" * 50)
            print(f"📂 Category: {tale.category}")
            print(f"📄 Filename: {tale.get_filename()}")
            print(f"📏 Size: {tale.metadata.get('size_chars', 0)} characters")
            print(f"🆔 Version: {tale.metadata.get('version', 1)}")
            if tale.tags:
                print(f"🏷️  Tags: {', '.join(tale.tags)}")
            print(f"📅 Created: {tale.metadata.get('created', '')}")
            print("")
            
            if tale.content:
                print("📖 CONTENT PREVIEW:")
                preview = tale.content[:200]
                if len(tale.content) > 200:
                    preview += "..."
                print(preview)
                print("")
            
            print(f"💡 Use 'clay_load_tale {tale.name}' to load this tale")
            print(f"💡 Use 'clay_update_tale {tale.name}' to modify content")
    
    except Exception as e:
        if args.json if 'args' in locals() else False:
            error_result = {
                "status": "error",
                "error": str(e),
                "message": f"Failed to create tale: {str(e)}"
            }
            print(json.dumps(error_result, indent=2, ensure_ascii=False))
        else:
            print(f"❌ ERROR: Failed to create tale: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
