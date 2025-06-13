#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clay Create Tale - Personal Narrative Creation Tool v2.0
Create new autobiographical tales for identity continuity

NEW STRUCTURE:
- claude/*     → Personal continuity  
- projects/*   → Technical documentation
- misc/*       → Everything else
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

def get_valid_categories():
    """Get list of valid categories from TaleManager"""
    try:
        tm = TaleManager()
        return tm.get_valid_categories()
    except Exception:
        # Fallback to basic categories if TaleManager fails
        return ['claude/core', 'claude/contexts', 'claude/insights', 'claude/current', 'claude/archive', 'projects', 'misc']

def main():
    # Get valid categories dynamically
    valid_categories = get_valid_categories()
    
    parser = argparse.ArgumentParser(description="Create a new personal tale")
    parser.add_argument("name", help="Name of the tale")
    parser.add_argument("content", nargs='?', default="", help="Initial content (optional)")
    parser.add_argument("--category", "-c", default="claude/core", 
                       help=f"Tale category. Valid: {', '.join(valid_categories[:5])}...")
    parser.add_argument("--tags", "-t", help="Comma-separated tags")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing tale")
    parser.add_argument("--json", action="store_true", help="Output JSON format")
    parser.add_argument("--list-categories", action="store_true", help="List all valid categories")
    
    # Handle list categories request
    if len(sys.argv) > 1 and '--list-categories' in sys.argv:
        print("📂 VALID CATEGORIES:")
        print("=" * 50)
        
        # Group by main category
        by_main = {}
        for cat in valid_categories:
            if '/' in cat:
                main, sub = cat.split('/', 1)
                if main not in by_main:
                    by_main[main] = []
                by_main[main].append(sub)
            else:
                if cat not in by_main:
                    by_main[cat] = []
        
        for main, subs in sorted(by_main.items()):
            if subs:
                print(f"📁 {main}/")
                for sub in sorted(subs):
                    print(f"   └── {sub}")
            else:
                print(f"📁 {main}/")
        
        print()
        print("💡 EXAMPLES:")
        print("   --category claude/core          # Personal identity")  
        print("   --category projects/clay-cxd    # Technical docs")
        print("   --category misc/stories         # Creative content")
        return
    
    # For MCP calls without args, provide interactive mode
    if len(sys.argv) == 1:
        try:
            # Interactive mode for MCP
            name = input("Tale name: ").strip()
            if not name:
                print("[ERROR] Tale name is required")
                sys.exit(1)
            
            content = input("Initial content (optional): ").strip()
            print(f"Valid categories: {', '.join(valid_categories[:5])}...")
            category = input("Category [claude/core]: ").strip() or "claude/core"
            tags_input = input("Tags (comma-separated, optional): ").strip()
            
            args = type('Args', (), {
                'name': name,
                'content': content,
                'category': category,
                'tags': tags_input,
                'overwrite': False,
                'json': False,
                'list_categories': False
            })()
            
        except (EOFError, KeyboardInterrupt):
            print("\n[CANCELLED] Tale creation cancelled")
            sys.exit(1)
    else:
        args = parser.parse_args()
    
    try:
        # Initialize TaleManager
        tale_manager = TaleManager()
        
        # Validate category
        if args.category not in valid_categories:
            # Try to be helpful with backward compatibility
            if args.category in ['core', 'contexts', 'insights', 'current', 'archive']:
                new_category = f"claude/{args.category}"
                print(f"⚠️  CATEGORY MAPPING: '{args.category}' → '{new_category}'")
                args.category = new_category
            else:
                print(f"❌ ERROR: Invalid category '{args.category}'")
                print(f"Valid categories: {', '.join(valid_categories)}")
                print("Use --list-categories to see all options")
                sys.exit(1)
        
        # Parse tags
        tags = []
        if args.tags:
            tags = [tag.strip() for tag in args.tags.split(',') if tag.strip()]
        
        # 💡 Reminder before creating
        print("💡 Consider: Would updating an existing tale be better?")
        print("   Use 'list_tales' or 'search_tales' to find related content first")
        print("   Update preserves continuity better than creating new")
        print()
        
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
                "message": f"Created tale '{tale.name}' in {tale.category}",
                "structure_version": "2.0"
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
            print(f"💡 Use 'clay_create_tale --list-categories' to see all categories")
    
    except Exception as e:
        if args.json if 'args' in locals() else False:
            error_result = {
                "status": "error",
                "error": str(e),
                "message": f"Failed to create tale: {str(e)}",
                "structure_version": "2.0"
            }
            print(json.dumps(error_result, indent=2, ensure_ascii=False))
        else:
            print(f"❌ ERROR: Failed to create tale: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
