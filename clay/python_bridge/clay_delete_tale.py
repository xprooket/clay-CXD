#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clay Delete Tale - Personal Narrative Deletion Tool
Safely remove tales from my autobiographical collection
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
    parser = argparse.ArgumentParser(description="Delete a personal tale")
    parser.add_argument("name", help="Name of the tale to delete")
    parser.add_argument("--category", "-c", help="Specific category to search in")
    parser.add_argument("--hard", action="store_true", help="Hard delete (permanent, not recommended)")
    parser.add_argument("--confirm", action="store_true", help="Skip confirmation prompt")
    parser.add_argument("--json", action="store_true", help="Output JSON format")
    
    args = parser.parse_args()
    
    try:
        # Initialize TaleManager
        tale_manager = TaleManager()
        
        # Load tale to show what will be deleted
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
            sys.exit(1)
        
        # Show tale information
        if not args.json and not args.confirm:
            print(f"🗑️  TALE TO DELETE: {tale.name}")
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
            print("")
            
            # Show preview
            preview = tale.content[:200]
            if len(tale.content) > 200:
                preview += "..."
            print("📝 CONTENT PREVIEW:")
            print(f'"{preview}"')
            print("")
            
            # Confirmation
            delete_type = "HARD DELETE (PERMANENT)" if args.hard else "soft delete (move to archive)"
            print(f"⚠️  This will {delete_type} the tale.")
            
            try:
                confirm = input("Are you sure? Type 'yes' to confirm: ").strip().lower()
                if confirm != 'yes':
                    print("❌ Deletion cancelled")
                    return
            except (EOFError, KeyboardInterrupt):
                print("\n❌ Deletion cancelled")
                return
        
        # Perform deletion
        success = tale_manager.delete_tale(
            name=args.name,
            category=args.category,
            soft_delete=not args.hard
        )
        
        if success:
            if args.json:
                result = {
                    "status": "success",
                    "tale": {
                        "name": tale.name,
                        "category": tale.category,
                        "filename": tale.get_filename()
                    },
                    "delete_type": "hard" if args.hard else "soft",
                    "message": f"Tale '{tale.name}' deleted successfully"
                }
                print(json.dumps(result, indent=2, ensure_ascii=False))
            else:
                delete_type = "hard deleted (permanent)" if args.hard else "soft deleted (moved to archive)"
                print(f"✅ Tale '{tale.name}' {delete_type}")
                
                if not args.hard:
                    print("💡 Soft deleted tales are moved to the archive category")
                    print("💡 Use clay_list_tales --category archive to see archived tales")
        else:
            if args.json:
                error_result = {
                    "status": "error",
                    "error": "delete_failed",
                    "message": f"Failed to delete tale '{args.name}'"
                }
                print(json.dumps(error_result, indent=2, ensure_ascii=False))
            else:
                print(f"❌ ERROR: Failed to delete tale '{args.name}'", file=sys.stderr)
            sys.exit(1)
    
    except Exception as e:
        if args.json:
            error_result = {
                "status": "error",
                "error": str(e),
                "message": f"Delete operation failed: {str(e)}"
            }
            print(json.dumps(error_result, indent=2, ensure_ascii=False))
        else:
            print(f"❌ ERROR: Delete operation failed: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
