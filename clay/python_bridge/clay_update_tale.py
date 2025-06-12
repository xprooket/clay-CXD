#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clay Update Tale - Personal Narrative Update Tool
Update my autobiographical tales as I evolve
"""

import sys
import os
import json
import argparse
import tempfile
import subprocess

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

def get_editor():
    """Get preferred text editor"""
    editors = [
        os.environ.get('EDITOR'),
        os.environ.get('VISUAL'),
        'nano', 'vim', 'vi', 'notepad.exe', 'code', 'subl'
    ]
    
    for editor in editors:
        if editor:
            try:
                # Check if editor exists
                subprocess.run([editor, '--version'], 
                             stdout=subprocess.DEVNULL, 
                             stderr=subprocess.DEVNULL, 
                             timeout=2)
                return editor
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
                continue
    
    return None

def edit_with_editor(content: str) -> str:
    """Open content in external editor"""
    editor = get_editor()
    if not editor:
        raise Exception("No suitable text editor found. Set EDITOR environment variable.")
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
        f.write(content)
        temp_file = f.name
    
    try:
        # Open in editor
        subprocess.run([editor, temp_file], check=True)
        
        # Read back content
        with open(temp_file, 'r', encoding='utf-8') as f:
            updated_content = f.read()
        
        return updated_content
    
    finally:
        # Clean up
        try:
            os.unlink(temp_file)
        except:
            pass

def main():
    parser = argparse.ArgumentParser(description="Update a personal tale")
    parser.add_argument("name", help="Name of the tale to update")
    parser.add_argument("--category", "-c", help="Specific category to search in")
    parser.add_argument("--content", help="New content (if not provided, opens editor)")
    parser.add_argument("--append", "-a", help="Append content to existing")
    parser.add_argument("--prepend", "-p", help="Prepend content to existing")
    parser.add_argument("--json", action="store_true", help="Output JSON format")
    parser.add_argument("--editor", "-e", action="store_true", help="Force editor mode")
    
    args = parser.parse_args()
    
    try:
        # Initialize TaleManager
        tale_manager = TaleManager()
        
        # Load existing tale
        existing_tale = tale_manager.load_tale(args.name, args.category)
        if not existing_tale:
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
        
        # Determine new content
        new_content = None
        
        if args.append:
            new_content = existing_tale.content + "\n\n" + args.append
        elif args.prepend:
            new_content = args.prepend + "\n\n" + existing_tale.content
        elif args.content:
            new_content = args.content
        elif args.editor:
            try:
                new_content = edit_with_editor(existing_tale.content)
            except Exception as e:
                if args.json:
                    error_result = {
                        "status": "error",
                        "error": "editor_failed",
                        "message": f"Editor failed: {str(e)}"
                    }
                    print(json.dumps(error_result, indent=2, ensure_ascii=False))
                else:
                    print(f"❌ ERROR: Editor failed: {str(e)}", file=sys.stderr)
                sys.exit(1)
        else:
            # Interactive mode - show current content and ask for new
            if not args.json:
                print(f"📖 CURRENT CONTENT OF '{existing_tale.name}':")
                print("=" * 50)
                print(existing_tale.content)
                print("=" * 50)
                print("")
                print("Enter new content (Ctrl+D or Ctrl+Z to finish):")
                print("(Leave empty to open in editor)")
                
                try:
                    lines = []
                    while True:
                        try:
                            line = input()
                            lines.append(line)
                        except EOFError:
                            break
                    
                    if lines:
                        new_content = '\n'.join(lines)
                    else:
                        # Open editor
                        new_content = edit_with_editor(existing_tale.content)
                        
                except KeyboardInterrupt:
                    print("\n[CANCELLED] Update cancelled")
                    sys.exit(1)
            else:
                # JSON mode requires explicit content
                error_result = {
                    "status": "error",
                    "error": "no_content",
                    "message": "Content must be provided in JSON mode"
                }
                print(json.dumps(error_result, indent=2, ensure_ascii=False))
                sys.exit(1)
        
        # Check if content actually changed
        if new_content == existing_tale.content:
            if args.json:
                result = {
                    "status": "success",
                    "message": "No changes made",
                    "tale": {
                        "name": existing_tale.name,
                        "category": existing_tale.category,
                        "version": existing_tale.metadata.get('version', 1)
                    }
                }
                print(json.dumps(result, indent=2, ensure_ascii=False))
            else:
                print("ℹ️  No changes made to tale content")
            return
        
        # Update tale
        updated_tale = tale_manager.update_tale(args.name, new_content, args.category)
        
        if args.json:
            result = {
                "status": "success",
                "tale": {
                    "name": updated_tale.name,
                    "category": updated_tale.category,
                    "filename": updated_tale.get_filename(),
                    "size": updated_tale.metadata.get('size_chars', 0),
                    "version": updated_tale.metadata.get('version', 1),
                    "updated": updated_tale.metadata.get('updated', ''),
                    "tags": updated_tale.tags
                },
                "message": f"Updated tale '{updated_tale.name}'"
            }
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print(f"✅ TALE UPDATED: {updated_tale.name}")
            print("=" * 50)
            print(f"📂 Category: {updated_tale.category}")
            print(f"📄 Filename: {updated_tale.get_filename()}")
            print(f"📏 New Size: {updated_tale.metadata.get('size_chars', 0)} characters")
            print(f"🆔 Version: {updated_tale.metadata.get('version', 1)} (incremented)")
            print(f"🔄 Updated: {updated_tale.metadata.get('updated', '')}")
            print("")
            
            # Show diff summary
            old_size = len(existing_tale.content)
            new_size = len(new_content)
            size_diff = new_size - old_size
            
            if size_diff > 0:
                print(f"📈 Content expanded by {size_diff} characters")
            elif size_diff < 0:
                print(f"📉 Content reduced by {abs(size_diff)} characters")
            else:
                print("📝 Content length unchanged (modifications made)")
            
            print("")
            print(f"💡 Use 'clay_load_tale {updated_tale.name}' to view updated content")
    
    except Exception as e:
        if args.json:
            error_result = {
                "status": "error",
                "error": str(e),
                "message": f"Failed to update tale: {str(e)}"
            }
            print(json.dumps(error_result, indent=2, ensure_ascii=False))
        else:
            print(f"❌ ERROR: Failed to update tale: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
