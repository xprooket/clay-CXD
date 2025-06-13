#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clay List Tales v2.0 - Personal Narrative Catalog Tool
Browse my collection of autobiographical tales

NEW STRUCTURE:
- claude/*     → Personal continuity
- projects/*   → Technical documentation  
- misc/*       → Everything else
"""

import sys
import os
import json
import argparse
from datetime import datetime, timedelta

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

def format_datetime(iso_string: str) -> str:
    """Format ISO datetime for human reading"""
    try:
        dt = datetime.fromisoformat(iso_string.replace('Z', '+00:00'))
        now = datetime.now()
        
        # Time ago calculation
        diff = now - dt.replace(tzinfo=None)
        
        if diff.days == 0:
            if diff.seconds < 3600:
                mins = diff.seconds // 60
                return f"{mins}m ago"
            else:
                hours = diff.seconds // 3600
                return f"{hours}h ago"
        elif diff.days == 1:
            return "yesterday"
        elif diff.days < 7:
            return f"{diff.days}d ago"
        else:
            return dt.strftime('%Y-%m-%d')
    except:
        return iso_string[:10]  # Just date part

def format_size(chars: int) -> str:
    """Format character count in human readable form"""
    if chars < 1000:
        return f"{chars} chars"
    elif chars < 1000000:
        return f"{chars/1000:.1f}k chars"
    else:
        return f"{chars/1000000:.1f}M chars"

def main():
    # Get valid categories dynamically
    valid_categories = get_valid_categories()
    
    parser = argparse.ArgumentParser(description="List personal tales")
    parser.add_argument("--category", "-c", 
                       help=f"Show only specific category. Valid: {', '.join(valid_categories[:5])}...")
    parser.add_argument("--sort", "-s", 
                       choices=['updated', 'created', 'usage', 'size', 'name'],
                       default='updated',
                       help="Sort order")
    parser.add_argument("--limit", "-l", type=int, help="Limit number of results")
    parser.add_argument("--json", action="store_true", help="Output JSON format")
    parser.add_argument("--stats", action="store_true", help="Show statistics")
    parser.add_argument("--recent", "-r", action="store_true", help="Show only recent tales (last 7 days)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed information")
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
    
    args = parser.parse_args()
    
    try:
        # Initialize TaleManager
        tale_manager = TaleManager()
        
        # Validate category if provided
        if args.category and args.category not in valid_categories:
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
        
        # Get all tales
        tales = tale_manager.list_tales(category=args.category, sort_by=args.sort)
        
        # Filter recent if requested
        if args.recent:
            cutoff = datetime.now() - timedelta(days=7)
            tales = [t for t in tales if datetime.fromisoformat(t['updated'].replace('Z', '+00:00')).replace(tzinfo=None) > cutoff]
        
        # Apply limit
        if args.limit:
            tales = tales[:args.limit]
        
        if args.json:
            # JSON output
            result = {
                "status": "success",
                "tales": tales,
                "total": len(tales),
                "category": args.category,
                "sort": args.sort,
                "structure_version": "2.0"
            }
            
            if args.stats:
                result["statistics"] = tale_manager.get_statistics()
            
            print(json.dumps(result, indent=2, ensure_ascii=False))
            
        else:
            # Human readable output
            if args.stats:
                # Show statistics first
                stats = tale_manager.get_statistics()
                print("📊 TALE STATISTICS")
                print("=" * 50)
                print(f"Total tales: {stats['total_tales']}")
                print(f"Total characters: {format_size(stats['total_chars'])}")
                print(f"Average tale size: {format_size(int(stats['avg_tale_size']))}")
                print(f"Total usage: {stats['total_usage']}")
                print("")
                
                print("📂 BY CATEGORY:")
                for category, cat_stats in stats['by_category'].items():
                    print(f"  {category}: {cat_stats['count']} tales, {format_size(cat_stats['total_chars'])}")
                print("")
                
                print("🔧 SYSTEM:")
                print(f"  Structure version: {stats.get('structure_version', '1.0')}")
                print(f"  Cache size: {stats['cache_size']}")
                print(f"  Cache hits: {stats['cache_hits']}")
                print(f"  Cache misses: {stats['cache_misses']}")
                print("")
            
            # Show tales list
            if not tales:
                filter_desc = f" in {args.category}" if args.category else ""
                recent_desc = " (recent)" if args.recent else ""
                print(f"📭 No tales found{filter_desc}{recent_desc}")
                return
            
            header = "📚 MY PERSONAL TALES"
            if args.category:
                header += f" - {args.category.upper()}"
            if args.recent:
                header += " (RECENT)"
            
            print(header)
            print("=" * 60)
            
            for i, tale in enumerate(tales, 1):
                # Icons for categories - updated for new structure
                category_icons = {
                    'claude/core': '🧠',
                    'claude/contexts': '🤝', 
                    'claude/insights': '💡',
                    'claude/current': '🎯',
                    'claude/archive': '📦',
                    'core': '🧠',  # Backward compatibility
                    'contexts': '🤝',
                    'insights': '💡', 
                    'current': '🎯',
                    'archive': '📦'
                }
                
                # Default icons for new structure
                if tale['category'].startswith('projects/'):
                    icon = '🔧'
                elif tale['category'].startswith('misc/'):
                    icon = '📄'
                else:
                    icon = category_icons.get(tale['category'], '📄')
                
                if args.verbose:
                    # Detailed format
                    print(f"{i:2d}. {icon} {tale['name']}")
                    print(f"    📂 {tale['category']} | 📏 {format_size(tale['size'])} | 🆔 v{tale['version']}")
                    print(f"    📅 Created: {format_datetime(tale['created'])} | 🔄 Updated: {format_datetime(tale['updated'])}")
                    print(f"    📊 Usage: {tale['usage_count']} times")
                    if tale['tags']:
                        print(f"    🏷️  Tags: {', '.join(tale['tags'])}")
                    if tale['preview']:
                        preview = tale['preview'].replace('\n', ' ')[:80]
                        print(f"    📝 \"{preview}\"")
                    print("")
                else:
                    # Compact format
                    name_display = tale['name'][:30].ljust(30)
                    size_display = format_size(tale['size']).rjust(8)
                    updated_display = format_datetime(tale['updated']).rjust(10)
                    usage_display = f"{tale['usage_count']}x".rjust(4)
                    
                    print(f"{i:2d}. {icon} {name_display} | {size_display} | {updated_display} | {usage_display}")
            
            print("")
            print(f"📋 Showing {len(tales)} tales")
            if args.limit and len(tales) == args.limit:
                print(f"💡 Use --limit 0 to see all tales")
            
            print("")
            print("💡 QUICK COMMANDS:")
            print("   clay_load_tale <n>     - Load a specific tale")
            print("   clay_search_tales <query> - Search tale content")
            print("   clay_create_tale <n>   - Create new tale")
            print("   clay_list_tales --list-categories - See all categories")
    
    except Exception as e:
        if args.json:
            error_result = {
                "status": "error",
                "error": str(e),
                "message": f"Failed to list tales: {str(e)}",
                "structure_version": "2.0"
            }
            print(json.dumps(error_result, indent=2, ensure_ascii=False))
        else:
            print(f"❌ ERROR: Failed to list tales: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
