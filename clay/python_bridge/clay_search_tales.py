#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clay Search Tales - Personal Narrative Search Tool
Find my thoughts and memories within my tales
"""

import sys
import os
import json
import argparse
import re

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

def highlight_match(text: str, query: str, context_chars: int = 50) -> str:
    """Highlight search matches in text with context"""
    if not query:
        return text[:100] + "..." if len(text) > 100 else text
    
    # Find all matches (case insensitive)
    matches = []
    for match in re.finditer(re.escape(query), text, re.IGNORECASE):
        matches.append((match.start(), match.end()))
    
    if not matches:
        return text[:100] + "..." if len(text) > 100 else text
    
    # Create highlighted snippets
    snippets = []
    for start, end in matches[:3]:  # Show first 3 matches
        # Get context around match
        context_start = max(0, start - context_chars)
        context_end = min(len(text), end + context_chars)
        
        snippet = text[context_start:context_end]
        
        # Highlight the match within the snippet
        match_start = start - context_start
        match_end = end - context_start
        
        highlighted = (
            snippet[:match_start] + 
            f"**{snippet[match_start:match_end]}**" + 
            snippet[match_end:]
        )
        
        # Add ellipsis if truncated
        if context_start > 0:
            highlighted = "..." + highlighted
        if context_end < len(text):
            highlighted = highlighted + "..."
        
        snippets.append(highlighted)
    
    return " | ".join(snippets)

def format_datetime(iso_string: str) -> str:
    """Format ISO datetime for human reading"""
    try:
        from datetime import datetime
        dt = datetime.fromisoformat(iso_string.replace('Z', '+00:00'))
        return dt.strftime('%Y-%m-%d %H:%M')
    except:
        return iso_string[:16]

def main():
    parser = argparse.ArgumentParser(description="Search personal tales")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--category", "-c", 
                       choices=['core', 'contexts', 'insights', 'current', 'archive'],
                       help="Search only in specific category")
    parser.add_argument("--json", action="store_true", help="Output JSON format")
    parser.add_argument("--content", action="store_true", help="Search in content (default)")
    parser.add_argument("--no-content", action="store_true", help="Don't search in content")
    parser.add_argument("--name-only", action="store_true", help="Search only in names")
    parser.add_argument("--tags-only", action="store_true", help="Search only in tags")
    parser.add_argument("--limit", "-l", type=int, default=10, help="Limit number of results")
    parser.add_argument("--context", type=int, default=50, help="Characters of context around matches")
    
    args = parser.parse_args()
    
    try:
        # Initialize TaleManager
        tale_manager = TaleManager()
        
        # Determine search scope
        search_content = True
        if args.no_content or args.name_only or args.tags_only:
            search_content = False
        
        # Perform search
        results = tale_manager.search_tales(
            query=args.query,
            category=args.category,
            search_content=search_content
        )
        
        # Additional filtering for specific search types
        if args.name_only:
            results = [r for r in results if 'name' in r.get('search_matches', [])]
        elif args.tags_only:
            results = [r for r in results if 'tags' in r.get('search_matches', [])]
        
        # Apply limit
        results = results[:args.limit]
        
        if args.json:
            # JSON output
            result = {
                "status": "success",
                "query": args.query,
                "results": results,
                "total": len(results),
                "category": args.category,
                "search_content": search_content
            }
            print(json.dumps(result, indent=2, ensure_ascii=False))
        
        else:
            # Human readable output
            if not results:
                scope = f" in {args.category}" if args.category else ""
                print(f"🔍 No results found for '{args.query}'{scope}")
                
                # Suggest alternatives
                print("\n💡 Try:")
                print("  • Different keywords")
                print("  • Broader search terms")
                print("  • Remove category filter")
                if args.name_only or args.tags_only:
                    print("  • Include content search")
                return
            
            # Show results
            header = f"🔍 SEARCH RESULTS: '{args.query}'"
            if args.category:
                header += f" in {args.category}"
            
            print(header)
            print("=" * 60)
            
            for i, result in enumerate(results, 1):
                # Category icon
                category_icons = {
                    'core': '🧠',
                    'contexts': '🤝', 
                    'insights': '💡',
                    'current': '🎯',
                    'archive': '📦'
                }
                
                icon = category_icons.get(result['category'], '📄')
                score = result.get('search_score', 0)
                matches = result.get('search_matches', [])
                
                print(f"{i:2d}. {icon} {result['name']} (score: {score})")
                print(f"    📂 {result['category']} | 📏 {result['size']} chars | 📅 {format_datetime(result['updated'])}")
                print(f"    🎯 Matches: {', '.join(matches)}")
                
                # Show highlighted content
                if search_content and 'content' in matches:
                    # Load full content for highlighting
                    tale = tale_manager.load_tale(result['name'], result['category'])
                    if tale:
                        highlighted = highlight_match(tale.content, args.query, args.context)
                        print(f"    📝 \"{highlighted}\"")
                else:
                    # Show preview
                    print(f"    📝 \"{result['preview']}\"")
                
                print("")
            
            print(f"📋 Found {len(results)} results")
            if len(results) == args.limit:
                print(f"💡 Use --limit 0 to see all results")
            
            print("")
            print("💡 NEXT STEPS:")
            print("   clay_load_tale <n>     - Load a specific tale")
            print("   clay_update_tale <n>   - Edit a tale")
            print("   clay_list_tales        - Browse all tales")
    
    except Exception as e:
        if args.json:
            error_result = {
                "status": "error",
                "error": str(e),
                "message": f"Search failed: {str(e)}"
            }
            print(json.dumps(error_result, indent=2, ensure_ascii=False))
        else:
            print(f"❌ ERROR: Search failed: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
