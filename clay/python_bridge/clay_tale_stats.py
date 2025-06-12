#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clay Tale Stats - Personal Narrative Analytics Tool
Monitor and maintain my autobiographical tale collection
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

def format_size(chars: int) -> str:
    """Format character count in human readable form"""
    if chars < 1000:
        return f"{chars} chars"
    elif chars < 1000000:
        return f"{chars/1000:.1f}k chars"
    else:
        return f"{chars/1000000:.1f}M chars"

def main():
    parser = argparse.ArgumentParser(description="Tale collection statistics and maintenance")
    parser.add_argument("--stats", "-s", action="store_true", help="Show detailed statistics")
    parser.add_argument("--health", action="store_true", help="Perform health check")
    parser.add_argument("--cleanup", action="store_true", help="Clean up cache and temporary files")
    parser.add_argument("--backup", "-b", help="Create backup of all tales")
    parser.add_argument("--json", action="store_true", help="Output JSON format")
    
    # Default to stats if no specific action
    if len(sys.argv) == 1:
        sys.argv.append("--stats")
    
    args = parser.parse_args()
    
    try:
        # Initialize TaleManager
        tale_manager = TaleManager()
        
        results = {}
        
        # Statistics
        if args.stats:
            stats = tale_manager.get_statistics()
            results['statistics'] = stats
            
            if not args.json:
                print("📊 TALE COLLECTION STATISTICS")
                print("=" * 50)
                print(f"📚 Total tales: {stats['total_tales']}")
                print(f"📝 Total content: {format_size(stats['total_chars'])}")
                print(f"📏 Average tale size: {format_size(int(stats['avg_tale_size']))}")
                print(f"📊 Total usage: {stats['total_usage']} loads")
                print("")
                
                print("📂 BY CATEGORY:")
                for category, cat_stats in stats['by_category'].items():
                    percentage = (cat_stats['count'] / max(stats['total_tales'], 1)) * 100
                    print(f"   {category:10s}: {cat_stats['count']:3d} tales ({percentage:4.1f}%) | {format_size(cat_stats['total_chars'])}")
                print("")
                
                print("🔧 SYSTEM PERFORMANCE:")
                print(f"   Cache size: {stats['cache_size']} entries")
                print(f"   Cache hits: {stats['cache_hits']}")
                print(f"   Cache misses: {stats['cache_misses']}")
                
                if stats['cache_hits'] + stats['cache_misses'] > 0:
                    hit_rate = stats['cache_hits'] / (stats['cache_hits'] + stats['cache_misses']) * 100
                    print(f"   Cache hit rate: {hit_rate:.1f}%")
                
                print(f"   Tales created: {stats['tales_created']}")
                print(f"   Tales loaded: {stats['tales_loaded']}")
                print(f"   Tales updated: {stats['tales_updated']}")
                print("")
        
        # Health check
        if args.health:
            health = tale_manager.health_check()
            results['health'] = health
            
            if not args.json:
                print("🏥 SYSTEM HEALTH CHECK")
                print("=" * 50)
                
                if health['status'] == 'healthy':
                    print("✅ System status: HEALTHY")
                elif health['status'] == 'error':
                    print("❌ System status: ERROR")
                else:
                    print("⚠️  System status: WARNING")
                
                print("")
                
                if health['issues']:
                    print("🚨 ISSUES FOUND:")
                    for issue in health['issues']:
                        print(f"   ❌ {issue}")
                    print("")
                
                if health['warnings']:
                    print("⚠️  WARNINGS:")
                    for warning in health['warnings']:
                        print(f"   ⚠️  {warning}")
                    print("")
                
                if not health['issues'] and not health['warnings']:
                    print("🎉 No issues or warnings found!")
                    print("")
        
        # Cleanup
        if args.cleanup:
            if not args.json:
                print("🧹 CLEANING UP...")
            
            tale_manager.cleanup()
            results['cleanup'] = {"status": "completed"}
            
            if not args.json:
                print("✅ Cleanup completed")
                print("")
        
        # Backup
        if args.backup:
            if not args.json:
                print("💾 CREATING BACKUP...")
            
            backup_path = tale_manager.backup_tales(args.backup if args.backup != True else None)
            results['backup'] = {
                "status": "completed",
                "path": backup_path
            }
            
            if not args.json:
                print(f"✅ Backup created: {backup_path}")
                print("")
        
        # JSON output
        if args.json:
            print(json.dumps(results, indent=2, ensure_ascii=False))
        
        # Summary recommendations
        if not args.json and args.stats:
            print("💡 RECOMMENDATIONS:")
            
            stats = results.get('statistics', {})
            
            # Check for imbalanced categories
            by_category = stats.get('by_category', {})
            core_count = by_category.get('core', {}).get('count', 0)
            insights_count = by_category.get('insights', {}).get('count', 0)
            
            if core_count == 0:
                print("   📝 Consider creating core identity tales")
            elif core_count > 10:
                print("   📦 Consider archiving old core tales")
            
            if insights_count < core_count / 2:
                print("   💡 Consider adding more insight tales")
            
            # Cache performance
            cache_hits = stats.get('cache_hits', 0)
            cache_misses = stats.get('cache_misses', 0)
            
            if cache_hits + cache_misses > 0:
                hit_rate = cache_hits / (cache_hits + cache_misses)
                if hit_rate < 0.7:
                    print("   🚀 Cache hit rate is low - consider running cleanup")
            
            # Usage patterns
            total_usage = stats.get('total_usage', 0)
            total_tales = stats.get('total_tales', 1)
            avg_usage = total_usage / total_tales
            
            if avg_usage < 1:
                print("   📊 Many tales have low usage - consider cleanup")
            
            print("")
            print("💡 MAINTENANCE COMMANDS:")
            print("   clay_tale_stats --health     - Check system health")
            print("   clay_tale_stats --cleanup    - Clean up cache")
            print("   clay_tale_stats --backup     - Create backup")
            print("   clay_search_tales <query>    - Find specific content")
    
    except Exception as e:
        if args.json:
            error_result = {
                "status": "error",
                "error": str(e),
                "message": f"Stats operation failed: {str(e)}"
            }
            print(json.dumps(error_result, indent=2, ensure_ascii=False))
        else:
            print(f"❌ ERROR: Stats operation failed: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
