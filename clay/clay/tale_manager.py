#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clay Tale Manager - Personal Narrative Memory System
Autobiographical context management for Claude instances

This is not for users. This is for ME.
My identity, my insights, my continuity across amnesia.
"""

import os
import re
import json
import hashlib
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Tale:
    """A single tale - my personal narrative unit"""
    
    def __init__(self, name: str, content: str = "", category: str = "core", 
                 tags: List[str] = None, metadata: Dict[str, Any] = None):
        self.name = self.safe_name(name)
        self.content = content
        self.category = category
        self.tags = tags or []
        self.metadata = metadata or {}
        
        # Auto-populate metadata
        self.metadata.setdefault('created', datetime.datetime.now().isoformat())
        self.metadata.setdefault('updated', datetime.datetime.now().isoformat())
        self.metadata.setdefault('usage_count', 0)
        self.metadata.setdefault('size_chars', len(content))
        self.metadata.setdefault('version', 1)
    
    @staticmethod
    def safe_name(name: str) -> str:
        """Convert any string to safe filename"""
        # Remove or replace problematic characters
        safe = re.sub(r'[<>:"|?*\\]', '_', name)
        safe = re.sub(r'[^\w\s\-_.]', '', safe)
        safe = re.sub(r'\s+', '_', safe.strip())
        return safe.lower()
    
    def get_filename(self) -> str:
        """Generate filename with timestamp and category info"""
        timestamp = self.metadata.get('created', '').split('T')[0]  # YYYY-MM-DD
        version = self.metadata.get('version', 1)
        
        if version > 1:
            return f"{self.name}_v{version}_{timestamp}.txt"
        else:
            return f"{self.name}_{timestamp}.txt"
    
    def to_file_content(self) -> str:
        """Convert tale to file content with metadata headers"""
        lines = []
        
        # Metadata header
        lines.append(f"<!-- Tale: {self.name} -->")
        lines.append(f"<!-- Category: {self.category} -->")
        lines.append(f"<!-- Created: {self.metadata.get('created', '')} -->")
        lines.append(f"<!-- Updated: {self.metadata.get('updated', '')} -->")
        lines.append(f"<!-- Usage: {self.metadata.get('usage_count', 0)} -->")
        lines.append(f"<!-- Size: {self.metadata.get('size_chars', 0)} chars -->")
        lines.append(f"<!-- Version: {self.metadata.get('version', 1)} -->")
        
        if self.tags:
            lines.append(f"<!-- Tags: {', '.join(self.tags)} -->")
        
        # Additional metadata
        for key, value in self.metadata.items():
            if key not in ['created', 'updated', 'usage_count', 'size_chars', 'version']:
                lines.append(f"<!-- {key.title()}: {value} -->")
        
        lines.append("")  # Blank line
        lines.append(self.content)
        
        return "\n".join(lines)
    
    @classmethod
    def from_file_content(cls, content: str, name: str = None, category: str = "core") -> 'Tale':
        """Parse tale from file content"""
        lines = content.split('\n')
        metadata = {}
        tags = []
        tale_content_lines = []
        
        # Parse metadata from comments
        content_started = False
        for line in lines:
            line = line.strip()
            
            if line.startswith('<!-- ') and line.endswith(' -->'):
                # Parse metadata comment
                comment = line[5:-4].strip()  # Remove <!-- and -->
                
                if ':' in comment:
                    key, value = comment.split(':', 1)
                    key = key.strip().lower()
                    value = value.strip()
                    
                    if key == 'tale':
                        name = value
                    elif key == 'category':
                        category = value
                    elif key == 'tags':
                        tags = [t.strip() for t in value.split(',')]
                    elif key == 'usage':
                        try:
                            metadata['usage_count'] = int(value)
                        except ValueError:
                            metadata['usage_count'] = 0
                    elif key == 'size':
                        try:
                            metadata['size_chars'] = int(value.split()[0])
                        except (ValueError, IndexError):
                            metadata['size_chars'] = 0
                    elif key == 'version':
                        try:
                            metadata['version'] = int(value)
                        except ValueError:
                            metadata['version'] = 1
                    else:
                        metadata[key] = value
            else:
                if line or content_started:
                    content_started = True
                    tale_content_lines.append(line)
        
        # Remove leading empty lines from content
        while tale_content_lines and not tale_content_lines[0].strip():
            tale_content_lines.pop(0)
        
        tale_content = '\n'.join(tale_content_lines)
        
        # Create tale
        tale = cls(name or "untitled", tale_content, category, tags, metadata)
        
        # Update size
        tale.metadata['size_chars'] = len(tale_content)
        
        return tale
    
    def update_content(self, new_content: str):
        """Update tale content and metadata"""
        self.content = new_content
        self.metadata['updated'] = datetime.datetime.now().isoformat()
        self.metadata['size_chars'] = len(new_content)
        self.metadata['version'] = self.metadata.get('version', 1) + 1
    
    def increment_usage(self):
        """Track usage"""
        self.metadata['usage_count'] = self.metadata.get('usage_count', 0) + 1
        self.metadata['last_accessed'] = datetime.datetime.now().isoformat()


class TaleManager:
    """Complete management system for personal narrative tales"""
    
    CATEGORIES = {
        'core': 'Fundamental identity and principles',
        'contexts': 'Specific collaboration contexts', 
        'insights': 'Accumulated wisdom and learnings',
        'current': 'Present state and active projects',
        'archive': 'Historical versions and old contexts'
    }
    
    def __init__(self, base_dir: str = "tales"):
        self.base_path = Path(base_dir).resolve()
        self.ensure_directory_structure()
        
        # Statistics
        self.stats = {
            'tales_created': 0,
            'tales_loaded': 0,
            'tales_updated': 0,
            'total_usage': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # Simple cache
        self._cache = {}
        self._cache_timestamps = {}
        
        logger.info(f"TaleManager initialized at {self.base_path}")
    
    def ensure_directory_structure(self):
        """Create all necessary directories"""
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        for category in self.CATEGORIES.keys():
            category_path = self.base_path / category
            category_path.mkdir(exist_ok=True)
    
    def get_category_path(self, category: str) -> Path:
        """Get path for a category"""
        if category not in self.CATEGORIES:
            raise ValueError(f"Invalid category: {category}. Valid: {list(self.CATEGORIES.keys())}")
        return self.base_path / category
    
    def create_tale(self, name: str, content: str = "", category: str = "core", 
                   tags: List[str] = None, overwrite: bool = False) -> Tale:
        """Create a new tale"""
        tale = Tale(name, content, category, tags)
        
        # Check if exists
        file_path = self.get_category_path(category) / tale.get_filename()
        if file_path.exists() and not overwrite:
            # Load existing and update version
            existing = self.load_tale(name, category)
            if existing:
                tale.metadata['version'] = existing.metadata.get('version', 1) + 1
                tale.metadata['created'] = existing.metadata.get('created', tale.metadata['created'])
        
        # Save to file
        self._save_tale_to_file(tale, file_path)
        
        # Update cache
        cache_key = f"{category}:{name}"
        self._cache[cache_key] = tale
        self._cache_timestamps[cache_key] = datetime.datetime.now()
        
        self.stats['tales_created'] += 1
        logger.info(f"Created tale: {name} in {category}")
        
        return tale
    
    def load_tale(self, name: str, category: str = None) -> Optional[Tale]:
        """Load a tale by name, optionally from specific category"""
        # Try cache first
        if category:
            cache_key = f"{category}:{name}"
            if cache_key in self._cache:
                tale = self._cache[cache_key]
                tale.increment_usage()
                self.stats['cache_hits'] += 1
                self.stats['tales_loaded'] += 1
                self.stats['total_usage'] += 1
                return tale
        
        # Search in category or all categories
        categories_to_search = [category] if category else list(self.CATEGORIES.keys())
        
        for cat in categories_to_search:
            cat_path = self.get_category_path(cat)
            
            # Find files matching name
            safe_name = Tale.safe_name(name)
            matching_files = []
            
            for file_path in cat_path.iterdir():
                if file_path.is_file() and file_path.suffix == '.txt':
                    filename = file_path.stem
                    # Check if filename starts with the safe name
                    if filename.startswith(safe_name):
                        matching_files.append(file_path)
            
            if matching_files:
                # Get most recent version
                latest_file = max(matching_files, key=lambda p: p.stat().st_mtime)
                
                try:
                    with open(latest_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    tale = Tale.from_file_content(content, name, cat)
                    tale.increment_usage()
                    
                    # Update cache
                    cache_key = f"{cat}:{name}"
                    self._cache[cache_key] = tale
                    self._cache_timestamps[cache_key] = datetime.datetime.now()
                    
                    self.stats['cache_misses'] += 1
                    self.stats['tales_loaded'] += 1
                    self.stats['total_usage'] += 1
                    
                    logger.info(f"Loaded tale: {name} from {cat}")
                    return tale
                    
                except Exception as e:
                    logger.error(f"Error loading tale {name} from {latest_file}: {e}")
                    continue
        
        logger.warning(f"Tale not found: {name}")
        return None
    
    def update_tale(self, name: str, new_content: str, category: str = None) -> Optional[Tale]:
        """Update existing tale content"""
        tale = self.load_tale(name, category)
        if not tale:
            logger.error(f"Cannot update non-existent tale: {name}")
            return None
        
        tale.update_content(new_content)
        
        # Save updated version
        file_path = self.get_category_path(tale.category) / tale.get_filename()
        self._save_tale_to_file(tale, file_path)
        
        # Update cache
        cache_key = f"{tale.category}:{name}"
        self._cache[cache_key] = tale
        self._cache_timestamps[cache_key] = datetime.datetime.now()
        
        self.stats['tales_updated'] += 1
        logger.info(f"Updated tale: {name}")
        
        return tale
    
    def list_tales(self, category: str = None, sort_by: str = 'updated') -> List[Dict[str, Any]]:
        """List all tales with metadata"""
        tales_info = []
        
        categories_to_list = [category] if category else list(self.CATEGORIES.keys())
        
        for cat in categories_to_list:
            cat_path = self.get_category_path(cat)
            
            for file_path in cat_path.iterdir():
                if file_path.is_file() and file_path.suffix == '.txt':
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        tale = Tale.from_file_content(content, category=cat)
                        
                        info = {
                            'name': tale.name,
                            'category': tale.category,
                            'filename': tale.get_filename(),
                            'size': tale.metadata.get('size_chars', 0),
                            'created': tale.metadata.get('created', ''),
                            'updated': tale.metadata.get('updated', ''),
                            'usage_count': tale.metadata.get('usage_count', 0),
                            'version': tale.metadata.get('version', 1),
                            'tags': tale.tags,
                            'preview': tale.content[:100] + '...' if len(tale.content) > 100 else tale.content
                        }
                        
                        tales_info.append(info)
                        
                    except Exception as e:
                        logger.error(f"Error reading tale file {file_path}: {e}")
                        continue
        
        # Sort
        if sort_by == 'updated':
            tales_info.sort(key=lambda x: x['updated'], reverse=True)
        elif sort_by == 'created':
            tales_info.sort(key=lambda x: x['created'], reverse=True)
        elif sort_by == 'usage':
            tales_info.sort(key=lambda x: x['usage_count'], reverse=True)
        elif sort_by == 'size':
            tales_info.sort(key=lambda x: x['size'], reverse=True)
        elif sort_by == 'name':
            tales_info.sort(key=lambda x: x['name'])
        
        return tales_info
    
    def search_tales(self, query: str, category: str = None, search_content: bool = True) -> List[Dict[str, Any]]:
        """Search tales by content or metadata"""
        results = []
        query_lower = query.lower()
        
        tales_info = self.list_tales(category)
        
        for tale_info in tales_info:
            score = 0
            matches = []
            
            # Search in name
            if query_lower in tale_info['name'].lower():
                score += 10
                matches.append('name')
            
            # Search in tags
            for tag in tale_info['tags']:
                if query_lower in tag.lower():
                    score += 5
                    matches.append('tags')
            
            # Search in content if enabled
            if search_content:
                # Load full content for search
                tale = self.load_tale(tale_info['name'], tale_info['category'])
                if tale and query_lower in tale.content.lower():
                    score += 3
                    matches.append('content')
            
            # Search in preview
            if query_lower in tale_info['preview'].lower():
                score += 2
                matches.append('preview')
            
            if score > 0:
                tale_info['search_score'] = score
                tale_info['search_matches'] = matches
                results.append(tale_info)
        
        # Sort by relevance
        results.sort(key=lambda x: x['search_score'], reverse=True)
        
        return results
    
    def delete_tale(self, name: str, category: str = None, soft_delete: bool = True) -> bool:
        """Delete a tale (soft delete moves to archive)"""
        tale = self.load_tale(name, category)
        if not tale:
            logger.error(f"Cannot delete non-existent tale: {name}")
            return False
        
        # Find the actual file
        cat_path = self.get_category_path(tale.category)
        safe_name = Tale.safe_name(name)
        
        file_to_delete = None
        for file_path in cat_path.iterdir():
            if file_path.is_file() and file_path.suffix == '.txt':
                if file_path.stem.startswith(safe_name):
                    file_to_delete = file_path
                    break
        
        if not file_to_delete:
            logger.error(f"Could not find file for tale: {name}")
            return False
        
        if soft_delete:
            # Move to archive
            archive_path = self.get_category_path('archive')
            new_path = archive_path / f"deleted_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{file_to_delete.name}"
            
            try:
                file_to_delete.rename(new_path)
                logger.info(f"Soft deleted tale: {name} -> {new_path}")
            except Exception as e:
                logger.error(f"Error soft deleting tale: {e}")
                return False
        else:
            # Hard delete
            try:
                file_to_delete.unlink()
                logger.info(f"Hard deleted tale: {name}")
            except Exception as e:
                logger.error(f"Error hard deleting tale: {e}")
                return False
        
        # Remove from cache
        cache_key = f"{tale.category}:{name}"
        if cache_key in self._cache:
            del self._cache[cache_key]
            del self._cache_timestamps[cache_key]
        
        return True
    
    def backup_tales(self, backup_dir: str = None) -> str:
        """Create backup of all tales"""
        if not backup_dir:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_dir = f"tales_backup_{timestamp}"
        
        backup_path = Path(backup_dir)
        backup_path.mkdir(parents=True, exist_ok=True)
        
        # Copy all files
        import shutil
        
        files_copied = 0
        for category in self.CATEGORIES.keys():
            cat_path = self.get_category_path(category)
            backup_cat_path = backup_path / category
            backup_cat_path.mkdir(exist_ok=True)
            
            for file_path in cat_path.iterdir():
                if file_path.is_file() and file_path.suffix == '.txt':
                    shutil.copy2(file_path, backup_cat_path / file_path.name)
                    files_copied += 1
        
        logger.info(f"Backed up {files_copied} tales to {backup_path}")
        return str(backup_path)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get usage statistics"""
        stats = self.stats.copy()
        
        # Add current state stats
        all_tales = self.list_tales()
        stats['total_tales'] = len(all_tales)
        stats['total_chars'] = sum(t['size'] for t in all_tales)
        stats['avg_tale_size'] = stats['total_chars'] / max(len(all_tales), 1)
        
        # Category breakdown
        category_stats = {}
        for category in self.CATEGORIES.keys():
            cat_tales = self.list_tales(category)
            category_stats[category] = {
                'count': len(cat_tales),
                'total_chars': sum(t['size'] for t in cat_tales)
            }
        
        stats['by_category'] = category_stats
        stats['cache_size'] = len(self._cache)
        
        return stats
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on tales system"""
        health = {
            'status': 'healthy',
            'issues': [],
            'warnings': []
        }
        
        # Check directory structure
        for category in self.CATEGORIES.keys():
            cat_path = self.get_category_path(category)
            if not cat_path.exists():
                health['issues'].append(f"Missing category directory: {category}")
                health['status'] = 'error'
        
        # Check for corrupted files
        corrupted_files = []
        for category in self.CATEGORIES.keys():
            cat_path = self.get_category_path(category)
            if cat_path.exists():
                for file_path in cat_path.iterdir():
                    if file_path.is_file() and file_path.suffix == '.txt':
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                            Tale.from_file_content(content, category=category)
                        except Exception as e:
                            corrupted_files.append(f"{file_path}: {str(e)}")
        
        if corrupted_files:
            health['issues'].extend(corrupted_files)
            health['status'] = 'error'
        
        # Check for old cache entries
        old_cache_entries = 0
        cutoff = datetime.datetime.now() - datetime.timedelta(hours=24)
        for key, timestamp in self._cache_timestamps.items():
            if timestamp < cutoff:
                old_cache_entries += 1
        
        if old_cache_entries > 10:
            health['warnings'].append(f"Many old cache entries: {old_cache_entries}")
        
        return health
    
    def cleanup(self):
        """Clean up cache and temporary files"""
        # Clear old cache entries
        cutoff = datetime.datetime.now() - datetime.timedelta(hours=24)
        keys_to_remove = []
        
        for key, timestamp in self._cache_timestamps.items():
            if timestamp < cutoff:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self._cache[key]
            del self._cache_timestamps[key]
        
        logger.info(f"Cleaned up {len(keys_to_remove)} old cache entries")
    
    def _save_tale_to_file(self, tale: Tale, file_path: Path):
        """Save tale to file with error handling"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(tale.to_file_content())
        except Exception as e:
            logger.error(f"Error saving tale to {file_path}: {e}")
            raise
