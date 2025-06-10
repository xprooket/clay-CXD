#!/usr/bin/env python3
"""
Debug script to check which Python is being used
"""
import sys
import os

# FORCE UTF-8 I/O - CRITICAL para Windows
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

def main():
    print("=== PYTHON DEBUG INFO ===")
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")
    print(f"Python path: {sys.path[0:3]}")  # First 3 paths
    print()
    
    # Check for numpy
    try:
        import numpy
        print(f"[OK] numpy found: {numpy.__version__} at {numpy.__file__}")
    except ImportError as e:
        print(f"[ERROR] numpy not found: {e}")
    
    # Check working directory
    print(f"Working directory: {os.getcwd()}")
    print()
    
    # Check environment
    python_path = os.environ.get('PYTHONPATH', 'Not set')
    print(f"PYTHONPATH: {python_path}")

if __name__ == "__main__":
    main()
