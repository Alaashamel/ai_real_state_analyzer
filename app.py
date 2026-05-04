"""
Real Estate AI - Main Entry Point
Run: streamlit run app.py
"""

import streamlit as st
from pathlib import Path

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

# Import and run dashboard
from src.visualization.dashboard import main

if __name__ == "__main__":
    main()
