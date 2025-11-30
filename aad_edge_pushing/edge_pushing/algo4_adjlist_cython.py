"""
Wrapper for Cythonized Algorithm 4 (Stage 2B Phase 1).

This module provides a drop-in replacement for algo4_adjlist
using the Cython-optimized version.
"""

from .algo4_cython_simple import algo4_cython_simple as algo4_adjlist

__all__ = ['algo4_adjlist']
