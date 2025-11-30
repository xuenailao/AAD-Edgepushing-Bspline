#!/usr/bin/env python
"""
Setup script for compiling algo4_cython.pyx (Stage 2B Phase 1)

Compiles the Cythonized version of Algorithm 4 main loop.
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import sys

# Compiler flags for optimization
extra_compile_args = [
    '-O3',              # Maximum optimization
    '-march=native',    # CPU-specific instructions (AVX, SSE, etc.)
    '-std=c++11',       # C++11 standard
    '-fopenmp',         # OpenMP support (for Stage 2B Phase 2)
]

extra_link_args = [
    '-fopenmp',         # Link OpenMP library
]

# Extensions to compile
extensions = [
    Extension(
        name="aad_edge_pushing.edge_pushing.algo4_cython_simple",
        sources=["algo4_cython_simple.pyx"],
        language="c++",
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        include_dirs=[np.get_include()],
    ),
]

setup(
    name="algo4_cython",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            'language_level': '3',
            'boundscheck': False,
            'wraparound': False,
            'cdivision': True,
            'initializedcheck': False,
        },
        annotate=True,  # Generate HTML annotation file for optimization analysis
    ),
    include_dirs=[np.get_include()],
)

print("\n" + "="*70)
print("COMPILATION COMPLETE")
print("="*70)
print("\nGenerated files:")
print("  - algo4_cython.c (C++ source)")
print("  - algo4_cython*.so (shared library)")
print("  - algo4_cython.html (optimization report)")
print("\nTo use:")
print("  from aad_edge_pushing.edge_pushing.algo4_cython import algo4_cython")
print("  H = algo4_cython(output, inputs)")
