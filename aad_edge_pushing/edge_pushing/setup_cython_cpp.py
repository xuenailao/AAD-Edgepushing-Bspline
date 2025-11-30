"""
Setup script for compiling Cython C++ optimized version.

Build command:
    python setup_cython_cpp.py build_ext --inplace

This compiles symm_sparse_adjlist_cpp.pyx with C++ STL containers.
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "symm_sparse_adjlist_cpp",
        ["symm_sparse_adjlist_cpp.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=[
            "-O3",                # Maximum optimization
            "-march=native",      # Use CPU-specific instructions (AVX, SSE, etc.)
            "-ffast-math",        # Fast math operations
            "-std=c++11",         # C++11 standard for unordered_map/set
            "-fopenmp",           # Enable OpenMP (for future parallel version)
        ],
        extra_link_args=[
            "-fopenmp",           # Link OpenMP library
        ],
        language="c++",           # ← Key: Compile as C++
    )
]

setup(
    name="AAD Edge-Pushing C++ Optimized",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            'language_level': 3,
            'boundscheck': False,
            'wraparound': False,
            'cdivision': True,
            'initializedcheck': False,
            'nonecheck': False,
            'embedsignature': True,
        },
        annotate=True,  # Generate HTML annotation file for optimization analysis
    ),
)
