#!/usr/bin/env python
"""
Setup script for batch parallel Hessian computation.

This compiles the Cython module with OpenMP support for coarse-grained parallelization.

Usage:
    python setup_batch_parallel.py build_ext --inplace
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        name="algo4_batch_parallel",
        sources=["algo4_batch_parallel.pyx"],
        include_dirs=[np.get_include(), "."],
        extra_compile_args=[
            "-O3",              # Maximum optimization
            "-march=native",    # Use CPU-specific optimizations
            "-std=c++11",       # C++11 standard
            "-fopenmp",         # Enable OpenMP
            "-ffast-math",      # Fast math operations
            "-funroll-loops",   # Loop unrolling
        ],
        extra_link_args=["-fopenmp"],
        language="c++",
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    )
]

setup(
    name="algo4_batch_parallel",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
            "embedsignature": True,
        },
        annotate=False,  # Set to True to generate HTML annotation
    ),
    zip_safe=False,
)

if __name__ == "__main__":
    print("=" * 80)
    print("Compiling Batch Parallel Hessian Module")
    print("=" * 80)
    print()
    print("This module implements coarse-grained parallelization:")
    print("  - Parallelizes across multiple independent Hessian computations")
    print("  - No GIL contention (each thread works independently)")
    print("  - Expected 20-30× speedup on 32-core systems")
    print()
