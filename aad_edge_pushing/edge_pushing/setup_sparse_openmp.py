"""
Setup script for compiling algo4_sparse_openmp.pyx

Sparse + OpenMP parallelization for maximum performance!

Combines:
- Sparse tracking (60x speedup)
- OpenMP parallelization (2-4x additional speedup)

Usage:
    python setup_sparse_openmp.py build_ext --inplace
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "algo4_sparse_openmp",
        ["algo4_sparse_openmp.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=[
            "-O3",
            "-march=native",
            "-ffast-math",
            "-fopenmp",  # OpenMP support
        ],
        extra_link_args=["-fopenmp"],  # Link OpenMP library
        language="c++",
    )
]

setup(
    name="algo4_sparse_openmp",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            'language_level': 3,
            'boundscheck': False,
            'wraparound': False,
            'cdivision': True,
            'initializedcheck': False,
        },
        annotate=True,  # Generate .html annotation file
    ),
)
