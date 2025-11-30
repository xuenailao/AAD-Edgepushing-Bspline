"""
Cython build script for symm_sparse_adjlist optimization.

Phase 1: Cython + Python dict/set
Expected speedup: 12× (verified by 三今: 575s → 47s)

Usage:
    python setup_cython.py build_ext --inplace
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "symm_sparse_adjlist",
        ["symm_sparse_adjlist.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3", "-march=native", "-ffast-math"],
        language="c",
    )
]

setup(
    name="symm_sparse_adjlist",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            'language_level': 3,
            'boundscheck': False,
            'wraparound': False,
            'cdivision': True,
            'initializedcheck': False,
        },
        annotate=True,  # Generate .html file to inspect Python interactions
    )
)
