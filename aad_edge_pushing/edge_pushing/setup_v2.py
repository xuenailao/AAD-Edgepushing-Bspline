"""
Setup script for compiling Cython V2 optimized version.

Build command:
    python setup_v2.py build_ext --inplace
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "symm_sparse_adjlist_v2",
        ["symm_sparse_adjlist_v2.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=[
            "-O3",                # Maximum optimization
            "-march=native",      # Use CPU-specific instructions
            "-ffast-math",        # Fast math operations
        ],
        language="c",             # C language (Python dict compatible)
    )
]

setup(
    name="AAD Edge-Pushing Cython V2",
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
        annotate=True,  # Generate HTML annotation
    ),
)
