"""
Setup script for compiling algo4_openmp_safe.pyx with OpenMP support.

This compiles the OpenMP-parallelized version of Algorithm 4 for Stage 2B.

Usage:
    python setup_algo4_openmp.py build_ext --inplace
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

# Compiler and linker flags
extra_compile_args = [
    '-O3',                  # Maximum optimization
    '-march=native',        # CPU-specific optimizations
    '-std=c++11',          # C++11 for STL containers
    '-fopenmp',            # Enable OpenMP
    '-ffast-math',         # Fast floating-point math
    '-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION',
]

extra_link_args = [
    '-fopenmp',            # Link OpenMP library
]

extensions = [
    Extension(
        "algo4_openmp_safe",
        sources=["algo4_openmp_safe.pyx"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        include_dirs=[np.get_include()],
        language="c++",
    )
]

setup(
    name="algo4_openmp_safe",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            'language_level': '3',
            'boundscheck': False,
            'wraparound': False,
            'cdivision': True,
            'initializedcheck': False,
            'embedsignature': True,
        },
        annotate=True,  # Generate HTML annotation for optimization inspection
    ),
    zip_safe=False,
)
