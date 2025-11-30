"""
Setup script for compiling pure C++ OpenMP parallelized Algorithm 4.

This compiles:
1. C++ implementation (algo4_cpp_parallel.cpp)
2. Cython wrapper (algo4_cpp_wrapper.pyx)

Expected performance: 4-16× speedup over sequential version

Usage:
    python setup_algo4_cpp.py build_ext --inplace
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

# Compiler and linker flags for maximum performance
extra_compile_args = [
    '-O3',                  # Maximum optimization
    '-march=native',        # CPU-specific optimizations (AVX, SSE)
    '-std=c++11',          # C++11 for STL containers
    '-fopenmp',            # Enable OpenMP
    '-ffast-math',         # Fast floating-point math
    '-funroll-loops',      # Loop unrolling
    '-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION',
]

extra_link_args = [
    '-fopenmp',            # Link OpenMP library
]

extensions = [
    Extension(
        "algo4_cpp_wrapper",
        sources=[
            "algo4_cpp_wrapper.pyx",
            "algo4_cpp_parallel.cpp",  # Compile C++ implementation
        ],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        include_dirs=[
            np.get_include(),
            ".",  # For algo4_cpp_parallel.hpp
        ],
        language="c++",
    )
]

setup(
    name="algo4_cpp_wrapper",
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
        annotate=True,  # Generate HTML annotation
    ),
    zip_safe=False,
)
