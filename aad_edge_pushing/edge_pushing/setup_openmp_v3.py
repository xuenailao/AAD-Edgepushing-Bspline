"""
Setup script for compiling algo4_openmp_v3.pyx

真正的 nogil OpenMP 并行化！

Usage:
    python setup_openmp_v3.py build_ext --inplace
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "algo4_openmp_v3",
        ["algo4_openmp_v3.pyx"],
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
    name="algo4_openmp_v3",
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
