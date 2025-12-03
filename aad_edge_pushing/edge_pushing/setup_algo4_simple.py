"""
Setup script for compiling algo4_cython_simple.pyx

Usage:
    python setup_algo4_simple.py build_ext --inplace
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extra_compile_args = [
    '-O3',
    '-march=native',
    '-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION',
]

extensions = [
    Extension(
        "algo4_cython_simple",
        sources=["algo4_cython_simple.pyx"],
        extra_compile_args=extra_compile_args,
        include_dirs=[np.get_include()],
    )
]

setup(
    name="algo4_cython_simple",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            'language_level': '3',
            'boundscheck': False,
            'wraparound': False,
            'cdivision': True,
            'initializedcheck': False,
        },
    ),
    zip_safe=False,
)
