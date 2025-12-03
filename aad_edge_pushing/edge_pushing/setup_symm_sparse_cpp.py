"""
Setup script for compiling symm_sparse_adjlist_cpp.pyx

Usage:
    python setup_symm_sparse_cpp.py build_ext --inplace
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extra_compile_args = [
    '-O3',
    '-march=native',
    '-std=c++11',
    '-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION',
]

extensions = [
    Extension(
        "symm_sparse_adjlist_cpp",
        sources=["symm_sparse_adjlist_cpp.pyx"],
        extra_compile_args=extra_compile_args,
        include_dirs=[np.get_include()],
        language="c++",
    )
]

setup(
    name="symm_sparse_adjlist_cpp",
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
