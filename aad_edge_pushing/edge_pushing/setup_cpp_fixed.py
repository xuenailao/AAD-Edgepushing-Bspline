from setuptools import setup, Extension
from Cython.Build import cythonize

extensions = [
    Extension(
        "symm_sparse_adjlist_cpp_fixed",
        ["symm_sparse_adjlist_cpp_fixed.pyx"],
        extra_compile_args=["-O3", "-std=c++11"],
        language="c++",
    )
]

setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            'language_level': 3,
            'boundscheck': False,
            'wraparound': False,
        },
    ),
)
