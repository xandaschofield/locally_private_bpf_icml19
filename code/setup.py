#!/usr/bin/env python
import numpy as np
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import sys
import os

# if sys.platform == 'darwin':
#     os.environ["CC"] = "g++-5"
#     os.environ["CXX"] = "g++-5"

extensions = [
    Extension(
        "*",
        ["**/*.pyx"],
        extra_link_args=['-fopenmp', '-L/usr/lib64'],
        extra_compile_args=['-fopenmp', '-Wno-maybe-uninitialized'],
#        extra_compile_args=['-Wno-maybe-initialized']
    )
]

setup(
    cmdclass={'build_ext': build_ext},
    include_dirs=[np.get_include()],
    ext_modules=cythonize(extensions)
)
