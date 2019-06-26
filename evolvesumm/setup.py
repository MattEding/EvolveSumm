from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy


ext = Extension("util",
                sources=["utils.pyx"])

setup(name="util",
      ext_modules=cythonize(ext),
      include_dirs=[numpy.get_include()])
