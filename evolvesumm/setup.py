from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy


ext = Extension("inter",
                sources=["utils.pyx"])

setup(name="inter",
      ext_modules=cythonize(ext),
      include_dirs=[numpy.get_include()])
