from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy


ext = Extension(
    "util",
    sources=["utils.pyx"],
    # "cytest",
    # sources=["cytest.pyx"],
    language="c++",
    extra_compile_args=["-stdlib=libc++"],
    extra_link_args= ["-stdlib=libc++"],
)

setup(
    name="util",
    # name="cytest",
    ext_modules=cythonize(ext),
    include_dirs=[numpy.get_include()],
)
