from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
import subprocess

try:
    prefix = subprocess.check_output(['brew', '--prefix', 'tbb']).decode().strip()
    include_dirs = [f'{prefix}/include']
    library_dirs = [f'{prefix}/lib']
except:
    include_dirs = []
    library_dirs = []

ext_modules = [
    Pybind11Extension(
        "image_processor",
        ["image_proc.cpp"],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=["tbb"],
        extra_compile_args=['-std=c++17', '-O3'],
    ),
]

setup(
    name="image_processor",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)