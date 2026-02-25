import sys

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup


# 定义扩展模块
ext_modules = [
    Pybind11Extension(
        "sampler_core",  # 编译后的模块名
        ["sampler_core.cpp"],  # 源文件
        # 传递给编译器的参数（开启 OpenMP）
        extra_compile_args=["-fopenmp", "-O3"]
        if sys.platform != "win32"
        else ["/openmp", "/O2"],
        # 传递给链接器的参数
        extra_link_args=["-lgomp"] if sys.platform != "win32" else [],
    ),
]

setup(
    name="sampler_core",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)

# python setup.py build_ext --inplace
