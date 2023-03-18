from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CppExtension
setup(name='torchTT',
version='2.0',
description='Tensor-Train decomposition in pytorch',
url='https://github.com/ion-g-ion/torchTT',
author='Ion Gabriel Ion',
author_email='ion.ion.gabriel@gmail.com',
license='MIT',
packages=['torchtt'],
install_requires=['numpy>=1.18','torch>=1.7','opt_einsum'],
ext_modules=[
    CppExtension('torchttcpp', ['cpp/cpp_ext.cpp'], extra_compile_args=['-g', '-O2', '-w']),
],
cmdclass={
    'build_ext': BuildExtension
},
test_suite='tests',
zip_safe=False,
classifiers=[
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
]) 


