from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CppExtension
setup(name='torchTT',
version='1.0',
description='Tensor-Train decomposition in pytorch',
url='https://github.com/ion-g-ion/torchTT',
author='Ion Gabriel Ion',
author_email='ion.ion.gabriel@gmail.com',
license='MIT',
packages=['torchtt'],
install_requires=['numpy>=1.18','torch>=1.7','opt_einsum'],
ext_modules=[
    CppExtension('torchttcpp', ['cpp/cpp_ext.cpp']),
],
cmdclass={
    'build_ext': BuildExtension
},
test_suite='tests',
zip_safe=False) 

