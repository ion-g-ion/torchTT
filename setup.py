from setuptools import setup
setup(name='torchTT',
version='1.0',
description='Tensor-Train decomposition in pytorch',
url='https://github.com/ion-g-ion/torchTT',
author='Ion Gabriel Ion',
author_email='ion.ion.gabriel@gmail.com',
license='MIT',
packages=['torchtt'],
install_requires=['numpy','torch'],
zip_safe=False) 
