from setuptools import setup, Extension
import platform

logo_ascii = """
  _                 _   _____ _____ 
 | |_ ___  _ __ ___| |_|_   _|_   _| 
 | __/ _ \| '__/ __| '_ \| |   | |  
 | || (_) | | | (__| | | | |   | |  
  \__\___/|_|  \___|_| |_|_|   |_|  
                                    
"""

try:
    from torch.utils.cpp_extension import BuildExtension, CppExtension
except:
    raise Exception("Torch has to be installed first")

os_name = platform.system()
machine_arch = platform.machine()

if os_name == 'Darwin' and machine_arch == 'arm64':
    ext_modules=[
        CppExtension(
            'torchttcpp', 
            ['cpp/cpp_ext.cpp'], 
            extra_compile_args=['-Xclang', '-fopenmp', '-I/opt/homebrew/opt/libomp/include', '-lblas', '-llapack', '-std=c++14', '-Wno-c++11-narrowing', '-g', '-w', '-O3'],
            extra_link_args=['-L/opt/homebrew/opt/libomp/lib', '-lomp']
        ),
    ]
else:
    ext_modules=[
        CppExtension(
            'torchttcpp', 
            ['cpp/cpp_ext.cpp'], 
            extra_compile_args=['-lblas', '-llapack', '-std=c++14', '-Wno-c++11-narrowing', '-g', '-w', '-O3'],
        ),
    ]

print()
print(logo_ascii)
print()
if os_name == 'Linux' or os_name == 'Darwin':
    setup(
        name='torchTT',
        version='2.0',
        description='Tensor-Train decomposition in pytorch',
        url='https://github.com/ion-g-ion/torchTT',
        author='Ion Gabriel Ion',
        author_email='ion.ion.gabriel@gmail.com',
        license='MIT',
        packages=['torchtt'],
        install_requires=['pytest', 'numpy>=1.18','torch>=1.7','opt_einsum'],
        ext_modules=ext_modules,
        cmdclass={
            'build_ext': BuildExtension
        },
        test_suite='tests',
        zip_safe=False,
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
        ]
    )
else: 
    import warnings
    warnings.warn("\x1B[33m\nC++ implementation not available. Using pure Python.\n\033[0m")
    
    setup(
        name='torchTT',
        version='2.0',
        description='Tensor-Train decomposition in pytorch',
        url='https://github.com/ion-g-ion/torchTT',
        author='Ion Gabriel Ion',
        author_email='ion.ion.gabriel@gmail.com',
        license='MIT',
        packages=['torchtt'],
        install_requires=['numpy>=1.18','torch>=1.7','opt_einsum'],
        test_suite='tests',
        zip_safe=False
    )

