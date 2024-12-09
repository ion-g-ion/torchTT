from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import platform
from warnings import warn

try:
    import torch.utils.cpp_extension
    from torch.utils.cpp_extension import BuildExtension, CppExtension
except ImportError:
    raise Exception("Torch must be installed before running this setup.")

logo_ascii = """
  _                 _   _____ _____ 
 | |_ ___  _ __ ___| |_|_   _|_   _| 
 | __/ _ \\| '__/ __| '_ \\| |   | |  
 | || (_) | | | (__| | | | |   | |  
  \\__\\___/|_|  \\___|_| |_|_|   |_|  
                                    
"""

os_name = platform.system()

print("\n" + logo_ascii + "\n")

if os_name in ['Linux', 'Darwin']:
    try:
        # setup(
        #    # cmdclass={'build_ext': build_ext},
        #     ext_modules=[
        #         Extension(
        #             name='torchttcpp',
        #             sources=['cpp/cpp_ext.cpp'],
        #             include_dirs=torch.utils.cpp_extension.include_paths()+["cpp"],
        #             libray_dirs = torch.utils.cpp_extension.library_paths(),
        #             language='c++',
        #             extra_compile_args=[
        #                 '-lblas', '-llapack', '-std=c++17',
        #                 '-Wno-c++11-narrowing', '-g', '-w', '-O3'
        #             ])
        #     ]
        # )
        
        setup(
            cmdclass={'build_ext': BuildExtension},
            ext_modules=[
                CppExtension(
                    'torchttcpp',
                    ['cpp/cpp_ext.cpp'],
                    include_dirs=["cpp"],
                    libraries=["blas", "lapack", "stdc++"],
                    extra_compile_args=[
                        '-std=c++17',
                        '-Wno-c++11-narrowing', '-w', '-O3',
                    ],
                    extra_link_args=['-Wl,--no-as-needed', '-lm'],
                )
            ],
        )
    except Exception as e:
        warn("\x1B[33m\nC++ implementation not available. Falling back to pure Python.\n\033[0m")
        print(f"Error: {e}")
        #setup()
else:
    warn("\x1B[33m\nC++ implementation not supported on this OS. Falling back to pure Python.\n\033[0m")
    #setup()
