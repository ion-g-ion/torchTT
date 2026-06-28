from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import platform
import os
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

class OptionalBuildExtension(BuildExtension):
    def build_extension(self, ext):
        try:
            super().build_extension(ext)
        except Exception as e:
            warn(f"\x1B[33m\nFailed to build C++ extension '{ext.name}'. Falling back to pure Python.\n\033[0m")
            print(f"Error: {e}")

if os_name in ['Linux', 'Darwin']:
    if os_name == 'Darwin':
        if 'CXX' not in os.environ:
            if os.path.exists('/opt/homebrew/opt/llvm/bin/clang++'):
                os.environ['CXX'] = '/opt/homebrew/opt/llvm/bin/clang++'
            elif os.path.exists('/usr/local/opt/llvm/bin/clang++'):
                os.environ['CXX'] = '/usr/local/opt/llvm/bin/clang++'
            else:
                os.environ['CXX'] = 'clang++'
        extra_link_args = []
    else:
        extra_link_args = ['-Wl,--no-as-needed', '-lm']

    try:
        setup(
            name="torchTT",
            cmdclass={'build_ext': OptionalBuildExtension},
            ext_modules=[
                CppExtension(
                    'torchttcpp',
                    ['cpp/cpp_ext.cpp'],
                    include_dirs=["cpp"],
                    extra_compile_args=[
                        '-std=c++17',
                        '-Wno-c++11-narrowing', '-w', '-O3',
                    ],
                    extra_link_args=extra_link_args,
                )
            ],
        )
    except Exception as e:
        warn("\x1B[33m\nC++ implementation not available. Falling back to pure Python.\n\033[0m")
        print(f"Error: {e}")
        setup(name="torchTT")
else:
    warn("\x1B[33m\nC++ implementation not supported on this OS. Falling back to pure Python.\n\033[0m")
    setup(name="torchTT")
    
