"""
Module for the C++ backend.
"""

import warnings


try:
    import torchttcpp
    _cpp_available = True
except:
    warnings.warn("\x1B[33m\nC++ implementation not available. Using pure Python.\n\033[0m")
    _cpp_available = False
    
def cpp_avaible():
    """
    Return True if C++ backend is available.

    Returns:
        bool: True if C++ backend is available and False otherwise.
    """
    return _cpp_available

