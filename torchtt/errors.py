class ShapeMismatch(Exception):
    """The shape of the tensors does not match."""
    pass

class IncompatibleTypes(Exception):
    """The function arguments are not compatible"""
    pass