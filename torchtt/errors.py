class ShapeMismatch(Exception):
    """The shape of the tensors does not match."""
    pass

class RankMismatch(Exception):
    """The TT-ranks do not match."""
    pass

class IncompatibleTypes(Exception):
    """The function arguments are not compatible"""
    pass

class InvalidArguments(Exception):
    """The arguments are not valid."""
    pass