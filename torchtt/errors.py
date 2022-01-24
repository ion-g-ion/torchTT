"""
Contains the errors used in the `torchtt` package.    
"""
class ShapeMismatch(Exception):
    """The shape of the tensors does not match.
    
    This means that the inputs  have shapes that do not match.
    """
    pass

class RankMismatch(Exception):
    """The TT-ranks do not match.
    
    This means that the inputs shapes that do not match.
    """
    pass

class IncompatibleTypes(Exception):
    """The function arguments are not compatible.
    
    Usually means that a TT matrix was passed as argument instead of a TT tensor (or viceversa).
    """
    pass

class InvalidArguments(Exception):
    """The arguments are not valid.
    
    The arguments passed are not of valid type.
    """
    pass