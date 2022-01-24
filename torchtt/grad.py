"""
Adds AD functionality to torchtt.

"""

import torch as tn
import torchtt 

def watch(tens, core_indices = None):
    """
    Watch the TT-cores of a given tensor.

    Args:
        tens (TT-object): the TT-object to be watched.
        core_indices (list[int], optional): The list of cores to be watched. If None is provided, all the cores are watched. Defaults to None.
    """
    if core_indices == None:
        for i in range(len(tens.cores)):
            tens.cores[i].requires_grad_(True)
    else:
       for i in core_indices:
            tens.cores[i].requires_grad_(True)


def unwatch(tens):
    """
    Cancel the autograd graph recording.

    Args:
        tens (torchtt.TT): the tensor.
    """
    for i in range(len(tens.cores)):
        tens.cores[i].requires_grad_(False)


def grad(val, tens, core_indices = None):
    """
    Compute the gradient w.r.t. the cores of the given TT-tensor (or TT-matrix).

    Args:
        val (torch.tensor): Scalar tensor that has to be differentiated.
        tens (TT-object): The given tensor.
        core_indices (list[int], optional): The list of cores to construct the gradient. If None is provided, all the cores are watched. Defaults to None.

    Returns:
        list[torch.tensor]: the list of cores representing the derivative of the expression w.r.t the tensor.
    """
    val.backward()
    if core_indices == None:
        cores = [ c.grad for c in tens.cores]
    else:
        cores = []
        for idx in core_indices:
            cores.append(tens.cores[idx].grad)
    return cores