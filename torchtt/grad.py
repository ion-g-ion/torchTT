"""
Adds AD functionality to torchtt.

"""

import torch as tn
from torchtt import TT

def watch(tens, core_indices = None):
    """
    Watch the TT-cores of a given tensor.
    Necessary for autograd.
    
    Args:
        tens (torchtt.TT): the TT-object to be watched.
        core_indices (list[int], optional): The list of cores to be watched. If None is provided, all the cores are watched. Defaults to None.
    """
    if core_indices == None:
        for i in range(len(tens.cores)):
            tens.cores[i].requires_grad_(True)
    else:
       for i in core_indices:
            tens.cores[i].requires_grad_(True)

def watch_list(tensors):
    """
    Watch the TT-cores for amultiple tensors givgen in a list.
    Necessary for autograd.

    Args:
        tensors (list[torchtt.TT]): the list of tensors to be wtched for autograd.
    """
    for i in range(len(tensors)):
        for j in range(len(tensors[i].cores)):
            tensors[i].cores[j].requires_grad_(True)
            
            
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
        tens (torchtt.TT): The given tensor.
        core_indices (list[int], optional): The list of cores to construct the gradient. If None is provided, all the cores are watched. Defaults to None.

    Returns:
        list[torch.tensor]: the list of cores representing the derivative of the expression w.r.t the tensor.
    """
    val.retain_grad()
    val.backward()
    if core_indices == None:
        cores = [ c.grad for c in tens.cores]
    else:
        cores = []
        for idx in core_indices:
            cores.append(tens.cores[idx].grad)
    return cores

def grad_list(val, tensors, all_in_one = True):
    """
    Compute the gradient w.r.t. the cores of several given TT-tensors (or TT-oeprators).
    Watch must be called on all of them beforehand.
    
    Args:
        val (torch.tensor): scalar tensor to be differentiated.
        tensors (list[torch.TT]): the tensors with respect to which the differentiation is made.
        all_in_one (bool, optional): Put all the cores in one list or create a list of lists with the cores. Defaults to True.
    
    Returns:
        list[list[torchtt.TT]]: the resulting derivatives.
    """
    val.backward()
    cores_list = []
    if all_in_one:
        for t in tensors:
            cores_list += [ c.grad for c in t.cores]
    else:
        for t in tensors:
            cores_list.append([ c.grad for c in t.cores])
    return cores_list