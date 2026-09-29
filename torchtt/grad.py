"""
Adds automatic differentiation (AD) functionality to torchtt.

The TT cores are ``torch.Tensor`` objects, so derivatives with respect to them are computed by the PyTorch autograd engine.
The functions of this module are shortcuts for it and there is no other difference:

 * :func:`watch` sets ``requires_grad`` of the cores, like ``c.requires_grad_(True)`` for every core ``c`` in ``x.cores``.
 * :func:`grad` calls ``val.backward()`` and returns the gradients ``c.grad`` of the cores.

The following two computations give the same gradients:

.. code-block:: python

    import torch
    import torchtt

    x = torchtt.randn([4, 5, 6], [1, 2, 2, 1])

    # with torchtt.grad
    torchtt.grad.watch(x)
    g = torchtt.grad.grad(x.norm(), x)

    # with PyTorch
    cores = [c.detach().requires_grad_(True) for c in x.cores]
    g = torch.autograd.grad(torchtt.TT(cores).norm(), cores)

Use ``torch.autograd.grad()`` directly when the gradients of successive calls must not add up, when higher derivatives are needed (``create_graph=True``) or when the cores are computed from other tensors that require gradients (they are not leaf tensors, and :func:`grad` returns None for them).
The layers in :mod:`torchtt.nn` store their cores as ``torch.nn.Parameter`` and are trained with the PyTorch optimizers, without this module.

The gradient with respect to the cores is not the gradient with respect to the tensor.
For optimization on the manifold of TT tensors with fixed rank use :func:`torchtt.manifold.riemannian_gradient`, which returns the gradient with respect to the tensor projected on the tangent space, as a TT tensor.

**What can be differentiated.**
The operations that keep the ranks fixed are compositions of PyTorch operations and can be differentiated:
``+``, ``-``, the multiplication with a scalar, the exact products ``*`` and ``@`` (without rounding), the Kronecker product ``**``, :meth:`torchtt.TT.full`, :meth:`torchtt.TT.norm`, :meth:`torchtt.TT.sum`, indexing and slicing, :meth:`torchtt.TT.apply_mask`, :func:`torchtt.dot` and :func:`torchtt.bilinear_form`.

The operations that choose the ranks from the data (with truncated SVDs or iterations) are not supported:
the TT decomposition ``torchtt.TT(full_tensor)``, :meth:`torchtt.TT.round`, :meth:`torchtt.TT.to_qtt`, :func:`torchtt.reshape`, :func:`torchtt.permute`, the division ``/``, the products :func:`torchtt.matvec`, :func:`torchtt.matmat` and :func:`torchtt.hadamard`, and the cross interpolation of :mod:`torchtt.interpolate`.
Most of them raise an error if the cores require gradients; the others (e.g. ``method='swap'`` in the products) can return NaN gradients.
"""

import torch as tn
from torchtt import TT

def watch(tens, core_indices = None):
    """
    Watch the TT-cores of a given tensor.
    Necessary for autograd.

    The flag ``requires_grad`` of the cores is set in place, so PyTorch records every following operation with the tensor.
    Call it before computing the value that is differentiated.
    It is the same as ``tens.cores[i].requires_grad_(True)`` for the watched cores.
    The cores must be leaf tensors (not computed from other tensors that require gradients), otherwise :func:`grad` returns None for them.

    Examples:

        .. code-block:: python

            x = torchtt.randn([4, 5, 6], [1, 2, 2, 1])
            torchtt.grad.watch(x)
            val = (x * x).sum()
            grad_cores = torchtt.grad.grad(val, x)

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
    Same as :func:`watch` for every tensor of the list.

    Args:
        tensors (list[torchtt.TT]): the list of tensors to be wtched for autograd.
    """
    for i in range(len(tensors)):
        for j in range(len(tensors[i].cores)):
            tensors[i].cores[j].requires_grad_(True)


def unwatch(tens):
    """
    Cancel the autograd graph recording.
    The flag ``requires_grad`` of the cores is set to False. The gradients stored in ``c.grad`` from previous calls of :func:`grad` are kept.

    Args:
        tens (torchtt.TT): the tensor.
    """
    for i in range(len(tens.cores)):
        tens.cores[i].requires_grad_(False)


def grad(val, tens, core_indices = None):
    """
    Compute the gradient w.r.t. the cores of the given TT-tensor (or TT-matrix).
    The cores must be watched with :func:`watch` before ``val`` is computed.

    It calls ``val.backward()`` and returns ``c.grad`` for the cores, therefore:

     * the gradients are added to the ones of previous calls (set ``c.grad = None`` for the cores to start again from zero),
     * ``val`` can be differentiated only once (the recorded graph is freed).

    For the first call the result is the same as ``torch.autograd.grad(val, tens.cores)``, which has none of these side effects.

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
    Like :func:`grad`, it calls ``val.backward()``, so the gradients are added to the ones of previous calls.

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
