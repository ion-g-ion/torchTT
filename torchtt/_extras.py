""" 
This file implements additional functions that are visible in the module.
"""

import torch as tn
import torch.nn.functional as tnf
from torchtt._decomposition import mat_to_tt, to_tt, lr_orthogonal, round_tt, rl_orthogonal, QR, SVD, rank_chop
from torchtt._division import amen_divide
import numpy as np
import math
from torchtt._dmrg import dmrg_matvec
from torchtt._aux_ops import apply_mask, dense_matvec, bilinear_form_aux
from torchtt.errors import *
# from ._tt_base import TT
import torchtt._tt_base
import sys


def eye(shape, dtype=tn.float64, device=None):
    """
    Construct the TT decomposition of a multidimensional identity matrix.
    all the TT ranks are 1.

    Args:
        shape (list[int]): the shape.
        dtype (torch.dtype, optional): the dtype of the returned tensor. Defaults to tn.float64.
        device (torch.device, optional): the device where the TT cores are created (None means CPU). Defaults to None.

    Returns:
        torchtt.TT: the one tensor.
    """

    shape = list(shape)

    cores = [tn.unsqueeze(tn.unsqueeze(
        tn.eye(s, dtype=dtype, device=device), 0), 3) for s in shape]

    return torchtt._tt_base.TT(cores)


def zeros(shape, dtype=tn.float64, device=None):
    """
    Construct a tensor that contains only zeros.
    the shape can be a list of ints or a list of tuples of ints. The second case creates a TT matrix.

    Args:
        shape (list[int] | list[tuple[int]]): the shape.
        dtype (torch.dtype, optional): the dtype of the returned tensor. Defaults to tn.float64.
        device (torch.device, optional): the device where the TT cores are created (None means CPU). Defaults to None.

    Raises:
        InvalidArguments: Shape must be a list.

    Returns:
        torchtt.TT: the zero tensor.
    """
    if isinstance(shape, list):
        d = len(shape)
        if isinstance(shape[0], tuple):
            # we create a TT-matrix
            cores = [tn.zeros([1, shape[i][0], shape[i][1], 1],
                              dtype=dtype, device=device) for i in range(d)]

        else:
            # we create a TT-tensor
            cores = [tn.zeros([1, shape[i], 1], dtype=dtype,
                              device=device) for i in range(d)]

    else:
        raise InvalidArguments('Shape must be a list.')

    return torchtt._tt_base.TT(cores)


def kron(first, second):
    """
    Computes the tensor Kronecker product.
    If None is provided as input the reult is the other tensor.
    If A is N_1 x ... x N_d and B is M_1 x ... x M_p, then kron(A,B) is N_1 x ... x N_d x M_1 x ... x M_p


    Args:
        first (torchtt.TT | None): first argument.
        second (torchtt.TT | None): second argument.

    Raises:
        IncompatibleTypes: Incompatible data types (make sure both are either TT-matrices or TT-tensors).
        InvalidArguments: Invalid arguments.

    Returns:
        torchtt.TT: the result.
    """
    if first == None and isinstance(second, torchtt._tt_base.TT):
        cores_new = [c.clone() for c in second.cores]
        result = torchtt._tt_base.TT(cores_new)
    elif second == None and isinstance(first, torchtt._tt_base.TT):
        cores_new = [c.clone() for c in first.cores]
        result = torchtt._tt_base.TT(cores_new)
    elif isinstance(first, torchtt._tt_base.TT) and isinstance(second, torchtt._tt_base.TT):
        if first.is_ttm != second.is_ttm:
            raise IncompatibleTypes(
                'Incompatible data types (make sure both are either TT-matrices or TT-tensors).')

        # concatenate the result
        cores_new = [c.clone() for c in first.cores] + [c.clone()
                                                        for c in second.cores]
        result = torchtt._tt_base.TT(cores_new)
    else:
        raise InvalidArguments('Invalid arguments.')
    return result


def ones(shape, dtype=tn.float64, device=None):
    """
    Construct a tensor that contains only ones.
    the shape can be a list of ints or a list of tuples of ints. The second case creates a TT matrix.

    Args:
        shape (list[int] or list[tuple[int]]): the shape.
        dtype (torch.dtype, optional): the dtype of the returned tensor. Defaults to tn.float64.
        device (torch.device, optional): the device where the TT cores are created (None means CPU). Defaults to None.

    Raises:
        InvalidArguments: Shape must be a list.

    Returns:
        torchtt.TT: the one tensor.
    """
    if isinstance(shape, list):
        d = len(shape)
        if d == 0:
            return torchtt._tt_base.TT(None)
        else:
            if isinstance(shape[0], tuple):
                # we create a TT-matrix
                cores = [tn.ones([1, shape[i][0], shape[i][1], 1],
                                 dtype=dtype, device=device) for i in range(d)]

            else:
                # we create a TT-tensor
                cores = [tn.ones([1, shape[i], 1], dtype=dtype,
                                 device=device) for i in range(d)]

    else:
        raise InvalidArguments('Shape must be a list.')

    return torchtt._tt_base.TT(cores)


def xfun(shape, dtype=tn.float64, device=None):
    """
    Construct a tensor from 0 to tn.prod(shape)-1.
    the shape must be a list of ints.

    Args:
        shape (list[int]): the shape.
        dtype (torch.dtype, optional): the dtype of the returned tensor. Defaults to tn.float64.
        device (torch.device, optional): the device where the TT cores are created (None means CPU). Defaults to None.

    Raises:
        InvalidArguments: Shape must be a list.

    Returns:
        torchtt.TT: the xfun tensor.
    """

    if isinstance(shape, list):
        d = len(shape)
        if d == 0:
            return torchtt._tt_base.TT(None)

        if d == 1:
            return torchtt._tt_base.TT(tn.arange(shape[0], dtype=dtype, device=device))

        else:
            cores = []
            firstcore = tn.ones(1, shape[0], 2, dtype=dtype, device=device)
            firstcore[0, :, 0] = tn.arange(
                shape[0], dtype=dtype, device=device)

            cores.append(firstcore)
            ni = tn.tensor(shape[0], dtype=dtype, device=device)
            for i in range(1, d - 1):
                core = tn.zeros((2, shape[i], 2), dtype=dtype, device=device)
                for j in range(shape[i]):
                    core[:, j, :] = tn.eye(2, dtype=dtype, device=device)
                core[1, :, 0] = ni * \
                    tn.arange(shape[i], dtype=dtype, device=device)
                ni *= shape[i]
                cores.append(core)
            core = tn.ones((2, shape[d - 1], 1), dtype=dtype, device=device)
            core[1, :, 0] = ni * \
                tn.arange(shape[d - 1], dtype=dtype, device=device)
            cores.append(core)
    else:
        raise InvalidArguments('Shape must be a list.')

    return torchtt._tt_base.TT(cores)


def linspace(shape=[1], a=0.0, b=0.0, dtype=tn.float64, device=None):
    """
    Construct an evenly spaced tensor from a to b with a given shape in TT decomposition.
    the shape must be a list of ints.

    Args:
        shape (list[int]): the shape.
        a (float): start value
        b (float): end value
        dtype (torch.dtype, optional): the dtype of the returned tensor. Defaults to tn.float64.
        device (torch.device, optional): the device where the TT cores are created (None means CPU). Defaults to None.

    Raises:
        InvalidArguments: Shape must be a list.

    Returns:
        torchtt.TT: a linspace tensor.
    """
    if isinstance(shape, list):
        d = len(shape)
        if d == 0:
            return torchtt._tt_base.TT(None)

        if d == 1:
            return torchtt._tt_base.TT(tn.linspace(shape[0], a, b, dtype=dtype, device=device))

        else:
            x = xfun(shape)
            oneTensor = ones(shape)
            N = tn.prod(tn.tensor(shape), dtype=dtype, device=device).numpy()
            stepsize = (b - a) * 1.0 / (N - 1)
            T = a * oneTensor + x * stepsize
    else:
        raise InvalidArguments('Shape must be a list.')

    return T.round(1e-15)


def arange(shape=[1], a=0, b=0, step=1, dtype=tn.float64, device=None):
    """
    Construct a tensor of size (a-b)/step with a given shape, if possible.
    the shape must be a list of int and the vector has to fit the shape.

    Args:
        shape (list[int] or list[tuple[int]]): the shape.
        a (float): start value
        b (float): end value
        step (int): stepsize
        dtype (torch.dtype, optional): the dtype of the returned tensor. Defaults to tn.float64.
        device (torch.device, optional): the device where the TT cores are created (None means CPU). Defaults to None.

    Raises:
        InvalidArguments: Shape must be a list.

    Returns:
        torchtt.TT: an evenly spaced tensor within a given interval.
    """
    if isinstance(shape, list):
        d = len(shape)
        if d == 0:
            return torchtt._tt_base.TT(None)

        if d == 1:
            return torchtt._tt_base.TT(tn.arange(a, b, step, dtype=dtype, device=device))
    else:
        raise InvalidArguments('Shape must be a list.')

    return reshape(torchtt._tt_base.TT(tn.arange(a, b, step, dtype=dtype, device=device)), shape)


def random(N, R, dtype=tn.float64, device=None):
    """
    Returns a tensor of shape N with random cores of rank R.
    Each core is a normal distributed with mean 0 and variance 1.
    Check also the method torchtt.randn()for better random tensors in the TT format.

    Args:
        N (list[int] or list[tuple[int]]): the shape of the tensor. If the elements are tuples of integers, we deal with a TT-matrix.
        R (list[int] or int): can be a list if the exact rank is specified or an integer if the maximum rank is secified.
        dtype (torch.dtype, optional): the dtype of the returned tensor. Defaults to tn.float64.
        device (torch.device, optional): the device where the TT cores are created (None means CPU). Defaults to None.

    Raises:
        InvalidArguments: Check if N and R are right.

    Returns:
        torchtt.TT: the result.
    """

    if isinstance(R, int):
        R = [1]+[R]*(len(N)-1)+[1]
    elif len(N)+1 != len(R) or R[0] != 1 or R[-1] != 1 or len(N) == 0:
        raise InvalidArguments('Check if N and R are right.')

    cores = []

    for i in range(len(N)):
        cores.append(tn.randn([R[i], N[i][0], N[i][1], R[i+1]] if isinstance(
            N[i], tuple) else [R[i], N[i], R[i+1]], dtype=dtype, device=device))

    T = torchtt._tt_base.TT(cores)

    return T


def randn(N, R, var=1.0, dtype=tn.float64, device=None):
    """
    A torchtt.TT tensor of shape N = [N1 x ... x Nd] and rank R is returned. 
    The entries of the fuill tensor are alomst normal distributed with the variance var.

    Args:
        N (list[int]): the shape.
        R (list[int]): the rank.
        var (float, optional): the variance. Defaults to 1.0.
        dtype (torch.dtype, optional): the dtype of the returned tensor. Defaults to tn.float64.
        device (torch.device, optional): the device where the TT cores are created (None means CPU). Defaults to None.

    Returns:
        torchtt.TT: the result.
    """

    d = len(N)
    v1 = var / np.prod(R)
    v = v1**(1/d)
    cores = [None] * d
    for i in range(d):
        cores[i] = tn.randn([R[i], N[i][0], N[i][1], R[i+1]] if isinstance(
            N[i], tuple) else [R[i], N[i], R[i+1]], dtype=dtype, device=device)*np.sqrt(v)

    return torchtt._tt_base.TT(cores)


def reshape(tens, shape, eps=1e-16, rmax=sys.maxsize):
    """
    Reshapes a torchtt.TT tensor in the TT format.
    A rounding is also performed.

    Args:
        tens (torchtt.TT): the input tensor.
        shape (list[int] or list[tuple[int]]): the desired shape. In the case of a TT operator the shape has to be given as list of tuples of ints [(M1,N1),...,(Md,Nd)].
        eps (float, optional): relative accuracy. Defaults to 1e-16.
        rmax (int, optional): maximum rank. Defaults to the maximum possible integer.

    Raises:
        ShapeMismatch: The product of modes should remain equal. Check the given shape.

    Returns:
        torchtt.TT: the resulting tensor.
    """

    dfin = len(shape)
    cores, R = rl_orthogonal(tens.cores, tens.R, tens.is_ttm)
    if tens.is_ttm:
        M = []
        N = []
        for t in shape:
            M.append(t[0])
            N.append(t[1])
        if np.prod(tens.N) != np.prod(N) or np.prod(tens.M) != np.prod(M):
            raise ShapeMismatch(
                'The product of modes should remain equal. Check the given shape.')
        core = cores[0]
        cores_new = []

        idx = 0
        idx_shape = 0

        while True:
            if core.shape[1] % M[idx_shape] == 0 and core.shape[2] % N[idx_shape] == 0:
                if core.shape[1] // M[idx_shape] > 1 or core.shape[2] // N[idx_shape] > 1:
                    m1 = M[idx_shape]
                    m2 = core.shape[1] // m1
                    n1 = N[idx_shape]
                    n2 = core.shape[2] // n1
                    r1 = core.shape[0]
                    r2 = core.shape[-1]
                    tmp = tn.reshape(core, [r1*m1, m2, n1, n2*r2])

                    crz, _ = mat_to_tt(
                        tmp, [r1*m1, m2], [n1, n2*r2], eps/np.sqrt(dfin-1), rmax)

                    cores_new.append(tn.reshape(crz[0], [r1, m1, n1, -1]))

                    core = tn.reshape(crz[1], [-1, m2, n2, r2])
                else:
                    cores_new.append(core+0)
                    if idx == len(cores)-1:
                        break
                    else:
                        idx += 1
                        core = cores[idx]
                idx_shape += 1
                if idx_shape == len(shape):
                    break
            else:
                idx += 1
                if idx >= len(cores):
                    break

                core = tn.einsum('ijkl,lmno->ijmkno', core, cores[idx])
                core = tn.reshape(
                    core, [core.shape[0], core.shape[1]*core.shape[2], -1, core.shape[-1]])

    else:
        if np.prod(tens.N) != np.prod(shape):
            raise ShapeMismatch(
                'The product of modes should remain equal. Check the given shape.')

        core = cores[0]
        cores_new = []

        idx = 0
        idx_shape = 0
        while True:
            if core.shape[1] % shape[idx_shape] == 0:
                if core.shape[1] // shape[idx_shape] > 1:
                    s1 = shape[idx_shape]
                    s2 = core.shape[1] // s1
                    r1 = core.shape[0]
                    r2 = core.shape[2]
                    tmp = tn.reshape(core, [r1*s1, s2*r2])

                    crz, _ = to_tt(tmp, tmp.shape, eps/np.sqrt(dfin-1), rmax)

                    cores_new.append(tn.reshape(crz[0], [r1, s1, -1]))

                    core = tn.reshape(crz[1], [-1, s2, r2])
                else:
                    cores_new.append(core+0)
                    if idx == len(cores)-1:
                        break
                    else:
                        idx += 1
                        core = cores[idx]
                idx_shape += 1
                if idx_shape == len(shape):
                    break
            else:
                idx += 1
                if idx >= len(cores):
                    break

                core = tn.einsum('ijk,klm->ijlm', core, cores[idx])
                core = tn.reshape(core, [core.shape[0], -1, core.shape[-1]])

        idx_shape += 1
        while idx_shape < len(shape):
            cores_new.append(
                tn.ones((1, 1, 1), dtype=cores_new[-1].dtype, device=cores_new[-1].device))
            idx_shape += 1

    return torchtt._tt_base.TT(cores_new).round(eps)


def meshgrid(vectors):
    """
    Creates a meshgrid of torchtt.TT objects. Similar to numpy.meshgrid or torch.meshgrid.
    The input is a list of d torch.tensor vectors of sizes N_1, ... ,N_d
    The result is a list of torchtt.TT instances of shapes N1 x ... x Nd.

    Args:
        vectors (list[torch.tensor]): the vectors (1d tensors).

    Returns:
        list[TT]: the resulting meshgrid.
    """

    Xs = []
    dtype = vectors[0].dtype
    for i in range(len(vectors)):
        lst = [tn.ones((1, v.shape[0], 1), dtype=dtype) for v in vectors]
        lst[i] = tn.reshape(vectors[i], [1, -1, 1])
        Xs.append(torchtt._tt_base.TT(lst))
    return Xs


def dot(a, b, axis=None):
    """
    Computes the dot product between 2 tensors in TT format.
    If both a and b have identical mode sizes the result is the dot product.
    If a and b have inequal mode sizes, the function perform index contraction. 
    The number of dimensions of a must be greater or equal as b.
    The modes of the tensor a along which the index contraction with b is performed are given in axis.
    For the compelx case (a,b) = b^H . a.

    Examples:

        .. code-block:: python

            a = torchtt.randn([3,4,5,6,7],[1,2,2,2,2,1])
            b = torchtt.randn([3,4,5,6,7],[1,2,2,2,2,1])
            c = torchtt.randn([3,5,6],[1,2,2,1])
            print(torchtt.dot(a,b))
            print(torchtt.dot(a,c,[0,2,3]))


    Args:
        a (torchtt.TT): the first tensor.
        b (torchtt.TT): the second tensor.
        axis (list[int], optional): the mode indices for index contraction. Defaults to None.

    Raises:
        InvalidArguments: Both operands should be TT instances.
        NotImplementedError: Operation not implemented for TT-matrices.
        ShapeMismatch: Operands are not the same size.
        ShapeMismatch: Number of the modes of the first tensor must be equal with the second.

    Returns:
        float or torchtt.TT: the result. If no axis index is provided the result is a scalar otherwise a torchtt.TT object.
    """

    if not isinstance(a, torchtt._tt_base.TT) or not isinstance(b, torchtt._tt_base.TT):
        raise InvalidArguments('Both operands should be TT instances.')

    if axis == None:
        # treat first the full dot product
        # faster than partial projection
        if a.is_ttm or b.is_ttm:
            raise NotImplementedError(
                'Operation not implemented for TT-matrices.')
        if a.N != b.N:
            raise ShapeMismatch('Operands are not the same size.')

        result = tn.tensor([[1.0]], dtype=a.cores[0].dtype,
                           device=a.cores[0].device)

        for i in range(len(a.N)):
            result = tn.einsum('ab,aim,bin->mn', result,
                               a.cores[i], tn.conj(b.cores[i]))
        result = tn.squeeze(result)
    else:
        # partial case
        if a.is_ttm or b.is_ttm:
            raise NotImplementedError(
                'Operation not implemented for TT-matrices.')
        if len(a.N) < len(b.N):
            raise ShapeMismatch(
                'Number of the modes of the first tensor must be equal with the second.')
        # if a.N[axis] != b.N:
        #     raise Exception('Dimension mismatch.')

        k = 0  # index for the tensor b
        cores_new = []
        rank_left = 1
        for i in range(len(a.N)):
            if i in axis:
                cores_new.append(tn.conj(b.cores[k]))
                rank_left = b.cores[k].shape[2]
                k += 1
            else:
                rank_right = b.cores[k].shape[0] if i+1 in axis else rank_left
                cores_new.append(tn.conj(tn.einsum('ik,j->ijk', tn.eye(rank_left, rank_right,
                                 dtype=a.cores[0].dtype), tn.ones([a.N[i]], dtype=a.cores[0].dtype))))

        result = (a*torchtt._tt_base.TT(cores_new)).sum(axis)
    return result


def bilinear_form(x, A, y):
    """
    Computes the bilinear form x^T A y for TT tensors:

    Args:
        x (torchtt.TT): the tensors.
        A (torchtt.TT): the tensors (must be TT matrix).
        y (torchtt.TT): the tensors.

    Raises:
        InvalidArguments: Inputs must be torchtt.TT instances.
        IncompatibleTypes: x and y must be TT tensors and A must be TT matrix.
        ShapeMismatch: Check the shapes. Required is x.N == A.M and y.N == A.N.

    Returns:
        torch.tensor: the result of the bilienar form as tensor with 1 element.
    """
    if not isinstance(x, torchtt._tt_base.TT) or not isinstance(A, torchtt._tt_base.TT) or not isinstance(y, torchtt._tt_base.TT):
        raise InvalidArguments("Inputs must be torchtt.TT instances.")
    if x.is_ttm or y.is_ttm or A.is_ttm == False:
        raise IncompatibleTypes(
            "x and y must be TT tensors and A must be TT matrix.")
    if x.N != A.M or y.N != A.N:
        raise ShapeMismatch(
            "Check the shapes. Required is x.N == A.M and y.N == A.N.")
    d = len(x.N)
    return bilinear_form_aux(x.cores, A.cores, y.cores, d)


def elementwise_divide(x, y, eps=1e-12, starting_tensor=None, nswp=50, kick=4, local_iterations=40, resets=2, preconditioner=None, verbose=False):
    """
    Perform the elemntwise division x/y of two tensors in the TT format using the AMEN method.
    Use this method if different AMEN arguments are needed.
    This method does not check the validity of the inputs.

    Args:
        x (torchtt.TT or scalar): first tensor (can also be scalar of type float, int, torch.tensor with shape (1)).
        y (torchtt.TT): second tensor.
        eps (float, optional): relative acccuracy. Defaults to 1e-12.
        starting_tensor (torchtt.TT or None, optional): initial guess of the result (None for random initial guess). Defaults to None.
        nswp (int, optional): number of iterations. Defaults to 50.
        kick (int, optional): size of rank enrichment. Defaults to 4.
        local_iterations (int, optional): the number of iterations for the local iterative solver. Defaults to 40.
        resets (int, optional): the number of restarts in the GMRES solver. Defaults to 2.
        preconditioner (string, optional): Use preconditioner for the local solver (possible vaules None, 'c'). Defaults to None. 
        verbose (bool, optional): display debug info. Defaults to False.

    Returns:
        torchtt.TT: the result
    """

    cores_new = amen_divide(y, x, nswp, starting_tensor, eps, rmax=1000, kickrank=kick,
                            local_iterations=local_iterations, resets=resets, verbose=verbose, preconditioner=preconditioner)
    return torchtt._tt_base.TT(cores_new)


def rank1TT(elements):
    """
    Compute the rank 1 TT from a list of vectors (or matrices).

    Args:
        elements (list[torch.tensor]): the list of vectors (or matrices in case a TT matrix should be created).

    Returns:
        torchtt.TT: the resulting TT object.
    """

    return torchtt._tt_base.TT([e[None, ..., None] for e in elements])


def numel(tensor):
    """
    Return the number of entries needed to store the TT cores for the given tensor.

    Args:
        tensor (torchtt.TT): the TT representation of the tensor.

    Returns:
        int: number of floats stored for the TT decomposition.
    """

    return sum([tn.numel(tensor.cores[i]) for i in range(len(tensor.N))])


def diag(input):
    """
    Creates diagonal TT matrix from TT tensor or extracts the diagonal of a TT matrix:

    * If a TT matrix is provided the result is a TT tensor representing the diagonal :math:` \\mathsf{x}_{i_1...i_d} = \\mathsf{A}_{i_1...i_d,i_1...i_d} `

    * If a TT tensor is provided the result is a diagonal TT matrix with the entries :math:` \\mathsf{A}_{i_1...i_d,j_1...j_d} = \\mathsf{x}_{i_1...i_d} \\delta_{i_1}^{j_1} \\cdots \\delta_{i_d}^{j_d} `

    Args:
        input (TT): the input. 

    Raises:
        InvalidArguments: Input must be a torchtt.TT instance.

    Returns:
        torchtt.TT: the result.
    """

    if not isinstance(input, torchtt._tt_base.TT):
        raise InvalidArguments("Input must be a torchtt.TT instance.")

    if input.is_ttm:
        return torchtt._tt_base.TT([tn.diagonal(c, dim1=1, dim2=2).permute([0, 2, 1]) for c in input.cores])
    else:
        return torchtt._tt_base.TT([tn.einsum('ijk,jm->ijmk', c, tn.eye(c.shape[1])) for c in input.cores])


def permute(input, dims, eps=1e-12):
    """
    Permutes the dimensions of the tensor. Works similarily to ``torch.permute``.
    Works like a bubble sort for both TT tensors and TT matrices.

    Examples:

        .. code-block:: python

            x_tt = torchtt.random([5,6,7,8,9],[1,2,3,4,2,1])
            xp_tt = torchtt.permute(x_tt, [4,3,2,1,0], 1e-10)
            print(xp_tt) # the shape of this tensor should be [9,8,7,6,5]


    Args:
        input (torchtt.TT): the input tensor.
        dims (list[int]): the order of the indices in the new tensor.
        eps (float, optional): the relative accuracy of the decomposition. Defaults to 1e-12.

    Raises:
        InvalidArguments: The input must be a TT tensor dims must be a list of integers or a tple of integers.
        ShapeMismatch: `dims` must be the length of the number of dimensions.
        InvalidArguments: Duplicate dims are not allowed.
        InvalidArguments: Dims should only contain integers from 0 to d-1.
    Returns:
        torchtt.TT: the resulting tensor.
    """
    if not isinstance(input, torchtt._tt_base.TT):
        raise InvalidArguments(
            "The input must be a TT tensor dims must be a list of integers or a tple of integers.")
    if len(dims) != len(input.N):
        raise ShapeMismatch(
            "`dims` must be the length of the number of dimensions.")
    if len(dims) != len(set(dims)):
        raise InvalidArguments("Duplicate dims are not allowed.")
    if min(dims) != 0 or max(dims) != len(input.N)-1:
        raise InvalidArguments(
            "Dims should only contain integers from 0 to d-1.")

    cores, R = rl_orthogonal(input.cores, input.R, input.is_ttm)
    d = len(cores)
    eps = eps/(d**1.5)
    indices = list(range(d))

    last_idx = 0

    inversions = True
    while inversions:
        inversions = False

        for i in range(d-1):
            i1 = indices[i]
            i2 = indices[i+1]
            if dims.index(i1) > dims.index(i2):
                # inverion in the index permutation => the cores must be swapped.
                inversions = True

                indices[i] = i2
                indices[i+1] = i1

                # print(indices,' permute ', i1, i2)

                last_idx = i
                if input.is_ttm:
                    # reorthonormalize
                    for k in range(last_idx, i):
                        Q, R = QR(tn.reshape(cores[k], [
                                  cores[k].shape[0]*cores[k].shape[1]*cores[k].shape[2], cores[k].shape[3]]))
                        R[k+1] = Q.shape[1]
                        cores[k] = tn.reshape(
                            Q, [cores[k].shape[0], cores[k].shape[1], cores[k].shape[2], -1])
                        cores[k+1] = tn.einsum('ij,jkl->ikl', R, cores[k+1])

                    n2 = [cores[i].shape[1], cores[i].shape[2]]
                    core = tn.einsum('ijkl,lmno->ijkmno', cores[i], cores[i+1])
                    core = tn.permute(core, [0, 3, 4, 1, 2, 5])
                    U, S, V = SVD(tn.reshape(
                        core, [core.shape[0]*core.shape[1]*core.shape[2], -1]))
                    if S.is_cuda:
                        r_now = min(
                            [rank_chop(S.cpu().numpy(), tn.linalg.norm(S).cpu().numpy()*eps)])
                    else:
                        r_now = min(
                            [rank_chop(S.numpy(), tn.linalg.norm(S).numpy()*eps)])

                    US = U[:, :r_now]@tn.diag(S[:r_now])
                    V = V[:r_now, :]

                    cores[i] = tn.reshape(
                        US, [cores[i].shape[0], cores[i+1].shape[1], cores[i+1].shape[2], -1])
                    R[i+1] = cores[i].shape[2]
                    cores[i+1] = tn.reshape(V, [-1] +
                                            n2 + [cores[i+1].shape[3]])

                else:

                    # reorthonormalize
                    for k in range(last_idx, i):
                        Q, R = QR(tn.reshape(
                            cores[k], [cores[k].shape[0]*cores[k].shape[1], cores[k].shape[2]]))
                        R[k+1] = Q.shape[1]
                        cores[k] = tn.reshape(
                            Q, [cores[k].shape[0], cores[k].shape[1], -1])
                        cores[k+1] = tn.einsum('ij,jkl->ikl', R, cores[k+1])

                    n2 = cores[i].shape[1]
                    core = tn.einsum('ijk,klm->ijlm', cores[i], cores[i+1])
                    core = tn.permute(core, [0, 2, 1, 3])
                    U, S, V = SVD(tn.reshape(
                        core, [core.shape[0]*core.shape[1], -1]))
                    if S.is_cuda:
                        r_now = min(
                            [rank_chop(S.cpu().numpy(), tn.linalg.norm(S).cpu().numpy()*eps)])
                    else:
                        r_now = min(
                            [rank_chop(S.numpy(), tn.linalg.norm(S).numpy()*eps)])

                    US = U[:, :r_now]@tn.diag(S[:r_now])
                    V = V[:r_now, :]

                    cores[i] = tn.reshape(
                        US, [cores[i].shape[0], cores[i+1].shape[1], -1])
                    R[i+1] = cores[i].shape[2]
                    cores[i+1] = tn.reshape(V, [-1, n2, cores[i+1].shape[2]])

    return torchtt._tt_base.TT(cores)


def save(tensor, path):
    """
    Save a `torchtt.TT` object in a file.

    Examples:

        .. code-block:: python

            import torchtt
            #generate a TT object
            A = torchtt.randn([10,20,30,40,4,5],[1,6,5,4,3,2,1])
            # save the TT object
            torchtt.save(A,"./test.TT")
            # load the TT object
            B = torchtt.load("./test.TT")
            # the loaded should be the same
            print((A-B).norm()/A.norm())


    Args:
        tensor (torchtt.TT): the tensor to be saved.
        path (str): the file name.

    Raises:
        InvalidArguments: First argument must be a torchtt.TT instance.
    """
    if not isinstance(tensor, torchtt._tt_base.TT):
        raise InvalidArguments("First argument must be a torchtt.TT instance.")

    if tensor.is_ttm:
        dct = {"is_ttm": tensor.is_ttm, "R": tensor.R,
               "M": tensor.M, "N": tensor.N, "cores": tensor.cores}
        tn.save(dct, path)
    else:
        dct = {"is_ttm": tensor.is_ttm, "R": tensor.R,
               "N": tensor.N, "cores": tensor.cores}
        tn.save(dct, path)


def load(path):
    """
    Load a torchtt.TT object from a file.

    Examples:

        .. code-block:: python

            import torchtt
            #generate a TT object
            A = torchtt.randn([10,20,30,40,4,5],[1,6,5,4,3,2,1])
            # save the TT object
            torchtt.save(A,"./test.TT")
            # load the TT object
            B = torchtt.load("./test.TT")
            # the loaded should be the same
            print((A-B).norm()/A.norm())


    Args:
        path (str): the file name.

    Returns:
        torchtt.TT: the tensor.
    """
    dct = tn.load(path)

    return torchtt._tt_base.TT(dct['cores'])


def cat(tensors, dim=0):
    """
    Concatenate tensors in the TT format along a given dimension `dim`. Only works for TT tensors and not TT matrices.

    Examples:

        .. code-block:: python 

            import torchtt 
            import torch 


            a1 = torchtt.randn((3,4,2,6,7), [1,2,3,4,2,1])
            a2 = torchtt.randn((3,4,8,6,7), [1,3,1,7,5,1])
            a3 = torchtt.randn((3,4,15,6,7), [1,3,10,2,4,1])

            a = torchtt.cat((a1,a2,a3),2)

            af = torch.cat((a1.full(), a2.full(),
            print(torch.linalg.norm(a.full()-af))
        `

    Args:
        tensors (tuple[TT]): the tensors to be concatenated. Their mode sizes must match for all modex except the concatenating dimension.
        dim (int, optional): The dimension to be concatenated after. Defaults to 0.

    Raises:
        InvalidArguments: Not implemented for tensor matrices.
        InvalidArguments: The mode sizes must be the same on the nonconcatenated dimensions for all the provided tensors.
        InvalidArguments: The tensors must have the same number of dimensions.

    Returns:
        torchtt.TT: the result.
    """

    if (len(tensors) == 0):
        return None

    if tensors[0].is_ttm:
        raise InvalidArguments("Not implemented for tensor matrices.")
    Rs = [tensors[0].R]

    for i in range(1, len(tensors)):
        if tensors[i].is_ttm:
            raise InvalidArguments("Not implemented for tensor matrices.")
        if tensors[i].N[:dim] != tensors[0].N[:dim] and tensors[i].N[(dim+1):] != tensors[0].N[(dim+1):]:
            raise InvalidArguments(
                "The mode sizes must be the same on the nonconcatenated dimensions for all the provided tensors.")
        if len(tensors[i].N) != len(tensors[0].N):
            raise InvalidArguments(
                "The tensors must have the same number of dimensions.")
        Rs.append(tensors[i].R)

    cores = []

    if tensors[0].is_ttm:
        pass
    else:

        r_sum = [1]
        for i in range(1, len(tensors[0].N)):
            r_sum.append(sum([Rs[k][i] for k in range(len(tensors))]))
        r_sum.append(1)
        for i in range(len(tensors[0].N)):
            if i == dim:
                n = sum([t.N[dim] for t in tensors])
                cores.append(tn.zeros(
                    (r_sum[i], n, r_sum[i+1]), device=tensors[0].cores[0].device, dtype=tensors[0].cores[0].dtype))
            else:
                cores.append(tn.zeros((r_sum[i], tensors[0].N[i], r_sum[i+1]),
                             device=tensors[0].cores[0].device, dtype=tensors[0].cores[0].dtype))

            offset1 = 0
            offset2 = 0
            offset3 = 0

            for t in tensors:
                if i == dim:
                    cores[i][offset1:(offset1+t.cores[i].shape[0]), offset2:(
                        offset2+t.cores[i].shape[1]), offset3:(offset3+t.cores[i].shape[2])] = t.cores[i]
                    if i > 0:
                        offset1 += t.cores[i].shape[0]
                    offset2 += t.cores[i].shape[1]
                    if i < len(tensors[0].N)-1:
                        offset3 += t.cores[i].shape[2]
                else:
                    cores[i][offset1:(offset1+t.cores[i].shape[0]), :,
                             offset3:(offset3+t.cores[i].shape[2])] = t.cores[i]
                    if i > 0:
                        offset1 += t.cores[i].shape[0]
                    if i < len(tensors[0].N)-1:
                        offset3 += t.cores[i].shape[2]
        # for i in range(len(self.__N)):
        #    pad1 = (0,0 if i == len(self.__N)-1 else other.R[i+1] , 0,0 , 0,0 if i==0 else other.R[i])
        #    pad2 = (0 if i == len(self.__N)-1 else self.__R[i+1],0 , 0,0 , 0 if i==0 else self.R[i],0)
        #    cores.append(tnf.pad(self.cores[i],pad1)+tnf.pad(other.cores[i],pad2))
    return torchtt._tt_base.TT(cores)


def pad(tensor, padding, value=0.0):
    """
    Pad a tensor in the TT format.
    The `padding` argument is a tuple of tuples `((b1, a1), (b2, a2), ... , (bd, ad))`. 
    Each dimension is padded with `bk` at the beginning and `ak` at the end. The padding value is constant and is given as the argument `value`. 
    In case of a TT operator, duiagual padding is performed. On the diagonal, the provided `value` is inserted.

    Args:
        tensor (TT): the tensor to be padded.
        padding (tuple(tuple(int))): the paddings.
        value (float, optional): the value to pad. Defaults to 0.0.

    Raises:
        InvalidArguments: The number of paddings should not exceed the number of dimensions of the tensor.

    Returns:
        TT: the result.
    """
    if (len(padding) > len(tensor.N)):
        raise InvalidArguments(
            "The number of paddings should not exceed the number of dimensions of the tensor.")

    if tensor.is_ttm:
        cores = [c.clone() for c in tensor.cores]
        for pad, k in zip(reversed(padding), reversed(range(len(tensor.N)))):
            cores[k] = tnf.pad(cores[k], (1 if k < len(tensor.N)-1 else 0, 1 if k < len(tensor.N) -
                               1 else 0, pad[0], pad[1], pad[0], pad[1], 1 if k > 0 else 0, 1 if k > 0 else 0), value=0)
            cores[k][0, :pad[0], :pad[0], 0] = value * \
                tn.eye(pad[0], device=cores[k].device, dtype=cores[k].dtype)
            cores[k][-1, (pad[0]+tensor.M[k]):, (pad[0]+tensor.N[k]):, -1] = value * \
                tn.eye(pad[1], device=cores[k].device, dtype=cores[k].dtype)
            value = 1
    else:
        rprod = np.prod(tensor.R)
        value = value/rprod

        cores = [c.clone() for c in tensor.cores]
        for pad, k in zip(reversed(padding), reversed(range(len(tensor.N)))):
            cores[k] = tnf.pad(
                cores[k], (0, 0, pad[0], pad[1], 0, 0), value=value)
            value = 1 if value != 0 else 0

    return torchtt._tt_base.TT(cores)


def shape_tuple_to_mn(shape):
    """
    Convert the shape of a TTM from tuple format to row and column shapes.

    Args:
        shape (list[tuple[int]]): shape.

    Returns:
        tuple[list[int],list[int]]: still the shape.
    """
    M = [s[0] for s in shape]
    N = [s[1] for s in shape]

    return M, N


def shape_mn_to_tuple(M, N):
    """
    Convert the shape of a TTM from row/column format to tuple format.

    Args:
        M (list[int]): row shapes.
        N (list[int]): column shapes.

    Returns:
        list[tuple[int]]: shape.
    """

    return [(m, n) for m, n in zip(M, N)]
