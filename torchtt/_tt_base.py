"""
This file implements the core TT class.
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
import torchtt._extras
import sys


class TT():

    # cores : list[tn.tensor]
    # """ The TT cores as a list of `torch.tensor` instances."""

    @property
    def is_ttm(self):
        """
        Check whether the instance is a TT operator or not.

        Returns:
            bool: the flag.
        """
        return self.__is_ttm

    @property
    def M(self):
        """
        Return the "row" shape in case of TT matrices.

        Raises:
            IncompatibleTypes: The field is_ttm is defined only for TT matrices.

        Returns:
            list[int]: the shape.
        """
        if not self.__is_ttm:
            raise IncompatibleTypes(
                "The field is_ttm is defined only for TT matrices.")
        return self.__M.copy()

    @property
    def N(self):
        """
        Return the shape of a tensor or the "column" shape of a TT operator.

        Returns:
            list[int]: the shape.
        """
        return self.__N.copy()

    @property
    def R(self):
        """
        The rank of the TT decomposition.
        It's length should be ``len(R)==len(N)+1``.

        Returns:
            list[int]: the rank.
        """
        return self.__R.copy()

    def __init__(self, source, shape=None, eps=1e-10, rmax=sys.maxsize):
        """
        Constructor of the TT class. Can convert full tensor in the TT-format (from `torch.tensor` or `numpy.array`).
        In the case of tensor operators of full shape `M1 x ... Md x N1 x ... x Nd`, the shape must be specified as a list of tuples `[(M1,N1),...,(Md,Nd)]`.
        A TT-object can also be computed from cores if the list of cores is passed as argument.
        If None is provided, an empty tensor is created.

        The TT decomposition of a tensor is

        :math:`\\mathsf{x}=\\sum\\limits_{r_1...r_{d-1}=1}^{R_1,...,R_{d-1}} \\mathsf{x}^{(1)}_{1i_1r_1}\\cdots\\mathsf{x}^{(d)}_{r_{d-1}i_d1},`

        where :math::`\\{\\mathsf{x}^{(k)}\\}_{k=1}^d` are the TT cores and ::math:`\\mathbf{R}=(1,R_1,...,R_{d-1},1)` is the TT rank.
        Using the constructor, a TT decomposition of a tensor can be computed. The TT cores are stored as a list in ``torchtt.TT.cores``.   
        This class implements basic operators such as `+,-,*,/,@,**` (add, subtract, elementwise multiplication, elementwise division, matrix vector product and Kronecker product) between TT instances.
        The `examples\` folder server as a tutorial for all the possibilities of the toolbox.

        Examples:

            .. code-block:: python

                import torchtt
                import torch
                x = torch.reshape(torch.arange(0,128,dtype = torch.float64),[8,4,4])
                xtt = torchtt.TT(x)
                ytt = torchtt.TT(torch.squeeze(x),[8,4,4])
                # create a TT matrix
                A = torch.reshape(torch.arange(0,20160,dtype = torch.float64),[3,5,7,4,6,8])
                Att = torchtt.TT(A,[(3,4),(5,6),(7,8)])
                print(Att)        


        Args:
            source (torch.tensor ot list[torch.tensor] or numpy.array or None): the input tensor in full format or the cores. If a `torch.tensor` or `numpy.array` is provided
            shape (list[int] or list[tuple[int]], optional): the shape (if it differs from the one provided). For the TT-matrix case is mandatory. Defaults to None.
            eps (float, optional): tolerance of the TT approximation. Defaults to 1e-10.
            rmax (int or list[int], optional): maximum rank (either a list of integer or an integer). Defaults to the maximum possible integer.

        Raises:
            RankMismatch: Ranks of the given cores do not match (change the spaces of the cores).
            InvalidArguments: Invalid input: TT-cores have to be either 4d or 3d.
            InvalidArguments: Check the ranks and the mode size.
            NotImplementedError: Function only implemented for torch tensors, numpy arrays, list of cores as torch tensors and None

        """

        if source is None:
            # empty TT
            self.cores = []
            self.__M = []
            self.__N = []
            self.__R = [1, 1]
            self.__is_ttm = False

        elif isinstance(source, list):
            # tt cores were passed directly

            # check if sizes are consistent
            prev = 1
            N = []
            M = []
            R = [source[0].shape[0]]
            d = len(source)
            for i in range(len(source)):
                s = source[i].shape

                if s[0] != R[-1]:
                    raise RankMismatch(
                        "Ranks of the given cores do not match: for core number %d previous rank is %d and and current rank is %d." % (i, R[-1], s[0]))
                if len(s) == 3:
                    R.append(s[2])
                    N.append(s[1])
                elif len(s) == 4:
                    R.append(s[3])
                    M.append(s[1])
                    N.append(s[2])
                else:
                    raise InvalidArguments(
                        "Invalid input: TT-cores have to be either 4d or 3d.")

            if len(N) != d or len(R) != d+1 or R[0] != 1 or R[-1] != 1 or (len(M) != 0 and len(M) != len(N)):
                raise InvalidArguments("Check the ranks and the mode size.")

            self.cores = source
            self.__R = R
            self.__N = N
            if len(M) == len(N):
                self.__M = M
                self.__is_ttm = True
            else:
                self.__is_ttm = False
            self.shape = [(m, n) for m, n in zip(self.__M, self.__N)
                          ] if self.__is_ttm else [n for n in self.N]

        elif tn.is_tensor(source):
            if shape == None:
                # no size is given. Deduce it from the tensor. No TT-matrix in this case.
                self.__N = list(source.shape)
                if len(self.__N) > 1:
                    self.cores, self.__R = to_tt(
                        source, self.__N, eps, rmax, is_sparse=False)
                else:
                    self.cores = [tn.reshape(source, [1, self.__N[0], 1])]
                    self.__R = [1, 1]
                self.__is_ttm = False
            elif isinstance(shape, list) and isinstance(shape[0], tuple):
                # if the size contains tuples, we have a TT-matrix.
                if len(shape) > 1:
                    self.__M = [s[0] for s in shape]
                    self.__N = [s[1] for s in shape]
                    self.cores, self.__R = mat_to_tt(
                        source, self.__M, self.__N, eps, rmax)
                    self.__is_ttm = True
                else:
                    self.__M = [shape[0][0]]
                    self.__N = [shape[0][1]]
                    self.cores, self.__R = [tn.reshape(
                        source, [1, shape[0][0], shape[0][1], 1])], [1, 1]
                    self.__is_ttm = True
            else:
                # TT-decomposition with prescribed size
                # perform reshape first
                self.__N = shape
                self.cores, self.__R = to_tt(tn.reshape(
                    source, shape), self.__N, eps, rmax, is_sparse=False)
                self.__is_ttm = False
            self.shape = [(m, n) for m, n in zip(self.__M, self.__N)
                          ] if self.__is_ttm else [n for n in self.N]

        elif isinstance(source, np.ndarray):
            source = tn.tensor(source)

            if shape == None:
                # no size is given. Deduce it from the tensor. No TT-matrix in this case.
                self.__N = list(source.shape)
                if len(self.__N) > 1:
                    self.cores, self.__R = to_tt(
                        source, self.__N, eps, rmax, is_sparse=False)
                else:
                    self.cores = [tn.reshape(source, [1, self.__N[0], 1])]
                    self.__R = [1, 1]
                self.__is_ttm = False
            elif isinstance(shape, list) and isinstance(shape[0], tuple):
                # if the size contains tuples, we have a TT-matrix.
                self.__M = [s[0] for s in shape]
                self.__N = [s[1] for s in shape]
                self.cores, self.__R = mat_to_tt(
                    source, self.__M, self.__N, eps, rmax)
                self.__is_ttm = True
            else:
                # TT-decomposition with prescribed size
                # perform reshape first
                self.__N = shape
                self.cores, self.__R = to_tt(tn.reshape(
                    source, shape), self.__N, eps, rmax, is_sparse=False)
                self.__is_ttm = False
            self.shape = [(m, n) for m, n in zip(self.__M, self.__N)
                          ] if self.__is_ttm else [n for n in self.N]
        else:
            raise NotImplementedError(
                "Function only implemented for torch tensors, numpy arrays, list of cores as torch tensors and None.")

    def cuda(self, device=None):
        """
        Return a torchtt.TT object on the CUDA device by cloning all the cores on the GPU.

        Args:
            device (torch.device, optional): The CUDA device (None for CPU). Defaults to None.

        Returns:
            torchtt.TT: The TT-object. The TT-cores are on CUDA.
        """

        t = TT([c.cuda(device) for c in self.cores])

        return t

    def cpu(self):
        """
        Retrive the cores from the GPU.

        Returns:
            torchtt.TT: The TT-object on CPU.
        """

        return TT([c.cpu() for c in self.cores])

    def is_cuda(self):
        """
        Return True if the tensor is on GPU.

        Returns:
            bool: Is the torchtt.TT on GPU or not.
        """
        return all([c.is_cuda for c in self.core])

    def to(self, device=None, dtype=None):
        """
        Moves the TT instance to the given device with the given dtype.

        Args:
            device (torch.device, optional): The desired device. If none is provided, the device is the CPU. Defaults to None.
            dtype (torch.dtype, optional): The desired dtype (torch.float64, torch.float32,...). If None is provided the dtype is not changed. Defaults to None.
        """
        return TT([c.to(device=device, dtype=dtype) for c in self.cores])

    def detach(self):
        """
        Detaches the TT tensor. Similar to ``torch.tensor.detach()``.

        Returns:
            torchtt.TT: the detached tensor.
        """
        return TT([c.detach() for c in self.cores])

    def clone(self):
        """
        Clones the torchtt.TT instance. Similar to torch.tensor.clone().

        Returns:
            torchtt.TT: the cloned TT object.
        """
        return TT([c.clone() for c in self.cores])

    def set_core(self, k, core):
        """
        Replaces the k-th TT core. This can change the mode size of the TT object.

        Args:
            k (int): the core index.
            core (torch.tensor): _description_

        Raises:
            InvalidArguments: The given core must match the the ranks and the dimensionality.
            InvalidArguments: The index of the core mst match the dimensionality.
        """
        if k >= len(self.__N) or k < 0:
            raise InvalidArguments(
                "The index of the core mst match the dimensionality.")
        if self.__is_ttm:
            if core.shape[0] != self.__R[k] or core.shape[3] != self.__R[k+1] or len(core.shape) != 4:
                raise InvalidArguments(
                    "The given core must match the the ranks and the dimensionality.")
            else:
                self.cores[k] = core.clone()
                self.__M[k] = core.shape[1]
                self.__N[k] = core.shape[2]
        else:
            if core.shape[0] != self.__R[k] or core.shape[2] != self.__R[k+1] or len(core.shape) != 3:
                raise InvalidArguments(
                    " The given core must match the the ranks and the dimensionality.")
            else:
                self.cores[k] = core.clone()
                self.__N[k] = core.shape[1]

    def full(self):
        """
        Return the full tensor.
        In case of a TTM, the result has the shape ``M1 x M2 x ... x Md x N1 x N2 x ... x Nd``.

        Returns:
            torch.tensor: the full tensor.
        """
        if self.__is_ttm:
            # the case of tt-matrix
            tfull = self.cores[0][0, :, :, :]
            for i in range(1, len(self.cores)-1):
                tfull = tn.einsum('...i,ijkl->...jkl', tfull, self.cores[i])
            if len(self.__N) != 1:
                tfull = tn.einsum('...i,ijk->...jk', tfull,
                                  self.cores[-1][:, :, :, 0])
                tfull = tn.permute(tfull, list(
                    np.arange(len(self.__N))*2)+list(np.arange(len(self.N))*2+1))
            else:
                tfull = tfull[:, :, 0]
        else:
            # the case of a normal tt
            tfull = self.cores[0][0, :, :]
            for i in range(1, len(self.cores)-1):
                tfull = tn.einsum('...i,ijk->...jk', tfull, self.cores[i])
            if len(self.__N) != 1:
                tfull = tn.einsum('...i,ij->...j', tfull,
                                  self.cores[-1][:, :, 0])
            else:
                tfull = tn.squeeze(tfull)
        return tfull

    def numpy(self):
        """
        Return the full tensor as a numpy.array.
        In case of a TTM, the result has the shape ``M1 x M2 x ... x Md x N1 x N2 x ... x Nd``.
        If it is involved in an AD graph, an error will occur.

        Returns:
            numpy.array: the full tensor in numpy.
        """
        return self.full().cpu().numpy()

    def __repr__(self):
        """
        Show the information as a string

        Returns:
            string: the string representation of a torchtt.TT
        """

        if self.__is_ttm:
            output = 'TT-matrix'
            output += ' with sizes and ranks:\n'
            output += 'M = ' + str(self.__M) + '\nN = ' + str(self.__N) + '\n'
            output += 'R = ' + str(self.__R) + '\n'
            output += 'Device: ' + \
                str(self.cores[0].device)+', dtype: ' + \
                str(self.cores[0].dtype)+'\n'
            entries = sum([tn.numel(c) for c in self.cores])
            output += '#entries ' + str(entries) + ' compression ' + str(entries/np.prod(
                np.array(self.__N, dtype=np.float64)*np.array(self.__M, dtype=np.float64))) + '\n'
        else:
            output = 'TT'
            output += ' with sizes and ranks:\n'
            output += 'N = ' + str(self.__N) + '\n'
            output += 'R = ' + str(self.__R) + '\n\n'
            output += 'Device: ' + \
                str(self.cores[0].device)+', dtype: ' + \
                str(self.cores[0].dtype)+'\n'
            entries = sum([tn.numel(c) for c in self.cores])
            output += '#entries ' + str(entries) + ' compression ' + str(
                entries/np.prod(np.array(self.__N, dtype=np.float64))) + '\n'

        return output

    def __radd__(self, other):
        """
        Addition in the TT format. Implements the "+" operator. This function is called in the case a non-torchtt.TT object is added to the left.

        Args:
            other (float | int | torch.tensor): the first operand. If a `torch.tensor` is provided, it must have 1 element.

        Returns:
            torchtt.TT: the result.
        """

        return self.__add__(other)

    def __add__(self, other):
        """
        Addition in the TT format. Implements the "+" operator. The following type pairs are supported:
            - both operands are TT-tensors.
            - both operands are TT-matrices.
            - first operand is a TT-tensor or a TT-matrix and the second is a scalar (either torch.tensor scalar or int or float).
        The broadcasting rules from `torch` apply here.

        Args:
            other (torchtt.TT | float | int | torch.tensor): the second operand. If a `torch.tensor` is provided, it must have 1 element.

        Raises:
            ShapeMismatch: Dimension mismatch.
            IncompatibleTypes: Addition between a tensor and a matrix is not defined.

        Returns:
            torchtt.TT: the result.
        """

        if np.isscalar(other) or (tn.is_tensor(other) and tn.numel(other) == 1):
            # the second term is a scalar
            cores = []

            for i in range(len(self.__N)):
                if self.__is_ttm:
                    pad1 = (0, 0 if i == len(self.__N)-1 else 1,
                            0, 0, 0, 0, 0, 0 if i == 0 else 1)
                    pad2 = (0 if i == len(
                        self.__N)-1 else self.__R[i+1], 0, 0, 0, 0, 0, 0 if i == 0 else self.R[i], 0)
                    othr = tn.ones(
                        [1, 1, 1, 1], dtype=self.cores[i].dtype) * (other if i == 0 else 1)
                else:
                    pad1 = (0, 0 if i == len(self.__N)-1 else 1,
                            0, 0, 0, 0 if i == 0 else 1)
                    pad2 = (0 if i == len(
                        self.__N)-1 else self.__R[i+1], 0, 0, 0, 0 if i == 0 else self.R[i], 0)
                    othr = tn.ones(
                        [1, 1, 1], dtype=self.cores[i].dtype) * (other if i == 0 else 1)

                cores.append(tnf.pad(self.cores[i], pad1)+tnf.pad(othr, pad2))

            result = TT(cores)
        elif isinstance(other, TT):
            # second term is TT object
            if self.__is_ttm and other.is_ttm:
                # both are TT-matrices
                if self.__M != self.M or self.__N != self.N:
                    raise ShapeMismatch("Shapes are incompatible: first operand is %s x %s, second operand is %s x %s." % (
                        str(self.M), str(self.N), str(other.M), str(other.N)))

                cores = []
                for i in range(len(self.__N)):
                    pad1 = (0, 0 if i == len(
                        self.__N)-1 else other.R[i+1], 0, 0, 0, 0, 0, 0 if i == 0 else other.R[i])
                    pad2 = (0 if i == len(
                        self.__N)-1 else self.__R[i+1], 0, 0, 0, 0, 0, 0 if i == 0 else self.R[i], 0)
                    cores.append(
                        tnf.pad(self.cores[i], pad1)+tnf.pad(other.cores[i], pad2))

                result = TT(cores)

            elif self.__is_ttm == False and other.is_ttm == False:
                # normal tensors in TT format.
                if self.__N == other.N:
                    cores = []
                    for i in range(len(self.__N)):
                        pad1 = (0, 0 if i == len(
                            self.__N)-1 else other.R[i+1], 0, 0, 0, 0 if i == 0 else other.R[i])
                        pad2 = (0 if i == len(
                            self.__N)-1 else self.__R[i+1], 0, 0, 0, 0 if i == 0 else self.R[i], 0)
                        cores.append(
                            tnf.pad(self.cores[i], pad1)+tnf.pad(other.cores[i], pad2))
                else:
                    if len(self.__N) < len(other.N):
                        raise ShapeMismatch("Shapes are incompatible: first operand is %s, second operand is %s." % (
                            str(self.N), str(other.N)))

                    cores = []
                    for i in range(len(self.cores)-len(other.cores)):
                        pad1 = (0, 0 if i == len(self.__N)-1 else 1,
                                0, 0, 0, 0 if i == 0 else 1)
                        pad2 = (0 if i == len(
                            self.__N)-1 else self.__R[i+1], 0, 0, 0, 0 if i == 0 else self.R[i], 0)
                        cores.append(tnf.pad(self.cores[i], pad1)+tnf.pad(
                            tn.ones((1, self.__N[i], 1), device=self.cores[i].device), pad2))

                    for k, i in zip(range(len(other.cores)), range(len(self.cores)-len(other.cores), len(self.cores))):
                        if other.N[k] == self.__N[i]:
                            pad1 = (0, 0 if i == len(
                                self.__N)-1 else other.R[k+1], 0, 0, 0, 0 if i == 0 else other.R[k])
                            pad2 = (0 if i == len(
                                self.__N)-1 else self.__R[i+1], 0, 0, 0, 0 if i == 0 else self.R[i], 0)
                            cores.append(
                                tnf.pad(self.cores[i], pad1)+tnf.pad(other.cores[k], pad2))

                        elif other.N[k] == 1:
                            pad1 = (0, 0 if i == len(
                                self.__N)-1 else other.R[k+1], 0, 0, 0, 0 if i == 0 else other.R[k])
                            pad2 = (0 if i == len(
                                self.__N)-1 else self.__R[i+1], 0, 0, 0, 0 if i == 0 else self.R[i], 0)
                            cores.append(tnf.pad(
                                self.cores[i], pad1)+tnf.pad(tn.tile(other.cores[k], (1, self.__N[i], 1)), pad2))
                        else:
                            raise ShapeMismatch("Shapes are incompatible: first operand is %s, second operand is %s." % (
                                str(self.N), str(other.N)))

                result = TT(cores)

            else:
                # incompatible types
                raise IncompatibleTypes(
                    'Addition between a tensor and a matrix is not defined.')
        else:
            InvalidArguments('Second term is incompatible.')

        return result

    def __rsub__(self, other):
        """
        Subtract 2 tensors in the TT format. Implements the "-" operator.  

        Args:
            other (torchtt.TT | float | int | torch.tensor): the first operand. If a `torch.tensor` is provided, it must have 1 element.

        Returns:
            torchtt.TT: the result.
        """

        T = self.__sub__(other)
        T.cores[0] = -T.cores[0]
        return T

    def __sub__(self, other):
        """
        Subtract 2 tensors in the TT format. Implements the "-" operator.
        Possible second operands are: torchtt.TT, float, int, torch.tensor with 1 element.
        Broadcasting rules from `torch` apply for this operation as well.

        Args:
            other (torchtt.TT | float | int | torch.tensor): the second operand. If a `torch.tensor` is provided, it must have 1 element.

        Raises:
            ShapeMismatch: Both dimensions of the TT matrix should be equal.
            ShapeMismatch: Dimension mismatch.
            IncompatibleTypes: Addition between a tensor and a matrix is not defined.
            InvalidArguments: Second term is incompatible (must be either torchtt.TT or int or float or torch.tensor with 1 element).

        Returns:
            torchtt.TT: the result.
        """
        if np.isscalar(other) or (tn.is_tensor(other) and other.shape == []):
            # the second term is a scalar
            cores = []

            for i in range(len(self.__N)):
                if self.__is_ttm:
                    pad1 = (0, 0 if i == len(self.__N)-1 else 1,
                            0, 0, 0, 0, 0, 0 if i == 0 else 1)
                    pad2 = (0 if i == len(
                        self.__N)-1 else self.__R[i+1], 0, 0, 0, 0, 0, 0 if i == 0 else self.R[i], 0)
                    othr = tn.ones(
                        [1, 1, 1, 1], dtype=self.cores[i].dtype) * (-other if i == 0 else 1)
                else:
                    pad1 = (0, 0 if i == len(self.__N)-1 else 1,
                            0, 0, 0, 0 if i == 0 else 1)
                    pad2 = (0 if i == len(
                        self.__N)-1 else self.__R[i+1], 0, 0, 0, 0 if i == 0 else self.R[i], 0)
                    othr = tn.ones(
                        [1, 1, 1], dtype=self.cores[i].dtype) * (-other if i == 0 else 1)
                cores.append(tnf.pad(self.cores[i], pad1)+tnf.pad(othr, pad2))
            result = TT(cores)

        elif isinstance(other, TT):
            # second term is TT object
            if self.__is_ttm and other.is_ttm:
                # both are TT-matrices
                if self.__M != self.M or self.__N != self.N:
                    raise ShapeMismatch("Shapes are incompatible: first operand is %s x %s, second operand is %s x %s." % (
                        str(self.M), str(self.N), str(other.M), str(other.N)))

                cores = []
                for i in range(len(self.__N)):
                    pad1 = (0, 0 if i == len(
                        self.__N)-1 else other.R[i+1], 0, 0, 0, 0, 0, 0 if i == 0 else other.R[i])
                    pad2 = (0 if i == len(
                        self.__N)-1 else self.__R[i+1], 0, 0, 0, 0, 0, 0 if i == 0 else self.R[i], 0)
                    cores.append(tnf.pad(
                        self.cores[i], pad1)+tnf.pad(-other.cores[i] if i == 0 else other.cores[i], pad2))

                result = TT(cores)

            elif self.__is_ttm == False and other.is_ttm == False:
                # normal tensors in TT format.
                if self.__N == other.N:
                    cores = []
                    for i in range(len(self.__N)):
                        pad1 = (0, 0 if i == len(
                            self.__N)-1 else other.R[i+1], 0, 0, 0, 0 if i == 0 else other.R[i])
                        pad2 = (0 if i == len(
                            self.__N)-1 else self.__R[i+1], 0, 0, 0, 0 if i == 0 else self.R[i], 0)
                        cores.append(tnf.pad(
                            self.cores[i], pad1)+tnf.pad(-other.cores[i] if i == 0 else other.cores[i], pad2))
                else:
                    if len(self.__N) < len(other.N):
                        raise ShapeMismatch("Shapes are incompatible: first operand is %s, second operand is %s." % (
                            str(self.N), str(other.N)))

                    cores = []
                    for i in range(len(self.cores)-len(other.cores)):
                        pad1 = (0, 0 if i == len(self.__N)-1 else 1,
                                0, 0, 0, 0 if i == 0 else 1)
                        pad2 = (0 if i == len(
                            self.__N)-1 else self.__R[i+1], 0, 0, 0, 0 if i == 0 else self.R[i], 0)
                        cores.append(tnf.pad(self.cores[i], pad1)+tnf.pad((-1 if i == 0 else 1)*tn.ones(
                            (1, self.__N[i], 1), device=self.cores[i].device), pad2))

                    for k, i in zip(range(len(other.cores)), range(len(self.cores)-len(other.cores), len(self.cores))):
                        if other.N[k] == self.__N[i]:
                            pad1 = (0, 0 if i == len(
                                self.__N)-1 else other.R[k+1], 0, 0, 0, 0 if i == 0 else other.R[k])
                            pad2 = (0 if i == len(
                                self.__N)-1 else self.__R[i+1], 0, 0, 0, 0 if i == 0 else self.R[i], 0)
                            cores.append(tnf.pad(
                                self.cores[i], pad1)+tnf.pad(-other.cores[k] if i == 0 else other.cores[k], pad2))

                        elif other.N[k] == 1:
                            pad1 = (0, 0 if i == len(
                                self.__N)-1 else other.R[k+1], 0, 0, 0, 0 if i == 0 else other.R[k])
                            pad2 = (0 if i == len(
                                self.__N)-1 else self.__R[i+1], 0, 0, 0, 0 if i == 0 else self.R[i], 0)
                            cores.append(tnf.pad(self.cores[i], pad1)+tnf.pad(
                                tn.tile(-other.cores[k] if i == 0 else other.cores[k], (1, self.__N[i], 1)), pad2))
                        else:
                            raise ShapeMismatch("Shapes are incompatible: first operand is %s, second operand is %s." % (
                                str(self.N), str(other.N)))

                result = TT(cores)

            else:
                # incompatible types
                raise IncompatibleTypes(
                    'Addition between a tensor and a matrix is not defined.')
        else:
            InvalidArguments(
                'Second term is incompatible (must be either torchtt.TT or int or float or torch.tensor with 1 element).')

        return result

    def __rmul__(self, other):
        """
        Elementwise multiplication in the TT format.
        This implements the "*" operator when the left operand is not torchtt.TT.
        Following are supported:

         * TT tensor and TT tensor
         * TT matrix and TT matrix
         * TT tensor and scalar(int, float or torch.tensor scalar)

        Args:
            other (torchtt.TT | float | int | torch.tensor): the first operand. If a `torch.tensor` is provided, it must have 1 element.

        Raises:
            ShapeMismatch: Shapes must be equal.
            IncompatibleTypes: Second operand must be the same type as the fisrt (both should be either TT matrices or TT tensors).
            InvalidArguments: Second operand must be of type: torchtt.TT, float, int of torch.tensor.

        Returns:
            torchtt.TT: [description]
        """

        return self.__mul__(other)

    def __mul__(self, other):
        """
        Elementwise multiplication in the TT format.
        This implements the "*" operator.
        Following are supported:
         - TT tensor and TT tensor
         - TT matrix and TT matrix
         - TT tensor and scalar(int, float, complex or torch.Tensor scalar)
        The broadcasting rules are the same as in torch (see [here](https://pytorch.org/docs/stable/notes/broadcasting.html)).

        Args:
            other (torchtt.TT | float | int | complex | torch.Tensor): the second operand. If a `torch.Tensor` is provided, it must have 1 element.

        Raises:
            ShapeMismatch: Shapes are incompatible (see the broadcasting rules).
            IncompatibleTypes: Second operand must be the same type as the fisrt (both should be either TT matrices or TT tensors).
            InvalidArguments: Second operand must be of type: torchtt.TT, float, int, complex or torch.Tensor.

        Returns:
            torchtt.TT: the result.
        """

        # elementwise multiplication
        if isinstance(other, TT):
            if self.__is_ttm and other.is_ttm:
                if self.__N == other.N and self.__M == other.M:
                    # raise ShapeMismatch('Shapes must be equal.')

                    cores_new = []

                    for i in range(len(self.cores)):
                        core = tn.reshape(tn.einsum('aijb,mijn->amijbn', self.cores[i], other.cores[i]), [
                                          self.__R[i]*other.R[i], self.__M[i], self.__N[i], self.R[i+1]*other.R[i+1]])
                        cores_new.append(core)

                else:
                    raise ShapeMismatch("Shapes are incompatible: first operand is %s x %s, second operand is %s x %s." % (
                        str(self.M), str(self.N), str(other.M), str(other.N)))
                    # if len(self.__N) < len(other.N):
                    #     raise ShapeMismatch("Shapes are incompatible: first operand is %s x %s, second operand is %s x %s."%(str(self.M), str(self.N), str(other.M), str(other.N)))

                    # cores_new = []
                    # raise NotImplementedError("Not yet implemented.")

            elif self.__is_ttm == False and other.is_ttm == False:
                # broadcasting rul;es have to be applied. Sperate if else to make the non-broadcasting case the fastest.
                if self.__N == other.N:
                    cores_new = []

                    for i in range(len(self.cores)):
                        core = tn.reshape(tn.einsum('aib,min->amibn', self.cores[i], other.cores[i]), [
                                          self.__R[i]*other.R[i], self.__N[i], self.R[i+1]*other.R[i+1]])
                        cores_new.append(core)
                else:
                    if len(self.__N) < len(other.N):
                        raise ShapeMismatch("Shapes are incompatible: first operand is %s, second operand is %s." % (
                            str(self.N), str(other.N)))

                    cores_new = []
                    for i in range(len(self.cores)-len(other.cores)):
                        cores_new.append(self.cores[i]*1)

                    for k, i in zip(range(len(other.cores)), range(len(self.cores)-len(other.cores), len(self.cores))):
                        if other.N[k] == self.__N[i]:
                            core = tn.reshape(tn.einsum('aib,min->amibn', self.cores[i], other.cores[k]), [
                                              self.__R[i]*other.R[k], self.__N[i], self.R[i+1]*other.R[k+1]])
                        elif other.N[k] == 1:
                            core = tn.reshape(tn.einsum('aib,mn->amibn', self.cores[i], other.cores[k][:, 0, :]), [
                                              self.__R[i]*other.R[k], self.__N[i], self.R[i+1]*other.R[k+1]])
                        else:
                            raise ShapeMismatch("Shapes are incompatible: first operand is %s, second operand is %s." % (
                                str(self.N), str(other.N)))

                        cores_new.append(core)

            else:
                raise IncompatibleTypes(
                    'Second operand must be the same type as the fisrt (both should be either TT matrices or TT tensors).')
            result = TT(cores_new)

        elif isinstance(other, int) or isinstance(other, float) or isinstance(other, complex) or isinstance(other, tn.Tensor):
            if other != 0:
                cores_new = [c+0 for c in self.cores]
                cores_new[0] *= other
                result = TT(cores_new)
            else:
                result = TT([tn.zeros((1, self.M[i], self.N[i], 1) if self.is_ttm else (
                    1, self.N[i], 1), device=self.cores[0].device, dtype=self.cores[0].dtype) for i in range(len(self.N))])
                # result = zeros([(m,n) for m,n in zip(self.M,self.N)] if self.is_ttm else self.N, device=self.cores[0].device)
        else:
            raise InvalidArguments(
                'Second operand must be of type: TT, float, int, complex or tensorflow Tensor.')

        return result

    def __matmul__(self, other):
        """
        Matrix-vector multiplication in TT-format
        Supported operands:
            - TT-matrix @ TT-tensor -> TT-tensor: y_i = A_ij * x_j
            - TT-tensor @ TT-matrix -> TT-tensor: y_j = x_i * A_ij 
            - TT-matrix @ TT-matrix -> TT-matrix: Y_ij = A_ik * B_kj
            - TT-matrix @ torch.tensor -> torch.tensor: y_bi = A_ij * x_bj 
        In the last case, the multiplication is performed along the last modes and a full torch.tensor is returned.

        Args:
            other (torchtt.TT | torch.tensor): the second operand.

        Raises:
            ShapeMismatch: Shapes do not match.
            InvalidArguments: Wrong arguments.

        Returns:
            torchtt.TT | torch.tensor: the result. Can be full tensor if the second operand is full tensor.
        """

        if self.__is_ttm and tn.is_tensor(other):
            if self.__N != list(other.shape)[-len(self.N):]:
                raise ShapeMismatch("Shapes do not match.")
            result = dense_matvec(self.cores, other)
            return result

        elif self.__is_ttm and other.is_ttm == False:
            # matrix-vector multiplication
            if self.__N != other.N:
                raise ShapeMismatch("Shapes do not match.")

            cores_new = []

            for i in range(len(self.cores)):
                core = tn.reshape(tn.einsum('ijkl,mkp->imjlp', self.cores[i], other.cores[i]), [
                                  self.cores[i].shape[0]*other.cores[i].shape[0], self.cores[i].shape[1], self.cores[i].shape[3]*other.cores[i].shape[2]])
                cores_new.append(core)

        elif self.__is_ttm and other.is_ttm:
            # multiplication between 2 TT-matrices
            if self.__N != other.M:
                raise ShapeMismatch("Shapes do not match.")

            cores_new = []

            for i in range(len(self.cores)):
                core = tn.reshape(tn.einsum('ijkl,mknp->imjnlp', self.cores[i], other.cores[i]), [
                                  self.cores[i].shape[0]*other.cores[i].shape[0], self.cores[i].shape[1], other.cores[i].shape[2], self.cores[i].shape[3]*other.cores[i].shape[3]])
                cores_new.append(core)
        elif self.__is_ttm == False and other.is_ttm:
            # vector-matrix multiplication
            if self.__N != other.M:
                raise ShapeMismatch("Shapes do not match.")

            cores_new = []

            for i in range(len(self.cores)):
                core = tn.reshape(tn.einsum('mkp,ikjl->imjlp', self.cores[i], other.cores[i]), [
                                  self.cores[i].shape[0]*other.cores[i].shape[0], other.cores[i].shape[2], self.cores[i].shape[2]*other.cores[i].shape[3]])
                cores_new.append(core)
        else:
            raise InvalidArguments("Wrong arguments.")

        result = TT(cores_new)
        return result

    def fast_matvec(self, other, eps=1e-12, initial=None, nswp=20, verb=False, use_cpp=True):
        """
        Fast matrix vector multiplication A@x using DMRG iterations. Faster than traditional matvec + rounding.

        Args:
            other (torchtt.TT): the TT tensor.
            eps (float, optional): relative accuracy for DMRG. Defaults to 1e-12.
            initial (None|torchtt.TT, optional): an approximation of the product (None means random initial guess). Defaults to None.
            nswp (int, optional): number of DMRG iterations. Defaults to 40.
            verb (bool, optional): show info for debug. Defaults to False.
            use_cpp (bool, optional): use the C++ implementation if available. Defaults to True.

        Raises:
            InvalidArguments: Second operand has to be TT object.
            IncompatibleTypes: First operand should be a TT matrix and second a TT vector.

        Returns:
            torchtt.TT: the result.
        """

        if not isinstance(other, TT):
            raise InvalidArguments('Second operand has to be TT object.')
        if not self.__is_ttm or other.is_ttm:
            raise IncompatibleTypes(
                'First operand should be a TT matrix and second a TT vector.')

        return dmrg_matvec(self, other, y0=initial, eps=eps, verb=verb, nswp=nswp, use_cpp=use_cpp)

    def apply_mask(self, indices):
        """
        Evaluate the tensor on the given index list.

        Examples:

            .. code-block:: python 

                x = torchtt.random([10,12,14],[1,4,5,1])
                indices = torch.tensor([[0,0,0],[1,2,3],[1,1,1]])
                val = x.apply_mask(indices)


        Args:
            indices (list[list[int]]): the index list where the tensor should be evaluated. Length is M.

        Returns:
            torch.tensor: the values of the tensor

        """
        result = apply_mask(self.cores, self.__R, indices)
        return result

    def __truediv__(self, other):
        """
        This function implements the "/" operator.
        This operation is performed using the AMEN solver. The number of sweeps and rthe relative accuracy are fixed.
        For most cases it is sufficient but sometimes it can fail.
        Check the function torchtt.elementwise_divide() if you want to change the arguments of the AMEN solver.


        Args:
            other (torchtt.TT | float | int | torch.tensor): the second operand. If a `torch.tensor` is provided, it must have 1 element.

        Raises:
            IncompatibleTypes: Operands should be either TT or TTM.
            ShapeMismatch: Both operands should have the same shape.
            InvalidArguments: Operand not permitted. A TT-object can be divided only with scalars.

        Returns:
            torchtt.TT: the result.
        """
        if isinstance(other, int) or isinstance(other, float) or tn.is_tensor(other):
            # divide by a scalar
            cores_new = self.cores.copy()
            cores_new[0] /= other
            result = TT(cores_new)
        elif isinstance(other, TT):
            if self.__is_ttm != other.is_ttm:
                raise IncompatibleTypes('Operands should be either TT or TTM.')
            if self.__N != other.N or (self.__is_ttm and self.__M != other.M):
                raise ShapeMismatch(
                    "Both operands should have the same shape.")
            result = TT(amen_divide(other, self, 50,
                        None, 1e-12, 500, verbose=False))
        else:
            raise InvalidArguments(
                'Operand not permitted. A TT-object can be divided only with scalars.')

        return result

    def __rtruediv__(self, other):
        """
        Right true division. this function is called when a non TT object is divided by a TT object.
        This operation is performed using the AMEN solver. The number of sweeps and rthe relative accuracy are fixed.
        For most cases it is sufficient but sometimes it can fail.
        Check the function torchtt.elementwise_divide() if you want to change the arguments of the AMEN solver.

        Example: 

            .. code-block:: python

                z = 1.0/x # x is TT instance


        Args:
            other (torchtt.TT | float | int | torch.tensor): the first operand. If a `torch.tensor` is provided, it must have 1 element.

        Raises:
            InvalidArguments: The first operand must be int, float or 1d torch.tensor.

        Returns:
            torchtt.TT: the result.
        """
        if isinstance(other, int) or isinstance(other, float) or (tn.is_tensor(other) and other.numel() == 1):
            # ones(self.__N,dtype=self.cores[0].dtype,device = self.cores[0].device)
            o = TT([tn.ones((1, n, 1), dtype=self.cores[0].dtype,
                   device=self.cores[0].device) for n in self.__N])
            o.cores[0] *= other
            cores_new = amen_divide(
                self, o, 50, None, 1e-12, 500, verbose=False)
        else:
            raise InvalidArguments(
                "The first operand must be int, float or 1d torch.tensor.")

        return TT(cores_new)

    def t(self):
        """
        Returns the transpose of a given TT matrix.


        Returns:
            torchtt.TT: the transpose.

        Raises:
            InvalidArguments: Has to be TT matrix.
        """
        if not self.__is_ttm:
            raise InvalidArguments('Has to be TT matrix.')

        cores_new = [tn.permute(c, [0, 2, 1, 3]) for c in self.cores]

        return TT(cores_new)

    def norm(self, squared=False):
        """
        Computes the frobenius norm of a TT object.

        Args:
            squared (bool, optional): returns the square of the norm if True. Defaults to False.

        Returns:
            torch.tensor: the norm.
        """

        if any([c.requires_grad or c.grad_fn != None for c in self.cores]):
            norm = tn.tensor([[1.0]], dtype=self.cores[0].dtype,
                             device=self.cores[0].device)

            if self.__is_ttm:
                for i in range(len(self.__N)):
                    norm = tn.einsum('ab,aijm,bijn->mn', norm,
                                     self.cores[i], tn.conj(self.cores[i]))
                norm = tn.squeeze(norm)
            else:

                for i in range(len(self.__N)):
                    norm = tn.einsum('ab,aim,bin->mn', norm,
                                     self.cores[i], tn.conj(self.cores[i]))
                norm = tn.squeeze(norm)
            if squared:
                return norm
            else:
                return tn.sqrt(tn.abs(norm))

        else:
            d = len(self.cores)

            core_now = self.cores[0]
            for i in range(d-1):
                if self.__is_ttm:
                    mode_shape = [core_now.shape[1], core_now.shape[2]]
                    core_now = tn.reshape(
                        core_now, [core_now.shape[0]*core_now.shape[1]*core_now.shape[2], -1])
                else:
                    mode_shape = [core_now.shape[1]]
                    core_now = tn.reshape(
                        core_now, [core_now.shape[0]*core_now.shape[1], -1])

                # perform QR
                Qmat, Rmat = QR(core_now)

                # take next core
                core_next = self.cores[i+1]
                shape_next = list(core_next.shape[1:])
                core_next = tn.reshape(core_next, [core_next.shape[0], -1])
                core_next = Rmat @ core_next
                core_next = tn.reshape(core_next, [Qmat.shape[1]]+shape_next)

                # update the cores

                core_now = core_next
            if squared:
                return tn.linalg.norm(core_next)**2
            else:
                return tn.linalg.norm(core_next)

    def sum(self, index=None):
        """
        Contracts a tensor in the TT format along the given indices and retuyrns the resulting tensor in the TT format.
        If no index list is given, the sum over all indices is performed.

        Examples:

            .. code-block:: python
            a = torchtt.ones([3,4,5,6,7])
            print(a.sum()) 
            print(a.sum([0,2,4]))
            print(a.sum([1,2]))
            print(a.sum([0,1,2,3,4]))


        Args:
            index (int | list[int] | None, optional): the indices along which the summation is performed. None selects all of them. Defaults to None.

        Raises:
            InvalidArguments: Invalid index.

        Returns:
            torchtt.TT/torch.tensor: the result.
        """

        if index != None and isinstance(index, int):
            index = [index]
        if not isinstance(index, list) and index != None:
            raise InvalidArguments('Invalid index.')

        if index == None:
            # the case we need to sum over all modes
            if self.__is_ttm:
                C = tn.reduce_sum(self.cores[0], [0, 1, 2])
                for i in range(1, len(self.__N)):
                    C = tn.sum(tn.einsum('i,ijkl->jkl',
                               C, self.cores[i]), [0, 1])
                S = tn.sum(C)
            else:
                C = tn.sum(self.cores[0], [0, 1])
                for i in range(1, len(self.__N)):
                    C = tn.sum(tn.einsum('i,ijk->jk', C, self.cores[i]), 0)
                S = tn.sum(C)
        else:
            # we return the TT-tensor with summed indices
            cores = []

            if self.__is_ttm:
                tmp = [1, 2]
            else:
                tmp = [1]

            for i in range(len(self.__N)):
                if i in index:
                    C = tn.sum(self.cores[i], tmp, keepdim=True)
                    cores.append(C)
                else:
                    cores.append(self.cores[i])

            S = TT(cores)
            S.reduce_dims()
            if len(S.cores) == 1 and tn.numel(S.cores[0]) == 1:
                S = tn.squeeze(S.cores[0])
        return S

    def to_ttm(self):
        """
        Converts a TT-tensor to the TT-matrix format. In the tensor has the shape N1 x ... x Nd, the result has the shape 
        N1 x ... x Nd x 1 x ... x 1.

        Returns:
            torch.TT: the result
        """

        cores_new = [tn.reshape(
            c, (c.shape[0], c.shape[1], 1, c.shape[2])) for c in self.cores]
        return TT(cores_new)

    def reduce_dims(self, exclude=[]):
        """
        Reduces the size 1 modes of the TT-object.
        At least one mode should be larger than 1.

        Args:
            exclude (list, optional): Indices to exclude. Defaults to [].
        """

        # TODO: implement a version that reduces the rank also. by spliting the cores with modes 1 into 2 using the SVD.

        if self.__is_ttm:
            cores_new = []

            for i in range(len(self.__N)):

                if self.cores[i].shape[1] == 1 and self.cores[i].shape[2] == 1 and not i in exclude:
                    if self.cores[i].shape[0] > self.cores[i].shape[3] or i == len(self.__N)-1:
                        # multiply to the left
                        if len(cores_new) > 0:
                            cores_new[-1] = tn.einsum('ijok,kl->ijol',
                                                      cores_new[-1], self.cores[i][:, 0, 0, :])
                        else:
                            # there is no core to the left. Multiply right.
                            if i != len(self.__N)-1:
                                self.cores[i+1] = tn.einsum(
                                    'ij,jkml->ikml', self.cores[i][:, 0, 0, :], self.cores[i+1])
                            else:
                                cores_new.append(self.cores[i])

                    else:
                        # multiply to the right. Set the carry
                        self.cores[i+1] = tn.einsum('ij,jkml->ikml',
                                                    self.cores[i][:, 0, 0, :], self.cores[i+1])

                else:
                    cores_new.append(self.cores[i])

            # update the cores and ranks and shape
            self.__N = []
            self.__M = []
            self.__R = [1]
            for i in range(len(cores_new)):
                self.__N.append(cores_new[i].shape[2])
                self.__M.append(cores_new[i].shape[1])
                self.__R.append(cores_new[i].shape[3])
            self.cores = cores_new
        else:
            cores_new = []

            for i in range(len(self.__N)):

                if self.cores[i].shape[1] == 1 and not i in exclude:
                    if self.cores[i].shape[0] > self.cores[i].shape[2] or i == len(self.__N)-1:
                        # multiply to the left
                        if len(cores_new) > 0:
                            cores_new[-1] = tn.einsum('ijk,kl->ijl',
                                                      cores_new[-1], self.cores[i][:, 0, :])
                        else:
                            # there is no core to the left. Multiply right.
                            if i != len(self.__N)-1:
                                self.cores[i+1] = tn.einsum(
                                    'ij,jkl->ikl', self.cores[i][:, 0, :], self.cores[i+1])
                            else:
                                cores_new.append(self.cores[i])

                    else:
                        # multiply to the right. Set the carry
                        self.cores[i+1] = tn.einsum('ij,jkl->ikl',
                                                    self.cores[i][:, 0, :], self.cores[i+1])

                else:
                    cores_new.append(self.cores[i])

            # update the cores and ranks and shape
            self.__N = []
            self.__R = [1]
            for i in range(len(cores_new)):
                self.__N.append(cores_new[i].shape[1])
                self.__R.append(cores_new[i].shape[2])
            self.cores = cores_new

        self.shape = [(m, n) for m, n in zip(self.__M, self.__N)
                      ] if self.__is_ttm else [n for n in self.N]

    def __getitem__(self, index):
        """
        Performs slicing of a TT object.
        Both TT matrix and TT tensor are supported.
        Similar to pytorch or numpy slicing.

        Args:
            index (tuple[slice] | tuple[int] | int | Ellipsis | slice): the slicing.

        Raises:
            NotImplementedError: Ellipsis are not supported.
            InvalidArguments: Slice size is invalid.
            InvalidArguments: Slice carguments not valid. They have to be either int, slice | None.
            InvalidArguments: Invalid slice. Tensor is not 1d.


        Returns:
            torchtt.TT | torch.tensor: the result. If all the indices are fixed, a scalar torch.tensor is returned otherwise a torchtt.TT.
        """

        # slicing function

        # TODO: include Ellipsis support for tensor operators.

        # if a slice containg integers is passed, an element is returned
        # if ranged slices are used, a TT-object has to be returned.

        exclude = []

        if isinstance(index, tuple):
            # check if more than two Ellipsis are to be found.
            if index.count(Ellipsis) > 1 or (self.is_ttm and index.count(Ellipsis) > 0):
                raise NotImplementedError(
                    'Ellipsis are not supported more than once of for tensor operators.')

            if self.__is_ttm:

                cores_new = []
                k = 0
                for i in range(len(index)//2):
                    idx1 = index[i]
                    idx2 = index[i+len(index)//2]
                    if isinstance(idx1, slice) and isinstance(idx2, slice):
                        cores_new.append(self.cores[k][:, idx1, idx2, :])
                        k += 1
                    elif idx1 == None and idx2 == None:
                        # extend the tensor
                        tmp = tn.eye(cores_new[-1].shape[-1] if len(cores_new) != 0 else 1,
                                     device=self.cores[0].device, dtype=self.cores[0].dtype)[:, None, None, :]
                        cores_new.append(tmp)
                        exclude.append(i)
                    elif isinstance(idx1, int) and isinstance(idx2, int):
                        cores_new.append(tn.reshape(self.cores[k][:, idx1, idx2, :], [
                                         self.__R[k], 1, 1, self.R[k+1]]))
                        k += 1
                    else:
                        raise InvalidArguments(
                            "Slice carguments not valid. They have to be either int, slice or None.")
                if k < len(self.cores):
                    raise InvalidArguments('Slice size is invalid.')

            else:
                # if len(index) != len(self.__N):
                #    raise InvalidArguments('Slice size is invalid.')
                num_none = sum([i is None for i in index])

                if index[0] == Ellipsis:
                    index = (slice(None, None, None),) * \
                        (len(self.__N)-len(index)+1+num_none) + index[1:]
                elif index[-1] == Ellipsis:
                    index = index[:-1] + (slice(None, None, None),) * \
                        (len(self.__N)-len(index)+1+num_none)
                cores_new = []
                k = 0
                for i, idx in enumerate(index):
                    if isinstance(idx, slice):
                        cores_new.append(self.cores[k][:, idx, :])
                        k += 1
                    elif idx is None:
                        # extend the tensor
                        tmp = tn.eye(cores_new[-1].shape[-1] if len(cores_new) != 0 else 1,
                                     device=self.cores[0].device, dtype=self.cores[0].dtype)[:, None, :]
                        cores_new.append(tmp)
                        exclude.append(i)
                    elif isinstance(idx, int):
                        cores_new.append(tn.reshape(self.cores[k][:, idx, :], [
                                         self.__R[k], -1, self.R[k+1]]))
                        k += 1
                    else:
                        raise InvalidArguments(
                            "Slice carguments not valid. They have to be either int, slice or None.")
                if k < len(self.cores):
                    raise InvalidArguments('Slice size is invalid.')

            sliced = TT(cores_new)
            sliced.reduce_dims(exclude)
            if (sliced.is_ttm == False and sliced.N == [1]) or (sliced.is_ttm and sliced.N == [1] and sliced.M == [1]):
                sliced = tn.squeeze(sliced.cores[0])

            # cores = None

        elif isinstance(index, int):
            # tensor is 1d and one element is retrived
            if len(self.__N) == 1:
                sliced = self.cores[0][0, index, 0]
            else:
                raise InvalidArguments('Invalid slice. Tensor is not 1d.')

            # TODO
        elif index == Ellipsis:
            # return a copy of the tensor
            sliced = TT([c.clone() for c in self.cores])

        elif isinstance(index, slice):
            # tensor is 1d and one slice is extracted
            if len(self.__N) == 1:
                sliced = TT(self.cores[0][:, index, :])
            else:
                raise InvalidArguments('Invalid slice. Tensor is not 1d.')
            # TODO
        else:
            raise InvalidArguments('Invalid slice.')

        return sliced

    def __pow__(self, other):
        """
        Computes the tensor Kronecker product.
        This implements the "**" operator.
        If None is provided as input the reult is the other tensor.
        If A is N_1 x ... x N_d and B is M_1 x ... x M_p, then kron(A,B) is N_1 x ... x N_d x M_1 x ... x M_p


        Args:
            first (torchtt.TT or None): first argument.
            second (torchtt.TT or none): second argument.

        Raises:
            IncompatibleTypes: Incompatible data types (make sure both are either TT-matrices or TT-tensors).
            InvalidArguments: Invalid arguments.

        Returns:
            torchtt.TT: the result.
        """

        if other == None:
            cores_new = [c.clone() for c in self.cores]
            result = TT(cores_new)
        elif isinstance(other, TT):
            if self.is_ttm != other.is_ttm:
                raise IncompatibleTypes(
                    'Incompatible data types (make sure both are either TT-matrices or TT-tensors).')

            # concatenate the result
            cores_new = [c.clone() for c in self.cores] + [c.clone()
                                                           for c in other.cores]
            result = TT(cores_new)
        else:
            raise InvalidArguments('Invalid arguments.')

        return result

    def __rpow__(self, other):
        """
        Computes the tensor Kronecker product.
        This implements the "**" operator.
        If None is provided as input the reult is the other tensor.
        If A is N_1 x ... x N_d and B is M_1 x ... x M_p, then kron(A,B) is N_1 x ... x N_d x M_1 x ... x M_p


        Args:
            first (torchtt.TT or None): first argument.
            second (torchtt.TT or none): second argument.

        Raises:
            IncompatibleTypes: Incompatible data types (make sure both are either TT-matrices or TT-tensors).
            InvalidArguments: Invalid arguments.

        Returns:
            torchtt.TT: the result.
        """

        result = kron(self, other)

        return result

    def __neg__(self):
        """
        Returns the negative of a given TT tensor.
        This implements the unery operator "-"

        Returns:
            torchtt.TT: the negated tensor.
        """

        cores_new = [c.clone() for c in self.cores]
        cores_new[0] = -cores_new[0]
        return TT(cores_new)

    def __pos__(self):
        """
        Implements the unary "+" operator returning a copy o the tensor.

        Returns:
            torchtt.TT: the tensor clone.
        """

        cores_new = [c.clone() for c in self.cores]

        return TT(cores_new)

    def round(self, eps=1e-12, rmax=sys.maxsize):
        """
        Implements the rounding operations within a given tolerance epsilon.
        The maximum rank is also provided.

        Args:
            eps (float, optional): the relative accuracy. Defaults to 1e-12.
            rmax (int, optional): the maximum rank. Defaults to the maximum possible integer.

        Returns:
            torchtt.TT: the result.
        """

        # rmax is not list
        if not isinstance(rmax, list):
            rmax = [1] + len(self.__N)*[rmax] + [1]

        # call the round function
        tt_cores, R = round_tt(
            self.cores, self.__R.copy(), eps, rmax, self.__is_ttm)
        # creates a new TT and return it
        T = TT(tt_cores)

        return T

    def to_qtt(self, eps=1e-12, mode_size=2, rmax=sys.maxsize):
        """
        Converts a tensor to the QTT format: N1 x N2 x ... x Nd -> mode_size x mode_size x ... x mode_size.
        The product of the mode sizes should be a power of mode_size.
        The tensor in QTT can be converted back using the qtt_to_tens() method.

        Examples:

            .. code-block:: python

                x = torchtt.random([16,8,64,128],[1,2,10,12,1])
                x_qtt = x.to_qtt()
                print(x_qtt)
                xf = x_qtt.qtt_to_tens(x.N) # a TT-rounding is recommended.


        Args:
            eps (float,optional): the accuracy. Defaults to 1e-12.
            mode_size (int, optional): the size of the modes. Defaults to 2.
            rmax (int): the maximum rank. Defaults to the maximum possible integer.


        Raises:
            ShapeMismatch: Only quadratic TTM can be tranformed to QTT.
            ShapeMismatch: Reshaping error: check if the dimensions are powers of the desired mode size.

        Returns:
            torchtt.TT: the resulting reshaped tensor.

        """

        cores_new = []
        if self.__is_ttm:
            shape_new = []
            for i in range(len(self.__N)):
                if self.__N[i] != self.__M[i]:
                    raise ShapeMismatch(
                        'Only quadratic TTM can be tranformed to QTT.')
                if self.__N[i] == mode_size**int(math.log(self.N[i], mode_size)):
                    shape_new += [(mode_size, mode_size)] * \
                        int(math.log(self.__N[i], mode_size))
                else:
                    raise ShapeMismatch('Reshaping error: check if the dimensions are powers of the desired mode size:\r\ncore size '+str(
                        list(self.cores[i].shape))+' cannot be reshaped.')

            result = torchtt._extras.reshape(self, shape_new, eps, rmax)
        else:
            for core in self.cores:
                if int(math.log(core.shape[1], mode_size)) > 1:
                    Nnew = [core.shape[0]*mode_size]+[mode_size] * \
                        (int(
                            math.log(core.shape[1], mode_size))-2)+[core.shape[2]*mode_size]
                    try:
                        core = tn.reshape(core, Nnew)
                    except:
                        raise ShapeMismatch('Reshaping error: check if the dimensions care powers of the desired mode size:\r\ncore size '+str(
                            list(core.shape))+' cannot be reshaped to '+str(Nnew))
                    cores, _ = to_tt(core, Nnew, eps, rmax, is_sparse=False)
                    cores_new.append(tn.reshape(
                        cores[0], [-1, mode_size, cores[0].shape[-1]]))
                    cores_new += cores[1:-1]
                    cores_new.append(tn.reshape(
                        cores[-1], [cores[-1].shape[0], mode_size, -1]))
                else:
                    cores_new.append(core)
            result = TT(cores_new)

        return result

    def qtt_to_tens(self, original_shape):
        """
        Transform a tensor back from QTT.

        Args:
            original_shape (list): the original shape.

        Raises:
            InvalidArguments: Original shape must be a list.
            ShapeMismatch: Mode sizes do not match.

        Returns:
            torchtt.TT: the folded tensor.
        """

        if not isinstance(original_shape, list):
            raise InvalidArguments("Original shape must be a list.")

        core = None
        cores_new = []

        if self.__is_ttm:
            pass
        else:
            k = 0
            for c in self.cores:
                if core == None:
                    core = c
                    so_far = core.shape[1]
                else:
                    core = tn.einsum('...i,ijk->...jk', core, c)
                    so_far *= c.shape[1]
                if so_far == original_shape[k]:
                    core = tn.reshape(
                        core, [core.shape[0], -1, core.shape[-1]])
                    cores_new.append(core)
                    core = None
                    k += 1
            if k != len(original_shape):
                raise ShapeMismatch('Mode sizes do not match.')
        return TT(cores_new)

    def mprod(self, factor_matrices, mode):
        """
        n-mode product.

        Args:
            factor_matrices (torch.tensor or list[torch.tensor]): either a single matrix is directly provided or a list of matrices for product along multiple modes.
            mode (int or list[int]): the mode for the product. If factor_matrices is a torch.tensor then mode is an integer and the multiplication will be performed along a single mode.
                                     If factor_matrices is a list, the mode has to be list[int] of equal size.

        Raises:
            InvalidArguments: Invalid arguments.
            ShapeMismatch: The n-th mode of the tensor must be equal with the 2nd mode of the matrix.
            IncompatibleTypes: n-model product works only with TT-tensors and not TT matrices.

        Returns:
            torchtt.TT: the result
        """
        if self.__is_ttm:
            raise IncompatibleTypes(
                "n-model product works only with TT-tensors and not TT matrices.")

        if isinstance(factor_matrices, list) and isinstance(mode, list):
            cores_new = [c.clone() for c in self.cores]
            for i in range(len(factor_matrices)):
                if cores_new[mode[i]].shape[1] != factor_matrices[i].shape[1]:
                    raise ShapeMismatch(
                        "The n-th mode of the tensor must be equal with the 2nd mode of the matrix.")
                # if self.__is_ttm else tn.einsum('ijk,lj->ilk',cores_new[mode[i]],factor_matrices[i])
                cores_new[mode[i]] = tn.einsum(
                    'ijk,lj->ilk', cores_new[mode[i]], factor_matrices[i])
        elif isinstance(mode, int) and tn.is_tensor(factor_matrices):
            cores_new = [c.clone() for c in self.cores]
            if cores_new[mode].shape[1] != factor_matrices.shape[1]:
                raise ShapeMismatch(
                    "The n-th mode of the tensor must be equal with the 2nd mode of the matrix.")
            # if self.__is_ttm else tn.einsum('ijk,lj->ilk',cores_new[mode],factor_matrices)
            cores_new[mode] = tn.einsum(
                'ijk,lj->ilk', cores_new[mode], factor_matrices)
        else:
            raise InvalidArguments('Invalid arguments.')

        return TT(cores_new)

    def conj(self):
        """
        Return the complex conjugate of a tensor in TT format.

        Returns:
            torchtt.TT: the complex conjugated tensor.
        """
        return TT([tn.conj(c) for c in self.cores])
