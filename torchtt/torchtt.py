"""
Basic class for TT decomposition.

@author: ion
"""

import torch as tn
import torch.nn.functional as tnf
from torchtt.decomposition import mat_to_tt, to_tt, lr_orthogonal, round_tt, rl_orthogonal, QR
from  torchtt.division import amen_divide
import numpy as np
import math 
from torchtt.dmrg import dmrg_matvec
from torchtt.aux_ops import apply_mask, dense_matvec
from torchtt.errors import *

class TT():
    
    def __init__(self, source, shape=None, eps=1e-10, rmax=2048):
        """
        Constructor of the TT class. Can convert full tensor in the TT-format (from torch.tensor or numpy.array).
        In the case of tensor operators of size M1 x ... Md x N1 x ... x Nd, the shape must be specified as a list of tuples [(M1,N1),...,(Md,Nd)].
        A TT-object can also be computed from cores if the list of cores is passd asa argument.
        If None is provided, an empty tensor is created.

        Args:
            source (torch.tensor ot list[torch.tensor] or numpy.array or None): The inout tensor in full format or the cores. If a torch.tensor or numpy array is provided
            shape (list[int] or list[tuple[int]], optional): The shape (if it differs from the one provided). For the TT-matrix case is mandatory. Defaults to None.
            eps (float, optional): Tolerance of the TT approximation.. Defaults to 1e-10.
            rmax (int or list[int], optional): Maximum rank (either a list of integer or an integer). Defaults to 1000.

        Raises:
            Exception: Invalid input: rank mismatch.
            Exception: [description]
            Exception: [description]
            Exception: [description]
        """
       
        
        if source is None:
            # empty TT
            self.cores = []
            self.M = []
            self.N = []
            self.R = [1,1]
            self.is_ttm = False
            
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
                    raise Exception("Invalid input: rank mismatch.")
                if len(s) == 3:
                    R.append(s[2])
                    N.append(s[1])
                elif len(s)==4:
                    R.append(s[3])
                    M.append(s[1])
                    N.append(s[2])
                else:
                    raise Exception("Invalid input: TT-cores have to be either 4d or 3d")
            
            if len(N) != d or len(R) != d+1 or R[0] != 1 or R[-1] != 1 or (len(M)!=0 and len(M)!=len(N)) :
                raise Exception("Invalid input")
            
            self.cores = source
            self.R = R
            self.N = N
            if len(M) == len(N):
                self.M = M
                self.is_ttm = True
            else:
                self.is_ttm = False
            self.shape = [ (m,n) for m,n in zip(self.M,self.N) ] if self.is_ttm else [n for n in self.N]     

        elif tn.is_tensor(source):
            if shape == None:
                # no size is given. Deduce it from the tensor. No TT-matrix in this case.
                self.N = list(source.shape)
                if len(self.N)>1:
                    self.cores, self.R = to_tt(source,self.N,eps,rmax,is_sparse=False)
                else:    
                    self.cores = [tn.reshape(source,[1,self.N[0],1])]
                    self.R = [1,1]
                self.is_ttm = False
            elif isinstance(shape,list) and isinstance(shape[0],tuple):
                # if the size contains tuples, we have a TT-matrix.
                if len(shape) > 1:
                    self.M = [s[0] for s in shape]
                    self.N = [s[1] for s in shape]
                    self.cores, self.R = mat_to_tt(source, self.M, self.N, eps, rmax)
                    self.is_ttm = True
                else:
                    self.M = [shape[0][0]]
                    self.N = [shape[0][1]]
                    self.cores, self.R = [tn.reshape(source,[1,shape[0][0],shape[0][1],1])], [1,1]
                    self.is_ttm = True
            else:
                # TT-decomposition with prescribed size
                # perform reshape first
                self.N = shape
                self.cores, self.R = to_tt(tn.reshape(source,shape),self.N,eps,rmax,is_sparse=False)
                self.is_ttm = False
            self.shape = [ (m,n) for m,n in zip(self.M,self.N) ] if self.is_ttm else [n for n in self.N]     

        elif isinstance(source, np.ndarray):
            source = tn.tensor(source) 
                    
            if shape == None:
                # no size is given. Deduce it from the tensor. No TT-matrix in this case.
                self.N = list(source.shape)
                if len(self.N)>1:
                    self.cores, self.R = to_tt(source,self.N,eps,rmax,is_sparse=False)
                else:    
                    self.cores = [tn.reshape(source,[1,self.N[0],1])]
                    self.R = [1,1]
                self.is_ttm = False
            elif isinstance(shape,list) and isinstance(shape[0],tuple):
                # if the size contains tuples, we have a TT-matrix.
                self.M = [s[0] for s in shape]
                self.N = [s[1] for s in shape]
                self.cores, self.R = mat_to_tt(source, self.M, self.N, eps, rmax)
                self.is_ttm = True
            else:
                # TT-decomposition with prescribed size
                # perform reshape first
                self.N = shape
                self.cores, self.R = to_tt(tn.reshape(source,shape),self.N,eps,rmax,is_sparse=False)
                self.is_ttm = False
            self.shape = [ (m,n) for m,n in zip(self.M,self.N) ] if self.is_ttm else [n for n in self.N]     
        else:
            raise NotImplementedError("Function only implemented for torch tensors, numpy arrays, list of cores as torch tensors and None.")

    def cuda(self, device = None):
        """
        Return a tensor on the CUDA device.

        Args:
            device (torch.device, optional): The CUDA device. Defaults to None.

        Returns:
            TT-oject: The TT-object. The TT-cores are on CUDA.
        """
         
        t = TT(None)
        t.N = self.N.copy()
        t.R = self.R.copy()
        t.is_ttm = self.is_ttm
        if self.is_ttm:
            t.M = self.M.copy()
        t.cores = [ c.cuda(device) for c in self.cores]

        return t

    def cpu(self):
        """
        Retrive the cores from the GPU.

        Returns:
            TT-object: The TT-object on CPU.
        """

        t = TT(None)
        t.N = self.N.copy()
        t.R = self.R.copy()
        t.is_ttm = self.is_ttm
        if self.is_ttm:
            t.M = self.M.copy()
        t.cores = [ c.cpu() for c in self.cores]

        return t

    def is_cuda(self):
        """
        Return True if the tensor is on GPU.

        Returns:
            bool: Is the tenro on GPU or not.
        """

        return all([c.is_cuda for c in self.core])

    
    def to(self, device = None, dtype = None):
        """
        Moves the TT instance to the given device with the given dtype.

        Args:
            device (torch.device, optional): The desired device. If none is provided, the device is the CPU. Defaults to None.
            dtype (torch.dtype, optional): The desired dtype (torch.float64, torch.float32,...). If none is provided the dtype is not changed. Defaults to None.
        """
        t = TT(None)
        t.N = self.N.copy()
        t.R = self.R.copy()
        t.is_ttm = self.is_ttm
        if self.is_ttm:
            t.M = self.M.copy()
        t.cores = [ c.to(device=device,dtype=dtype) for c in self.cores]

        return t
        
    def full(self):       
        """
        Return the full tensor.
        In case of a TTM, the result has the shape M1 x M2 x ... x Md x N1 x N2 x ... x Nd.

        Returns:
            torch.tensor: the full tensor.
        """
        if self.is_ttm:
            # the case of tt-matrix
            tfull = self.cores[0][0,:,:,:]
            for i in  range(1,len(self.cores)-1) :
                tfull = tn.einsum('...i,ijkl->...jkl',tfull,self.cores[i])
            if len(self.N) != 1:
                tfull = tn.einsum('...i,ijk->...jk',tfull,self.cores[-1][:,:,:,0])
                tfull = tn.permute(tfull,list(np.arange(len(self.N))*2)+list(np.arange(len(self.N))*2+1))
            else:
                tfull = tfull[:,:,0]
        else:
            # the case of a normal tt
            tfull = self.cores[0][0,:,:]
            for i in  range(1,len(self.cores)-1) :
                tfull = tn.einsum('...i,ijk->...jk',tfull,self.cores[i])
            if len(self.N) != 1:
                tfull = tn.einsum('...i,ij->...j',tfull,self.cores[-1][:,:,0])
            else:
                tfull = tn.squeeze(tfull)
        return tfull
    
    def numpy(self):
        """
        Return the full tensor as a numpy.array.
        In case of a TTM, the result has the shape M1 x M2 x ... x Md x N1 x N2 x ... x Nd.
        If it is involved in an AD graph, an error will occur.
        
        Returns:
            numpy.array: the full tensor in numpy.
        """
        return self.full().cpu().numpy()
    
    def __repr__(self):
        """
        Prints the instance of a tensor.

        Returns
        -------
        output : string
            The output that needs to be printed.

        """
        
        
        if self.is_ttm:
            output = 'TT-matrix' 
            output += ' with sizes and ranks:\n'
            output += 'M = ' + str(self.M) + '\nN = ' + str(self.N) + '\n'
            output += 'R = ' + str(self.R) + '\n'
            output += 'Device: '+str(self.cores[0].device)+', dtype: '+str(self.cores[0].dtype)+'\n'
            entries = sum([tn.numel(c)  for c in self.cores])
            output += '#entries ' + str(entries) +' compression ' + str(entries/np.prod(np.array(self.N,dtype=np.float64)*np.array(self.M,dtype=np.float64))) +  '\n'
        else:
            output = 'TT'
            output += ' with sizes and ranks:\n'
            output += 'N = ' + str(self.N) + '\n'
            output += 'R = ' + str(self.R) + '\n\n'
            output += 'Device: '+str(self.cores[0].device)+', dtype: '+str(self.cores[0].dtype)+'\n'
            entries = sum([tn.numel(c) for c in self.cores])
            output += '#entries ' + str(entries) +' compression '  + str(entries/np.prod(np.array(self.N,dtype=np.float64))) + '\n'
        
        return output
    
    def __radd__(self,other):
        """
        Addition A+B in TT format. This is the case when we add objects that are not TT with TT-objects:
            scalar + T
        
        Parameters
        ----------
        other : TT-tensor/TT-matrix or a scalar
            second operand.

        Raises
        ------
        Exception
            If the inputs are not compatible, an error is thrown.

        Returns
        -------
        result : TT object.
            The result. The rank of the output is the sum of the input ranks.

        """
        
        return self.__add__(other)

    def __add__(self,other):
        """
        Addition A+B in TT format. The following type pairs are supported:
            - both A and B are TT-tensors.
            - both A and B are TT-matrices.
            - A is a TT-tensor or a TT-matrix and B is a scalar (either torch or numpy).
        
        Parameters
        ----------
        other : TT-tensor/TT-matrix or a scalar
            second operand.

        Raises
        ------
        Exception
            If the inputs are not compatible, an error is thrown.

        Returns
        -------
        result : TT object.
            The result. The rank of the output is the sum of the input ranks.

        """
        if np.isscalar(other) or ( tn.is_tensor(other) and tn.numel(other) == 1):
            # the second term is a scalar
            cores =  []
            
            for i in range(len(self.N)):
                if self.is_ttm:
                    pad1 = (0,0 if i == len(self.N)-1 else 1 , 0,0 , 0,0 , 0,0 if i==0 else 1)
                    pad2 = (0 if i == len(self.N)-1 else self.R[i+1],0 , 0,0 , 0,0 , 0 if i==0 else self.R[i],0)
                    othr = tn.ones([1,1,1,1],dtype=self.cores[i].dtype) * (other if i ==0 else 1)
                else:
                    pad1 = (0,0 if i == len(self.N)-1 else 1 , 0,0 , 0,0 if i==0 else 1)
                    pad2 = (0 if i == len(self.N)-1 else self.R[i+1],0 , 0,0 , 0 if i==0 else self.R[i],0)
                    othr = tn.ones([1,1,1],dtype=self.cores[i].dtype) * (other if i ==0 else 1)
                

                cores.append(tnf.pad(self.cores[i],pad1)+tnf.pad(othr,pad2))

                
            result = TT(cores)
        elif isinstance(other,TT):
        #second term is TT object 
            if self.is_ttm and other.is_ttm:
                # both are TT-matrices
                if self.M != self.M or self.N != self.N:
                    raise Exception('Dimension mismatch.')
                    
                cores = []
                for i in range(len(self.N)):
                    pad1 = (0,0 if i == len(self.N)-1 else other.R[i+1], 0,0 , 0,0 , 0,0 if i==0 else other.R[i])
                    pad2 = (0 if i == len(self.N)-1 else self.R[i+1],0 , 0,0 , 0,0 , 0 if i==0 else self.R[i],0)
                    cores.append(tnf.pad(self.cores[i],pad1)+tnf.pad(other.cores[i],pad2))
                    
                result = TT(cores)
                
            elif self.is_ttm==False and other.is_ttm==False:
                # normal tensors in TT format.
                if self.N != self.N:
                    raise Exception('Dimension mismatch.')
                    
                cores = []
                for i in range(len(self.N)):
                    pad1 = (0,0 if i == len(self.N)-1 else other.R[i+1] , 0,0 , 0,0 if i==0 else other.R[i])
                    pad2 = (0 if i == len(self.N)-1 else self.R[i+1],0 , 0,0 , 0 if i==0 else self.R[i],0)
                    cores.append(tnf.pad(self.cores[i],pad1)+tnf.pad(other.cores[i],pad2))
                    
                    
                result = TT(cores)
                
                
            else:
                # incompatible types 
                raise Exception('Addition between a tensor and a matrix is not defined.')
        else:
            Exception('Second term is incompatible.')
            
        return result
    
    def __rsub__(self,other):
        """
        substraction A-B in TT format. The following type pairs are supported:
            - both A and B are TT-tensors.
            - both A and B are TT-matrices.
            - A is a TT-tensor or a TT-matrix and B is a scalar (either torch or numpy).
        
        Parameters
        ----------
        other : TT-tensor/TT-matrix or a scalar
            DESCRIPTION.

        Raises
        ------
        Exception
            If the inputs are not compatible, an error is thrown.

        Returns
        -------
        result : TT object.
            The result. The rank of the output is the sum of the input ranks.

        """
        T = self.__sub__(other)
        T.cores[0] = -T.cores[0]
        return T
    
    def __sub__(self,other):
        """
        substraction A-B in TT format. The following type pairs are supported:
            - both A and B are TT-tensors.
            - both A and B are TT-matrices.
            - A is a TT-tensor or a TT-matrix and B is a scalar (either torch or numpy).
        
        Parameters
        ----------
        other : TT-tensor/TT-matrix or a scalar
            DESCRIPTION.

        Raises
        ------
        Exception
            If the inputs are not compatible, an error is thrown.

        Returns
        -------
        result : TT object.
            The result. The rank of the output is the sum of the input ranks.

        """
        if np.isscalar(other) or ( tn.is_tensor(other) and other.shape == []):
            # the second term is a scalar
            cores =  []
            
            for i in range(len(self.N)):
                if self.is_ttm:
                    pad1 = (0,0 if i == len(self.N)-1 else 1 , 0,0 , 0,0 , 0,0 if i==0 else 1)
                    pad2 = (0 if i == len(self.N)-1 else self.R[i+1],0 , 0,0 , 0,0 , 0 if i==0 else self.R[i],0)
                    othr = tn.ones([1,1,1,1],dtype=self.cores[i].dtype) * (-other if i ==0 else 1)
                else:
                    pad1 = (0,0 if i == len(self.N)-1 else 1 , 0,0 , 0,0 if i==0 else 1)
                    pad2 = (0 if i == len(self.N)-1 else self.R[i+1],0 , 0,0 , 0 if i==0 else self.R[i],0)
                    othr = tn.ones([1,1,1],dtype=self.cores[i].dtype) * (-other if i ==0 else 1)
                cores.append(tnf.pad(self.cores[i],pad1)+tnf.pad(othr,pad2))
            result = TT(cores)

        elif isinstance(other,TT):
        #second term is TT object 
            if self.is_ttm and other.is_ttm:
                # both are TT-matrices
                if self.M != self.M or self.N != self.N:
                    raise Exception('Dimension mismatch.')
                    
                cores = []
                for i in range(len(self.N)):
                    pad1 = (0,0 if i == len(self.N)-1 else other.R[i+1] , 0,0 , 0,0 , 0,0 if i==0 else other.R[i])
                    pad2 = (0 if i == len(self.N)-1 else self.R[i+1],0 , 0,0 , 0,0 , 0 if i==0 else self.R[i],0)
                    cores.append(tnf.pad(self.cores[i],pad1)+tnf.pad(-other.cores[i] if i==0 else other.cores[i],pad2))
                    
                result = TT(cores)
                
            elif self.is_ttm==False and other.is_ttm==False:
                # normal tensors in TT format.
                if self.N != self.N:
                    raise Exception('Dimension mismatch.')
                    
                cores = []
                for i in range(len(self.N)):
                    pad1 = (0,0 if i == len(self.N)-1 else other.R[i+1] , 0,0 , 0,0 if i==0 else other.R[i])
                    pad2 = (0 if i == len(self.N)-1 else self.R[i+1],0 , 0,0 , 0 if i==0 else self.R[i],0)
                    cores.append(tnf.pad(self.cores[i],pad1)+tnf.pad(-other.cores[i] if i==0 else other.cores[i],pad2))
                    
                    
                result = TT(cores)
                
                
            else:
                # incompatible types 
                raise Exception('Addition between a tensor and a matrix is not defined.')
        else:
            Exception('Second term is incompatible.')
            
        return result
    
    def __rmul__(self,other):
        """
        Elementwise multiplication in the TT-format from the right.

        Parameters
        ----------
        other : TT-object or scalar.
            Second operand.

        Raises
        ------
        Exception
            If dimension is wrong or objects are not of the same type.

        Returns
        -------
        result : TT-object.
            The result.

        """
        
        return self.__mul__(other)
        
    def __mul__(self,other):
        """
        Elementwise multiplication in the TT-format.

        Parameters
        ----------
        other : TT-object or scalar.
            Second operand.

        Raises
        ------
        Exception
            If dimension is wrong or objects are not of the same type.

        Returns
        -------
        result : TT-object.
            The result.

        """
         # elementwise multiplication
        if isinstance(other, TT):
            if self.is_ttm and other.is_ttm:
                if self.N != other.N or self.M != other.M:
                    raise Exception('Dimension mismatch.') 
                    
                cores_new = []
                
                for i in range(len(self.cores)):
                    core = tn.reshape(tn.einsum('aijb,mijn->amijbn',self.cores[i],other.cores[i]),[self.R[i]*other.R[i],self.M[i],self.N[i],self.R[i+1]*other.R[i+1]])
                    cores_new.append(core)
    
            elif self.is_ttm == False and other.is_ttm == False:
                if self.N != other.N:
                    raise Exception('Dimension mismatch.')
                    
                cores_new = []
                
                for i in range(len(self.cores)):
                    core = tn.reshape(tn.einsum('aib,min->amibn',self.cores[i],other.cores[i]),[self.R[i]*other.R[i],self.N[i],self.R[i+1]*other.R[i+1]])
                    cores_new.append(core)
            else:
                raise Exception('Operation defined between objects of the same type.')
        elif isinstance(other,int) or isinstance(other,float) or isinstance(other,tn.tensor):
            cores_new = [c+0 for c in self.cores]
            cores_new[0] *= other
        else:
            raise Exception('Second operand must be of type: TT, float, int of tensorflow Tensor')
        result = TT(cores_new)            
        return result
        
    def __matmul__(self,other):
        """
        Matrix-vector multiplication in TT-format
        Suported operands:
            - TT-matrix @ TT-tensor -> TT-tensor: y_i = A_ij * x_j
            - TT-tensor @ TT-matrix -> TT-tensor: y_j = x_i * A_ij 
            - TT-matrix @ TT-matrix -> TT-matrix: Y_ij = A_ik * B_kj
            - TT-matrix @ torch.tensor -> torch.tensor: y_bi = A_ij * x_bj 
        In the last case, the multiplication is performed along the last modes.

        Args:
            other (torchtt.TT or torch.tensor): the second operand.

        Raises:
            Exception: [description]
            Exception: [description]
            Exception: [description]
            Exception: [description]
            Exception: [description]

        Returns:
            torchtt.TT or torch.tensor: the result. Can be full tensor if the second operand is full tensor.
        """
     
        if self.is_ttm and tn.is_tensor(other):
            if self.N != list(other.shape)[-len(self.N):]:
                raise Exception("Dimension mismatch.")
            result = dense_matvec(self.cores,other) 
            return result

        elif self.is_ttm and other.is_ttm == False:
            # matrix-vector multiplication
            if self.N != other.N:
                raise Exception('Dimension mismatch.')
                
            cores_new = []
            
            for i in range(len(self.cores)):
                core = tn.reshape(tn.einsum('ijkl,mkp->imjlp',self.cores[i],other.cores[i]),[self.cores[i].shape[0]*other.cores[i].shape[0],self.cores[i].shape[1],self.cores[i].shape[3]*other.cores[i].shape[2]])
                cores_new.append(core)
            
            
        elif self.is_ttm and other.is_ttm:
            # multiplication between 2 TT-matrices
            if self.N != other.M:
                raise Exception('Dimension mismatch')
                
            cores_new = []
            
            for i in range(len(self.cores)):
                core = tn.reshape(tn.einsum('ijkl,mknp->imjnlp',self.cores[i],other.cores[i]),[self.cores[i].shape[0]*other.cores[i].shape[0],self.cores[i].shape[1],other.cores[i].shape[2],self.cores[i].shape[3]*other.cores[i].shape[3]])
                cores_new.append(core)
        elif self.is_ttm == False and other.is_ttm:
            # vector-matrix multiplication
            if self.N != other.M:
                raise Exception('Dimension mismatch')
                
            cores_new = []
            
            for i in range(len(self.cores)):
                core = tn.reshape(tn.einsum('mkp,ikjl->imjlp',self.cores[i],other.cores[i]),[self.cores[i].shape[0]*other.cores[i].shape[0],other.cores[i].shape[2],self.cores[i].shape[2]*other.cores[i].shape[3]])
                cores_new.append(core)
        else:
            raise Exception('Operation undefined between 2 TT-tensors')
            
        result = TT(cores_new)
        return result

    def fast_matvec(self,other, eps = 1e-12, nswp = 20, verb = False):
        '''
        Fast matrix vector multiplication A@x using DMRG iterations. Faster than traditional matvec + rounding.
        
        Parameters
        ----------
        other : TT tensor.
            DESCRIPTION.
        eps : float, optional
            The epsilon accuracy. The default is 1e-12.
        nswp : int, optional
            Number of sweeps. The default is 20.
        verb : bool, optional
            show additional info. The default is False.

        Raises
        ------
        Exception
            Operands mismatch.

        Returns
        -------
        TT tensor
            The result.

        '''
        if not isinstance(other,TT):
            raise Exception('Second operand has to be TT object.')
        if not self.is_ttm or other.is_ttm:
            raise Exception('First operand should be a TT matrix and second a TT vector.')
            
        return dmrg_matvec(self, other, eps = eps, verb = verb, nswp = nswp)

    def apply_mask(self,indices):
        """
        Evaluate the tensor on the given index list.

        Example:
        x = torchtt.random([10,12,14],[1,4,5,1])
        indices = torch.tensor([[0,0,0],[1,2,3],[1,1,1]])
        val = x.apply_mask(indices)

        Args:
            indices (list[list[int]]): the index list where the tensor should be evaluated. Length is M.

        Returns:
            (torch.tensor): the values of the tensor
        """
        result = apply_mask(self.cores,self.R,indices)
        return result

    def __truediv__(self,other):
        '''
        Elementwise division of a TT-object by a scalar.

        Parameters
        ----------
        other : scalar (float, int of torch scalar)
            

        Raises
        ------
        Exception
            Operand not permitted. A TT-object can be divided only with scalars.

        Returns
        -------
        result : TT-object
            The result.

        '''
        if isinstance(other,int) or isinstance(other,float) or tn.is_tensor(other):
            # divide by a scalar
            cores_new = self.cores.copy()
            cores_new[0] /= other
            result = TT(cores_new)
        elif isinstance(other,TT):
            if self.is_ttm != other.is_ttm:
                raise Exception('Operands should be either TT or TTM.')
            if self.N != other.N or (self.is_ttm and self.M != other.M):
                raise ShapeMismatch("Both operands should have the same shape.")
            result = TT(amen_divide(other,self,50,None,1e-12,500,verbose=False))       
        else:
            raise Exception('Operand not permitted. A TT-object can be divided only with scalars.')
            
       
        return result
    
    def __rtruediv__(self,other):
        """
        Right true division. this function is called when a non TT object is divided by a TT object.
        Example: z = 1.0/x # x is TT instance

        Args:
            other (float,int or ): [description]

        Returns:
            torchtt.TT: the result.
        """
        if isinstance(other,int) or isinstance(other,float) or ( tn.is_tensor(other) and other.numel()==1):
            o = ones(self.N,dtype=self.cores[0].dtype,device = self.cores[0].device)
            o.cores[0] *= other
            cores_new = amen_divide(self,o,50,None,1e-12,500,verbose=False)
        else:
            raise Exception("The first operand must be int, float or 1d torch.tensor")   
         
        return TT(cores_new)

    
    
    def t(self):
        
        if not self.is_ttm:
            raise Exception('Has to TT-operator.')
            
        cores_new = [tn.permute(c,[0,2,1,3]) for c in self.cores]
        
        return TT(cores_new)
        
    
    def norm(self,squared=False):
        '''
        Computes the frobenius norm of TT-objects

        Returns
        -------
        norm : float
            The frobenius norm of the given TT.

        '''

        if any([c.requires_grad or c.grad_fn != None for c in self.cores]):
            norm = tn.tensor([[1.0]],dtype = self.cores[0].dtype, device=self.cores[0].device)
            
            if self.is_ttm:
                for i in range(len(self.N)):
                    norm = tn.einsum('ab,aijm,bijn->mn',norm, self.cores[i], self.cores[i])
                norm = tn.squeeze(norm)
            else:
                           
                for i in range(len(self.N)):
                    norm = tn.einsum('ab,aim,bin->mn',norm, self.cores[i], self.cores[i])
                norm = tn.squeeze(norm)
            if squared:
                return norm
            else:
                return tn.sqrt(tn.abs(norm))
 
        else:        
            d = len(self.cores)

            core_now = self.cores[0]
            for i in range(d-1):
                if self.is_ttm:
                    mode_shape = [core_now.shape[1],core_now.shape[2]]
                    core_now = tn.reshape(core_now,[core_now.shape[0]*core_now.shape[1]*core_now.shape[2],-1])
                else:
                    mode_shape = [core_now.shape[1]]
                    core_now = tn.reshape(core_now,[core_now.shape[0]*core_now.shape[1],-1])
                    
                # perform QR
                Qmat, Rmat = QR(core_now)
                     
                # take next core
                core_next = self.cores[i+1]
                shape_next = list(core_next.shape[1:])
                core_next = tn.reshape(core_next,[core_next.shape[0],-1])
                core_next = Rmat @ core_next
                core_next = tn.reshape(core_next,[Qmat.shape[1]]+shape_next)
                
                # update the cores
                
                core_now = core_next
            if squared:
                return tn.linalg.norm(core_next)**2
            else:
                return tn.linalg.norm(core_next)

    def sum(self,index = None):
        """
        Cumputes the sum. If a list of modes is given, the sum is computed w.r.t. the given modes.

        Parameters
        ----------
        index : list of integers, optional
            The list of modes. Pass None or -1 in order to syum over all modes. The default is None.

        Returns
        -------
        S : TT-tensor or double.
            IF no index is given, a scalar is reurned, otherwise a tensor.

        """
        if index != None and isinstance(index,int):
            index = [index]
        if not isinstance(index,list) and index != None:
            raise Exception('Invalid index.')
             
        if index == None: 
            # the case we need to sum over all modes
            if self.is_ttm:
                C = tn.reduce_sum(self.cores[0],[0,1,2])
                for i in range(1,len(self.N)):
                    C = tn.sum(tn.einsum('i,ijkl->jkl',C,self.cores[i]),[0,1])
                S = tn.sum(C)
            else:
                C = tn.sum(self.cores[0],[0,1])
                for i in range(1,len(self.N)):
                    C = tn.sum(tn.einsum('i,ijk->jk',C,self.cores[i]),0)
                S = tn.sum(C)
        else:
            # we return the TT-tensor with summed indices
            cores = []
            
            if self.is_ttm:
                tmp = [1,2]
            else:
                tmp = [1]
                
            for i in range(len(self.N)):
                if i in index:
                    C = tn.sum(self.cores[i], tmp, keepdim = True)
                    cores.append(C)
                else:
                    cores.append(self.cores[i])
                        
            S = TT(cores)
            S.reduce_dims()
            
        return S

    def to_ttm(self):
        """
        Converts a TT-tensor to the TT-matrix format. In the tensor has the shape N1 x ... x Nd, the result has the shape 
        N1 x ... x Nd x 1 x ... x 1.
    
        Returns
        -------
        TT-tensor. 
            the result

        """

        cores_new = [tn.reshape(c,(c.shape[0],c.shape[1],1,c.shape[2])) for c in self.cores]
        return TT(cores_new)

    def reduce_dims(self):
        """
        Reduces the size 1 modes of the TT-object.
        At least one mode should be larger than 1.
        Returns
        -------
        None.

        """
        
        # TODO: implement a version that reduces the rank also. by spliting the cores with modes 1 into 2 using the SVD.
        
        if self.is_ttm:
            cores_new = []
            
            for i in range(len(self.N)):
                
                if self.cores[i].shape[1] == 1 and self.cores[i].shape[2] == 1:
                    if self.cores[i].shape[0] > self.cores[i].shape[3] or i == len(self.N)-1:
                        # multiply to the left
                        if len(cores_new) > 0:
                            cores_new[-1] = tn.einsum('ijok,kl->ijol',cores_new[-1], self.cores[i][:,0,0,:])
                        else: 
                            # there is no core to the left. Multiply right.
                            if i != len(self.N)-1:
                                self.cores[i+1] = tn.einsum('ij,jkml->ikml', self.cores[i][:,0,0,:],self.cores[i+1])
                            else:
                                cores_new.append(self.cores[i])
                            
                    else:
                        # multiply to the right. Set the carry 
                        self.cores[i+1] = tn.einsum('ij,jkml->ikml',self.cores[i][:,0,0,:],self.cores[i+1])
                        
                else:
                    cores_new.append(self.cores[i])
                    
            # update the cores and ranks and shape
            self.N = []
            self.M = []
            self.R = [1]
            for i in range(len(cores_new)):
                self.N.append(cores_new[i].shape[1])
                self.M.append(cores_new[i].shape[2])
                self.R.append(cores_new[i].shape[3])
            self.cores = cores_new
        else:
            cores_new = []
            
            for i in range(len(self.N)):
                
                if self.cores[i].shape[1] == 1:
                    if self.cores[i].shape[0] > self.cores[i].shape[2] or i == len(self.N)-1:
                        # multiply to the left
                        if len(cores_new) > 0:
                            cores_new[-1] = tn.einsum('ijk,kl->ijl',cores_new[-1], self.cores[i][:,0,:])
                        else: 
                            # there is no core to the left. Multiply right.
                            if i != len(self.N)-1:
                                self.cores[i+1] = tn.einsum('ij,jkl->ikl', self.cores[i][:,0,:],self.cores[i+1])
                            else:
                                cores_new.append(self.cores[i])
                                
                            
                    else:
                        # multiply to the right. Set the carry 
                        self.cores[i+1] = tn.einsum('ij,jkl->ikl',self.cores[i][:,0,:],self.cores[i+1])
                        
                else:
                    cores_new.append(self.cores[i])
            
            
            # update the cores and ranks and shape
            self.N = []
            self.R = [1]
            for i in range(len(cores_new)):
                self.N.append(cores_new[i].shape[1])
                self.R.append(cores_new[i].shape[2])
            self.cores = cores_new
                    
                    
        self.shape = [ (m,n) for m,n in zip(self.M,self.N) ] if self.is_ttm else [n for n in self.N] 
        
    def __getitem__(self,index):
        """
        Returns a slice of a tensor. Similar to torch slicing.

        Parameters
        ----------
        index : tuple of slices / integers / Ellipsis.
            

        Raises
        ------
        Exception
            Raises exception if slice is invalid.

        Returns
        -------
        sliced : TT object or scalar
            the resulting sliced TT tensor. If the result has all mode sizes of 1, a scalar is returned.

        """
        
        
        # slicing function
        
        ##### TODO: include Ellipsis support.
        
        # if a slice containg integers is passed, an element is returned
        # if ranged slices are used, a TT-object has to be returned.
        
        if isinstance(index,tuple):
            # check if more than one Ellipsis are to be found.
            if index.count(Ellipsis) > 0:
                raise Exception('Ellipsis are not supported.')
            if self.is_ttm:
                if len(index) != len(self.N)*2:
                    raise Exception('Slice size is invalid.')
                    
                cores_new = []
                for i in range(len(self.cores)):
                    # cores_new.append(self.cores[i][:,index[i],index[i+len(self.N)],:])
                    if isinstance(index[i],slice):
                        cores_new.append(self.cores[i][:,index[i],index[i+len(self.N)],:])
                    else:
                        cores_new.append(tn.reshape(self.cores[i][:,index[i],index[i+len(self.N)],:],[self.R[i],1,1,self.R[i+1]]))
               
                
            else:
                if len(index) != len(self.N):
                    raise Exception('Slice size is invalid.')
                    
                cores_new = []
                for i in range(len(self.cores)):
                    if isinstance(index[i],slice):
                        cores_new.append(self.cores[i][:,index[i],:])
                    else:
                        cores_new.append(tn.reshape(self.cores[i][:,index[i],:],[self.R[i],-1,self.R[i+1]]))
            
            sliced = TT(cores_new)
            sliced.reduce_dims()
            if (sliced.is_ttm == False and sliced.N == [1]) or (sliced.is_ttm and sliced.N == [1] and sliced.M == [1]):
                sliced = tn.squeeze(sliced.cores[0])
                
                
            # cores = None
            
            
        elif isinstance(index,int):
            # tensor is 1d and one element is retrived
            if len(self.N) == 1:
                sliced = self.cores[0][0,index,0]
            else:
                raise Exception('Invalid slice. Tensor is not 1d.')
                
            ## TODO
        elif isinstance(index,Ellipsis):
            # return a copy of the tensor
            sliced = TT(self.cores)
            
        elif isinstance(index,slice):
            # tensor is 1d and one slice is extracted
            if len(self.N) == 1:
                sliced = TT(self.cores[0][:,index,:])
            else:
                raise Exception('Invalid slice. Tensor is not 1d.')
            ## TODO
        else:
            raise Exception('Invalid slice.')
            
        
        return sliced
    
    def __pow__(self,other):
        """
        Implements the tensor Kronecker product using the POWER operator.
        Example: C = A ** B
        (see kron function for details)
        
        Parameters
        ----------
        other : TT-object
            The second operand.

        Returns
        -------
        result : TT-object
            The resulting object.

        """
        
        result = kron(self,other)
        
        return result
    
    def __rpow__(self,other):
        """
        Implements the tensor Kronecker product using the POWER operator.
        Example: C = A ** B
        (see kron function for details)
        
        Parameters
        ----------
        other : TT-object
            The second operand.

        Returns
        -------
        result : TT-object
            The resulting object.

        """
        
        result = kron(self,other)
        
        return result
    
    def __neg__(self):
        '''
        Negates a tensor: -x

        Returns
        -------
        TT-tensor
             the negation of the tensor.

        '''
    
        cores_new = [c+0 for c in self.cores]
        cores_new[0] = -cores_new[0]
        return TT(cores_new)
    
    def __pos__(self):
        '''
        Returns +x for x tensor x.

        Returns
        -------
        TT-tensor
             +x.

        '''
        cores_new = [c for c in self.cores]

        return TT(cores_new)
    
    def round(self, eps=1e-12, rmax = 2048): 
        """
        Rounds a TT-tensor or a TT-matrix with prescribed eps. The maximum rand will not be larger than the given rmax.
        If rmax is a list, the modewise maximum rand is considered.

        Parameters
        ----------
        eps : double, optional
            desired accuracy of the rounding. The default is 1e-12.
        rmax : integer or list of integers, optional
            the given rank. The default is 100.

        Returns
        -------
        T : TT object
            The rounded TT-tensor or TT-matrix.

        """
        
        # rmax is not list
        if not isinstance(rmax,list):
            rmax = [1] + len(self.N)*[rmax] + [1]
            
        # call the round function
        tt_cores, R = round_tt(self.cores, self.R.copy(), eps, rmax,self.is_ttm)
        # creates a new TT and return it
        T = TT(tt_cores)
               
        return T
    
    def to_qtt(self,eps = 1e-12,mode_size=2,rmax=1000):
        '''
        Converts a tensor to the QTT format: N1 x N2 x ... x Nd -> mode_size x mode_size x ... x mode_size.
        The product of the mode sizes should be a power of mode_size.
        The tensor in QTT can be converted back using the qtt_to_tens() method.
        
        Example:
            
            import ttt
            
            x = ttt.random([16,8,64,128],[1,2,10,12,1])
            x_qtt = x.to_qtt()
            print(x_qtt)
            xf = x_qtt.qtt_to_tens(x.N) # a TT-rounding is recommended.
            
        
        Parameters
        ----------
        eps : float, optional
            the maximum truncation error. The default is 1e-12.
        mode_size : int, optional
            the desired mode size. The default is 2.
        rmax : int, optional
            maximum rank. The default is 1000.

        Raises
        ------
        Exception
            Incopatible shapes.

        Returns
        -------
        result : TT-tensor
            The result. 

        '''
        cores_new = []
        if self.is_ttm:
            shape_new = []
            for i in range(len(self.N)):
                if self.N[i]!=self.M[i]:
                    raise Exception('Only quadratic TTM can be tranformed to QTT.')
                if self.N[i]==mode_size**int(math.log(self.N[i],mode_size)):
                    shape_new += [(mode_size,mode_size)]*int(math.log(self.N[i],mode_size))
                else:
                    raise Exception('Reshaping error: check if the dimensions are powers of the desired mode size:\r\ncore size '+str(list(self.cores[i].shape))+' cannot be reshaped.')
                
            result = reshape(self, shape_new, eps, rmax)
        else:
            for core in self.cores:
                if int(math.log(core.shape[1],mode_size))>2:
                    Nnew = [core.shape[0]*mode_size]+[mode_size]*(int(math.log(core.shape[1],mode_size))-2)+[core.shape[2]*mode_size]
                    try:
                        core = tn.reshape(core,Nnew)
                    except:
                        raise Exception('Reshaping error: check if the dimensions care powers of the desired mode size:\r\ncore size '+str(list(core.shape))+' cannot be reshaped to '+str(Nnew))
                    cores,_ = to_tt(core,Nnew,eps,rmax,is_sparse=False)
                    cores_new.append(tn.reshape(cores[0],[-1,mode_size,cores[0].shape[-1]]))
                    cores_new += cores[1:-1]
                    cores_new.append(tn.reshape(cores[-1],[cores[-1].shape[0],mode_size,-1]))
                else: 
                    cores_new.append(core)
            result = TT(cores_new)
            
        return result
               
    def qtt_to_tens(self, original_shape):     
        '''
        Transforms a tensor from QTT back to TT.
        If not, the original size has to be given. 

        Parameters
        ----------
        original_shape : list of integers
            The original shape of the tensor.

        Raises
        ------
        Exception
            Dimensions mismatch.

        Returns
        -------
        TT-tensor
            The result.

        '''
        
        if not isinstance(original_shape,list):
            raise Exception("Original shape must be a list.")

        core = None
        cores_new = []
        
        if self.is_ttm:
            pass
        else:
            k = 0
            for c in self.cores:
                if core==None:
                    core = c
                    so_far = core.shape[1]
                else:
                    core = tn.einsum('...i,ijk->...jk',core,c)
                    so_far *= c.shape[1]
                if so_far==original_shape[k]:
                    core = tn.reshape(core,[core.shape[0],-1,core.shape[-1]])
                    cores_new.append(core)
                    core = None
                    k += 1
            if k!= len(original_shape):
                raise Exception('Mode sizes do not match')
        return TT(cores_new)
    
    def mprod(self, factor_matrices, mode):
        """
        n-mode product.

        Args:
            factor_matrices (torch.tensor or list[torch.tensor]): either a single matrix is directly provided or a list of matrices for product along multiple modes.
            mode (int or list[int]): the mode for the product. If factor_matrices is a torch.tensor then mode is an integer and the multiplication will be performed along a single mode.
                                     If factor_matrices is a list, the mode has to be list[int] of equal size.

        Raises:
            Exception: Invalid arguments.

        Returns:
            torchtt.TT: the result
        """
    
        if isinstance(factor_matrices,list) and isinstance(mode, list):
            cores_new = [c.clone() for c in self.cores]
            for i in range(len(factor_matrices)):
                cores_new[mode[i]] =  tn.einsum('imjk,lj->imlk',cores_new[mode[i]],factor_matrices[i]) if self.is_ttm else tn.einsum('ijk,lj->ilk',cores_new[mode[i]],factor_matrices[i]) 
        elif isinstance(factor_matrices, tn.tensor) and isinstance(mode, int):
            cores_new = [c.clone() for c in self.cores]
            cores_new[mode] =  tn.einsum('imjk,lj->imlk',cores_new[mode],factor_matrices) if self.is_ttm else tn.einsum('ijk,lj->ilk',cores_new[mode],factor_matrices) 
        else:
            raise Exception('Invalid arguments')
        
        return TT(cores_new)        
        
    
def eye(shape, dtype=tn.float64):
    """
    Computes the identity TT-matrix. 
    The identity matrix has equal left and right modes.
    

    Parameters
    ----------
    shape : list of integers.
        the mode sizes.
    dtype : torch dtype, optional
        The dype of the object. The default is tn.float64.

    Returns
    -------
    TT-matrix
        Identity TT matrix.

    """
    
    shape = list(shape)
    
    cores = [tn.unsqueeze(tn.unsqueeze(tn.eye(s, dtype=dtype),0),3) for s in shape]            
    
    return TT(cores)
    
def zeros(shape, dtype=tn.float64, device = None):
    """
    Creates a zero TT-tensor/TT-matrix. 
    The shape is either [N_1,...N_d] for a TT-tensor or [(M_1,N_1),...,(M_d,N_d)] for a TT-matrix, where M_k are the left modex and N_k are the right modes.
    
    Parameters
    ----------
    shape : list
        The shape of the TT object. Must contain integers or tuples of 2 integers corresponding to the size.
    dtype : torch dtype, optional
        The dype of the object. The default is tn.float64.

    Raises
    ------
    Exception
        Invalid shape.

    Returns
    -------
    TT object
        Zero TT object.

    """
    if isinstance(shape,list):
        d = len(shape)
        if isinstance(shape[0],tuple):
            # we create a TT-matrix
            cores = [tn.zeros([1,shape[i][0],shape[i][1],1],dtype=dtype, device = device) for i in range(d)]            
            
        else:
            # we create a TT-tensor
            cores = [tn.zeros([1,shape[i],1],dtype=dtype, device = device) for i in range(d)]
            
    else:
        raise Exception('Invalid shape.')
    
    return TT(cores)
    

def emptyTT():
    
    tens = TT(None)
    tens.is_ttm = False
    tens.N = []
    tens.R = []
    return tens
    
def emptyTTM():
    
    tens = TT(None)
    tens.is_ttm = True
    tens.N = []
    tens.R = []
    return tens
  
def kron(first, second):
    """
    Computes the tensor Kronecker product.
    The inputs are either TT-tensors or TT-matrices.
    If A is N_1 x ... x N_d and B is M_1 x ... x M_p, then kron(A,B) is N_1 x ... x N_d x M_1 x ... x M_p

    Parameters
    ----------
    first : TT-object
        First operand.
    second : TT-object.
        Second operand.

    Raises
    ------
    Exception
        Incompatible data types.

    Returns
    -------
    result : TT-object.
        The result.

    """
   
    if first == None and isinstance(second,TT):
        cores_new = [c.clone() for c in second.cores]
        result = TT(cores_new)
    elif second == None and isinstance(first,TT): 
        cores_new = [c.clone() for c in first.cores]
        result = TT(cores_new)
    elif isinstance(first,TT) and isinstance(second,TT):
        if first.is_ttm != second.is_ttm:
            raise Exception('Incompatible data types (make sure both are either TT-matrices or TT-tensors).')
    
        # concatenate the result
        cores_new = [c.clone() for c in first.cores] + [c.clone() for c in second.cores]
        result = TT(cores_new)
    else:
        raise Exception('Invalid arguments')
    return result



def ones(shape, dtype=tn.float64, device = None):
    """
    Creates a ones TT-tensor/TT-matrix. 
    The shape is either [N_1,...N_d] for a TT-tensor or [(M_1,N_1),...,(M_d,N_d)] for a TT-matrix, where M_k are the left modex and N_k are the right modes.
    
    Parameters
    ----------
    shape : list
        The shape of the TT object. Must contain integers or tuples of 2 integers corresponding to the size.
    dtype : torch dtype, optional
        The dype of the object. The default is tn.float64.

    Raises
    ------
    Exception
        Invalid shape.

    Returns
    -------
    TT object
        Ones TT object.

    """
    if isinstance(shape,list):
        d = len(shape)
        if d==0:
            return TT(None)
        else:
            if isinstance(shape[0],tuple):
                # we create a TT-matrix
                cores = [tn.ones([1,shape[i][0],shape[i][1],1],dtype=dtype,device=device) for i in range(d)]            
                
            else:
                # we create a TT-tensor
                cores = [tn.ones([1,shape[i],1],dtype=dtype,device=device) for i in range(d)]
            
    else:
        raise Exception('Invalid shape.')
    
    return TT(cores)


def random(N, R, dtype = tn.float64, device = None):
    """
    Returns a tensor of shape N with random cores of rank R.

    Parameters
    ----------
    N : list of integers or list of tuples of integers.
        The shape of the tensor. If the elements are tuples of integers, we deal with a TT-matrix.
    R : list of integers or int.
        The rank. Must have the form [1,...,1] or an integer is specified so that the rank will be [1,R,R,...,R,1]
    dtype : tf.dtype, optional
        The type (float32, float64, ... ). The default is tn.float64.

    Raises
    ------
    Exception
        Invalid arguments.

    Returns
    -------
    T : TT-object
        The random tensor.

    """
    if isinstance(R,int):
        R = [1]+[R]*(len(N)-1)+[1]
    elif len(N)+1 != len(R) or R[0] != 1 or R[-1] != 1 or len(N)==0:
        raise Exception('Invalid arguments. Check if N and R are right.')
        
    cores = []
    
    for i in range(len(N)):
        cores.append(tn.randn([R[i],N[i][0],N[i][1],R[i+1]] if isinstance(N[i],tuple) else [R[i],N[i],R[i+1]], dtype = dtype, device = device))
        
    T = TT(cores)
    
    return T

def randn(N, R, var = 1.0, dtype = tn.float64, device = None):
    """
    A torchtt.TT tensor of shape N = [N1 x ... x Nd] and rank R is returned. 
    The entries of the fuill tensor are alomst normal distributed with the variance var.
    
    Args:
        N (list[int]): the shape.
        R (list[int]): the rank.
        var (float, optional): the variance. Defaults to 1.0.
        dtype (torch.dtype, optional): [description]. Defaults to tn.float64.
        device (torch.device, optional): [description]. Defaults to None.

    Returns:
        torchtt.TT: the result.
    """
    d = len(N)
    v1 = var / np.prod(R)
    v = v1**(1/d)
    cores = [None] * d
    for i in range(d):
        cores[i] = tn.randn([R[i],N[i][0],N[i][1],R[i+1]] if isinstance(N[i],tuple) else [R[i],N[i],R[i+1]], dtype = dtype, device = device)*np.sqrt(v)

    return TT(cores)

def reshape(tens, shape, eps = 1e-16, rmax = 1024):
    '''
    Reshapes a tensor in the TT-format. similat to tensorflow.reshape()

    Parameters
    ----------
    tens : TT-object
        The tensor that needs to be reshaped.
    shape : list of integers.
        The target shape.
    eps : float, optional
        The tolerance of the conversion process. The default is 1e-12.
    rmax : int, optional
        the maximum rank. The default is 1000.

    Raises
    ------
    Exception
        Shape mismatch (the product of modes should remain equal).

    Returns
    -------
    TT-object
        The reshaped tensor.

    '''
    
    if tens.is_ttm:
        M = []
        N = []
        for t in shape:
            M.append(t[0])
            N.append(t[1])
        if np.prod(tens.N)!=np.prod(N) or np.prod(tens.M)!=np.prod(M):
            raise Exception('Shape mismatch (the product of modes should remain equal).')
        core = tens.cores[0]
        cores_new = []
        
        idx = 0
        idx_shape = 0
        
        while True:
            if core.shape[1] % M[idx_shape] == 0 and core.shape[2] % N[idx_shape] == 0:
                if core.shape[1] // M[idx_shape] > 1 and core.shape[2] // N[idx_shape] > 1:
                    m1 = M[idx_shape]
                    m2 = core.shape[1] // m1
                    n1 = N[idx_shape]
                    n2 = core.shape[2] // n1
                    r1 = core.shape[0]
                    r2 = core.shape[-1]
                    tmp = tn.reshape(core,[r1*m1,m2,n1,n2*r2])
                    
                    crz,_ = mat_to_tt(tmp, [r1*m1,m2], [n1,n2*r2], eps, rmax)
                    
                    cores_new.append(tn.reshape(crz[0],[r1,m1,n1,-1]))
                    
                    core = tn.reshape(crz[1],[-1,m2,n2,r2]) 
                else:
                    cores_new.append(core+0)
                    if idx == len(tens.cores)-1:
                        break
                    else:
                        idx+=1
                        core = tens.cores[idx]
                idx_shape += 1
                if idx_shape == len(shape):
                    break
            else: 
                idx += 1
                if idx>=len(tens.cores):
                    break
                
                core = tn.einsum('ijkl,lmno->ijmkno',core,tens.cores[idx])
                core = tn.reshape(core,[core.shape[0],core.shape[1]*core.shape[2],-1,core.shape[-1]])
                
    else:
        if np.prod(tens.N)!=np.prod(shape):
            raise Exception('Shape mismatch (the product of modes should remain equal).')
            
        core = tens.cores[0]
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
                    tmp = tn.reshape(core,[r1*s1,s2*r2])
                    
                    crz,_ = to_tt(tmp,tmp.shape,eps,rmax)
                    
                    cores_new.append(tn.reshape(crz[0],[r1,s1,-1]))
                    
                    core = tn.reshape(crz[1],[-1,s2,r2]) 
                else:
                    cores_new.append(core+0)
                    if idx == len(tens.cores)-1:
                        break
                    else:
                        idx+=1
                        core = tens.cores[idx]
                idx_shape += 1
                if idx_shape == len(shape):
                    break
            else: 
                idx += 1
                if idx>=len(tens.cores):
                    break
                
                core = tn.einsum('ijk,klm->ijlm',core,tens.cores[idx])
                core = tn.reshape(core,[core.shape[0],-1,core.shape[-1]])
                
    return TT(cores_new).round(eps)
        
        
def meshgrid(vectors):
    """
    Creates a meshgrid of torchtt.TT objects. Similar to numpy.meshgrid or torch.meshgrid.
    The input is a list of d torch.tensor vectors of sizes N_1, ... ,N_d
    The result is a list of torchtt.TT instances of shapes N1 x ... x Nd.
    
    Args:
        vectors (list[torch.tensor]): the vectors (1d tensors).

    Returns:
        list[torchtt.TT]: the resulting meshgrid.
    """
    
    Xs = []
    dtype = vectors[0].dtype
    for i in range(len(vectors)):
        lst = [tn.ones((1,v.shape[0],1),dtype=dtype) for v in vectors]
        lst[i] = tn.reshape(vectors[i],[1,-1,1])
        Xs.append(TT(lst))
    return Xs
    
def dot(a,b,axis=None):
    """
    Computes the dot product between 2 tensors in TT format.
    If both a and b have identical mode sizes the result is the dot product.
    If a and b have inequal mode sizes, the function perform index contraction. 
    The number of dimensions of a must be greater or equal as b.
    The modes of the tensor a along which the index contraction with b is performed are given in axis.

    Args:
        a (torchtt.TT): the first tensor.
        b (torchtt.TT): the second tensor.
        axis (list[int], optional): the mode indices for index contraction. Defaults to None.

    Raises:
        Exception: [description]
        Exception: [description]
        Exception: [description]
        Exception: [description]
        Exception: [description]

    Returns:
        float or torchtt.TT: the result. If no axis index is provided the result is a scalar otherwise a torchtt.TT object.
    """
    
    if not isinstance(a, TT) or not isinstance(b, TT):
        raise Exception('Both operands should be TT instances.')
    
    
    if axis == None:
        # treat first the full dot product
        # faster than partial projection
        if a.is_ttm or b.is_ttm:
            raise Exception('Operation not implemented for TT-matrices.')
        if a.N != b.N:
            raise Exception('Operands are not the same size.')
        
        result = tn.tensor([[1.0]],dtype = a.cores[0].dtype, device=a.cores[0].device)
        
        for i in range(len(a.N)):
            result = tn.einsum('ab,aim,bin->mn',result, a.cores[i], b.cores[i])
        result = tn.squeeze(result)
    else:
        # partial case
        if a.is_ttm or b.is_ttm:
            raise Exception('Operation not implemented for TT-matrices.')
        if len(a.N)<len(b.N):
            raise Exception('Number of the modes of the first tensor must be equal with the second.')
        # if a.N[axis] != b.N:
        #     raise Exception('Dimension mismatch.')
        
        k = 0 # index for the tensor b
        cores_new = []
        rank_left = 1
        for i in range(len(a.N)):
            if i in axis:
                cores_new.append(b.cores[k])
                rank_left = b.cores[k].shape[2]
                k+=1
            else:
                rank_right = b.cores[k].shape[0] if i+1 in axis else rank_left                
                cores_new.append(tn.einsum('ik,j->ijk',tn.eye(rank_left,rank_right,dtype=a.cores[0].dtype),tn.ones([a.N[i]],dtype=a.cores[0].dtype)))
        
        result = (a*TT(cores_new)).sum(axis)
    return result

def elementwise_divide(x, y, eps = 1e-12, starting_tensor = None, nswp = 50, kick = 4):
    """
    Perform the elemntwise division of two tensors in the TT format using the AMEN method.
    Use this method if different AMEN arguments are needed.
    This method does not check the inputs.
    
    Args:
        x ([type]): [description]
        y ([type]): [description]
        eps ([type], optional): [description]. Defaults to 1e-12.
        starting_tensor ([type], optional): [description]. Defaults to None.
        nswp (int, optional): [description]. Defaults to 50.
        kick (int, optional): [description]. Defaults to 4.

    Returns:
        torchtt.TT: the result
    """

    cores_new = amen_divide(y,x,nswp,starting_tensor,eps,rmax = 1000, kickrank = kick, verbose=False)
    return TT(cores_new)

def rank1TT(vectors):
    """
    Compute the rank 1 TT from a list of vectors.

    Args:
        vectors (list[torch.tensor]): the list of vectors.

    Returns:
        torchtt.TT: the resulting TT object.
    """
    
    return TT([tn.reshape(vectors[i],[1,-1,1]) for i in range(len(vectors))])

 
def numel(tensor):
    """
    Return the number of entries needed to store the TT cores for the given tensor.

    Args:
        tensor (torchtt.TT): the TT representation of the tensor.

    Returns:
        int: number of floats stored for the TT decomposition.
    """
    
    return sum([tn.numel(tensor.cores[i]) for i in range(len(tensor.N))])