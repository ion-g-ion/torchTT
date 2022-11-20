"""
Finite difference operators and tools.

@author: Zyntec
"""


import torch as tn
import torchtt as tntt
import numpy as np
from torchtt.errors import *


def Laplacian(d, n, dtype = tn.float64, device = None):
    """
    Construct a d-dimensional Laplacian operator with n mesh points in each direction.
    
    Args:
        d (int): the dimensions.
        n (int): the meshpoints.
        dtype (torch.dtype, optional): the dtype of the returned tensor. Defaults to tn.float64.
        device (torch.device, optional): the device where the TT cores are created (None means CPU). Defaults to None.

    Raises:
        InvalidArguments: d must be greater than 0.

    Returns:
        torchtt.TT: The d-dimensional Laplacian operator.
    """
    L_1 = -2 * tn.eye(n, dtype = dtype, device = device) + tn.diag(tn.ones(n - 1, dtype = dtype, device = device), -1) + tn.diag(tn.ones(n - 1, dtype = dtype, device = device), 1)
    L_1[0, 1] = 0
    L_1[-1, -2] = 0
    L_1 /= 1 / n ** 2
    L_1 = tntt.TT(L_1, [(n, n)])
   
    
    if d == 0:
        raise InvalidArguments('d must be greater than 0.')
        
    elif d == 1:
        return L_1
    
    else:
        L_tt = tntt.zeros([(n, n)] * d)
        for i in range(1, d - 1):
            L_tt = L_tt + tntt.eye([n] * i) ** L_1 ** tntt.eye([n] * (d - 2))
            L_tt = L_tt.round(1e-14)
        L_tt = L_tt + L_1 ** tntt.eye([n] * (d - 1)) +  tntt.eye([n] * (d - 1)) ** L_1
    

    return L_tt.round(1e-14)


def boundarydom(d, n, ref_solution = None, dtype = tn.float64, device = None):
    """
    Construct a d-dimensional boundary domain operator with n mesh points in each direction.
    If a reference solution is given, it is automatically inserted into the boundary domain operator.
    
    Args:
        d (int): the dimensions.
        n (int): the meshpoints.
        ref_solution (tntt.tensor): the reference solution which is given for the boundary
        dtype (torch.dtype, optional): the dtype of the returned tensor. Defaults to tn.float64.
        device (torch.device, optional): the device where the TT cores are created (None means CPU). Defaults to None.

    Raises:
        InvalidArguments: d must be greater than 0.

    Returns:
        torchtt.TT: The d-dimensional boundary domain operator.
    """
    bd_1 = tn.zeros(n,n, dtype = dtype, device = device)
    bd_1[0,0] = 1
    bd_1[-1,-1] = 1
    bd_1 /= (1 / n ** 2) / (2 * d)
    bd_1=tntt.TT(bd_1, [(n, n)])
    
    if d == 0:
        raise InvalidArguments('d must be greater than 0')
        
    elif d == 1:
        bd_tt = bd_1
        
    else:
        bd_tt = tntt.zeros([(n, n)] * d)
        for i in range(1, d - 1):
            bd_tt = bd_tt + tntt.eye([n] * i) ** bd_1 ** tntt.eye([n] * (d - 2))
            bd_tt = bd_tt.round(1e-14)
        bd_tt = bd_tt + bd_1 ** tntt.eye([n] * (d - 1)) +  tntt.eye([n] * (d - 1)) ** bd_1
                    
    if ref_solution != None:
        bd_tt = bd_tt @ ref_solution
    
    return bd_tt.round(1e-14)


def innerdom(d, n, ref_solution = None, dtype = tn.float64, device = None):   
    """
    Construct a d-dimensional inner domain operator with n mesh points in each direction.
    If a reference solution is given, it is automatically inserted into the boundary domain operator.
    
    Args:
        d (int): the dimensions.
        n (int): the meshpoints.
        ref_solution (tntt.tensor): the reference solution which is given for the inner domain
        dtype (torch.dtype, optional): the dtype of the returned tensor. Defaults to tn.float64.
        device (torch.device, optional): the device where the TT cores are created (None means CPU). Defaults to None.

    Raises:
        InvalidArguments: d must be greater than 0.

    Returns:
        torchtt.TT: The d-dimensional inner domain operator.
    """
    i_1 = tn.eye(n, dtype = dtype, device = device)
    i_1[0,:] = 0
    i_1[-1,:] = 0 
    i_1=tntt.TT(i_1, [(n, n)])
    
    if d == 0:
        raise InvalidArguments('d must be greater than 0')
     
    elif d == 1:
        i_tt = i_1
        
    else:
        i_tt = i_1
        for i in range(d-1):
            i_tt = i_tt ** i_1
    if ref_solution != None:
        i_tt = i_tt @ ref_solution
        
    return i_tt.round(1e-14)    
 
           
def righthandside(d, n, inner_solution = None, boundary_solution = None, inner = None, boundary = None, dtype = tn.float64, device = None):
    """
    Construct a the d-dimensional righthandside of the equation Ax=b with either given an inner solution and boundary_solution or with
    a preconstructed inner and boundary domain operator.
    If a inner solution and boundary solution is given, it will automatical contruct both operators insert the given solutions and construct
    the righthandside. If only one of the two operators is given, the solution to create the other operator must be provided.
    
    Args:
        d (int): the dimensions.
        n (int): the meshpoints.
        boundary_solution (tntt.tensor): the reference solution which is given for the boundary domain.
        inner_solution (tntt.tensor): the reference solution which is given for the inner domain.
        inner (tntt.matrix): preconstructed inner domain operator. 
        boundary (tntt.matrix): preconstructed boundary domain operator.
        dtype (torch.dtype, optional): the dtype of the returned tensor. Defaults to tn.float64.
        device (torch.device, optional): the device where the TT cores are created (None means CPU). Defaults to None.

    Raises:
        InvalidArguments: d must be greater than 0.

    Returns:
        torchtt.TT: The d-dimensional d-dimensional righthandside operator.
    """
    
    if inner_solution == None or boundary_solution == None:
        if inner == None and boundary != None:
            raise InvalidArguments('only boundarydomain is given. innerdomain solution or Innerdomain is needed')
        elif boundary == None and inner != None:
            raise InvalidArguments('only innerdomain is given. innerdomain solution or boundarydomain is needed')
        
        return inner + boundary
    
    elif inner == None and boundary != None:
        
        return innerdom(d, n, ref_solution, dtype = dtype, device = device) + boundary
    
    elif boundary == None and inner != None:
        
        return boundarydom(d, n, ref_solution, dtype = dtype, device = device) + inner
    
    elif boundary != None and inner != None:
        raise InvalidArguments('to much arguments are given')
    
    
    return innerdom(d, n, inner_solution, dtype = dtype, device = device)+boundarydom(d, n, boundary_solution, dtype = dtype, device = device)


def centralfd(d, n, dtype = tn.float64, device = None):
    """
    Construct a d-dimensional central difference operator with n mesh points in each direction.
    
    Args:
        d (int): the dimensions.
        n (int): the meshpoints.
        dtype (torch.dtype, optional): the dtype of the returned tensor. Defaults to tn.float64.
        device (torch.device, optional): the device where the TT cores are created (None means CPU). Defaults to None.

    Raises:
        InvalidArguments: d must be greater than 0.

    Returns:
        torchtt.TT: The d-dimensional central difference operator.
    """
    Cfd_1 = -1 * tn.diag(tn.ones(n - 1, dtype = dtype, device = device), -1) + tn.diag(tn.ones(n - 1, dtype = dtype, device = device), 1)
    Cfd_1[0, 1] = 0
    Cfd_1[-1, -2] = 0
    Cfd_1 /= 1 / (2 * n)
    Cfd_1 = tntt.TT(Cfd_1, [(n, n)])
       
    
    if d == 0:
        raise InvalidArguments('d must be greater than 0')
        
    elif d == 1:
        return Cfd_1
    
    else:
        Cfd_tt = tntt.zeros([(n, n)] * d)
        for i in range(1, d - 1):
            Cfd_tt = Cfd_tt + tntt.eye([n] * i) ** Cfd_1 ** tntt.eye([n] * (d - 2))
            Cfd_tt = Cfd_tt.round(1e-14)
        Cfd_tt = Cfd_tt + Cfd_1 ** tntt.eye([n] * (d - 1)) +  tntt.eye([n] * (d - 1)) ** Cfd_1

  
    return Cfd_tt.round(1e-14)

    
def forwardfd(d, n, dtype = tn.float64, device = None):
    """
    Construct a d-dimensional forward difference operator with n mesh points in each direction.
    
    Args:
        d (int): the dimensions.
        n (int): the meshpoints.
        dtype (torch.dtype, optional): the dtype of the returned tensor. Defaults to tn.float64.
        device (torch.device, optional): the device where the TT cores are created (None means CPU). Defaults to None.

    Raises:
        InvalidArguments: d must be greater than 0.

    Returns:
        torchtt.TT: The d-dimensional forward difference operator.
    """
    ffd_1 = -1 * tn.eye(n, dtype = dtype, device = device) + tn.diag(tn.ones(n - 1, dtype = dtype, device = device), 1)
    ffd_1[0, 1] = 0
    ffd_1[-1, -2] = 0
    ffd_1 /= 1 / n
    ffd_1 = tntt.TT(ffd_1, [(n, n)])
   
    
    if d == 0:
        raise InvalidArguments('d must be greater than 0')
        
    elif d == 1:
        return ffd_1
    
    else:
        ffd_tt = tntt.zeros([(n, n)] * d)
        for i in range(1, d - 1):
            ffd_tt = ffd_tt + tntt.eye([n] * i) ** ffd_1 ** tntt.eye([n] * (d - 2))
            ffd_tt = ffd_tt.round(1e-14)
        ffd_tt = ffd_tt + ffd_1 ** tntt.eye([n] * (d - 1)) +  tntt.eye([n] * (d - 1)) ** ffd_1
    
  
    return ffd_tt.round(1e-14)


def backwardfd(d, n, dtype = tn.float64, device = None):
    """
    Construct a d-dimensional backward difference operator with n mesh points in each direction.
    
    Args:
        d (int): the dimensions.
        n (int): the meshpoints.
        dtype (torch.dtype, optional): the dtype of the returned tensor. Defaults to tn.float64.
        device (torch.device, optional): the device where the TT cores are created (None means CPU). Defaults to None.

    Raises:
        InvalidArguments: d must be greater than 0.

    Returns:
        torchtt.TT: The d-dimensional backward difference operator.
    """
    bfd_1 = -1 * tn.diag(tn.ones(n - 1, dtype = dtype, device = device), -1) + tn.eye(n, dtype = dtype, device = device)
    bfd_1[0, 1] = 0
    bfd_1[-1, -2] = 0
    bfd_1 /= 1 / n
    bfd_1 = tntt.TT(bfd_1, [(n, n)])
   
    
    if d == 0:
        raise InvalidArguments('d must be greater than 0')
        
    elif d == 1:
        return bfd_1
    
    else:
        bfd_tt = tntt.zeros([(n, n)] * d)
        for i in range(1, d - 1):
            bfd_tt = bfd_tt + tntt.eye([n] * i) ** bfd_1 ** tntt.eye([n] * (d - 2))
            bfd_tt = bfd_tt.round(1e-14)
        bfd_tt = bfd_tt + bfd_1 ** tntt.eye([n] * (d - 1)) +  tntt.eye([n] * (d - 1)) ** bfd_1
    
  
    return bfd_tt.round(1e-14)

