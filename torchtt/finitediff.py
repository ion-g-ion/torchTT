"""
Finite difference operators and tools.

@author: Zyntec
"""


import torch as tn
import torchtt as tntt
import numpy as np
from torchtt.errors import *


def laplacian(n, d, derivative = None, dtype = tn.float64, device = None):
    """
    Construct a d-dimensional Laplacian operator with n mesh points in each direction.
    
    Args:
        n (int): the meshpoints.
        d (int): the dimensions.
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
        
    if derivative == None:
        
        for i in range(1, d - 1):
            L_tt = L_tt + tntt.eye([n] * i) ** L_1 ** tntt.eye([n] * (d - 2))
            L_tt = L_tt.round(1e-14)
            
        L_tt = L_tt + L_1 ** tntt.eye([n] * (d - 1)) +  tntt.eye([n] * (d - 1)) ** L_1
        
    else:
        
        if isinstance(derivative, list) and len(derivative) == d:
            
            for i in range(1, d - 1):
                
                if derivative[i] == 1:
                    L_tt = L_tt + tntt.eye([n] * i) ** L_1 ** tntt.eye([n] * (d - 2))
                    L_tt = L_tt.round(1e-14)
                    
            if derivative[0] == 1:
                L_tt = L_tt + L_1 ** tntt.eye([n] * (d - 1))
                
            if derivative[d - 1] == 1:
                L_tt = L_tt + tntt.eye([n] * (d - 1)) ** L_1
                
        else: 
            raise InvalidArguments('Shape must be a list and have the length d.')
            
    return L_tt.round(1e-14)


def boundarydom(n, d, ref_solution = None, dtype = tn.float64, device = None):
    """
    Construct a d-dimensional boundary domain operator with n mesh points in each direction.
    If a reference solution is given, it is automatically inserted into the boundary domain operator.
    
    Args:
        n (int): the meshpoints.
        d (int): the dimensions.
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
    bd_1 = tntt.TT(bd_1, [(n, n)])
    
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


def innerdom(n, d, ref_solution = None, dtype = tn.float64, device = None):   
    """
    Construct a d-dimensional inner domain operator with n mesh points in each direction.
    If a reference solution is given, it is automatically inserted into the boundary domain operator.
    
    Args:
        n (int): the meshpoints.
        d (int): the dimensions.
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
    i_1 = tntt.TT(i_1, [(n, n)])
    
    if d == 0:
        raise InvalidArguments('d must be greater than 0')
     
    elif d == 1:
        i_tt = i_1
        
    else:
        i_tt = i_1
        
        for i in range(d - 1):
            i_tt = i_tt ** i_1
            
    if ref_solution != None:
        i_tt = i_tt @ ref_solution
        
    return i_tt.round(1e-14)    
 
           
def righthandside(n, d, inner_solution = None, boundary_solution = None, inner = None, boundary = None, dtype = tn.float64, device = None):
    """
    Construct a the d-dimensional righthandside of the equation Ax=b with either given an inner solution and boundary_solution or with
    a preconstructed inner and boundary domain operator.
    If a inner solution and boundary solution is given, it will automatical contruct both operators insert the given solutions and construct
    the righthandside. If only one of the two operators is given, the solution to create the other operator must be provided.
    
    Args:
        n (int): the meshpoints.
        d (int): the dimensions.
        boundary_solution (tntt.tensor): the reference solution which is given for the boundary domain.
        inner_solution (tntt.tensor): the reference solution which is given for the inner domain.
        inner (tntt.matrix): preconstructed inner domain operator. 
        boundary (tntt.matrix): preconstructed boundary domain operator.
        dtype (torch.dtype, optional): the dtype of the returned tensor. Defaults to tn.float64.
        device (torch.device, optional): the device where the TT cores are created (None means CPU). Defaults to None.

    Raises:
        InvalidArguments: d must be greater than 0.

    Returns:
        torchtt.TT: The d-dimensional righthandside operator.
    """   
    if inner_solution != None and boundary_solution == None:       
        if boundary == None:
            raise InvalidArguments('only innersolution is given. boundarydomain solution or boundarydomain is needed')
        
        elif boundary != None and inner != None:
            raise InvalidArguments('to much arguments are given')    
        
        return innerdom(n, d, inner_solution, dtype = dtype, device = device) + boundary
    
    elif boundary_solution != None and inner_solution == None:
        if inner == None:
            raise InvalidArguments('only boundarysolution is given. innerdomain solution or Innerdomain is needed')
        
        elif boundary != None and inner != None:
            raise InvalidArguments('to much arguments are given')     
        
        return boundarydom(n, d, boundary_solution, dtype = dtype, device = device) + inner
    
    elif boundary_solution == None and inner_solution == None:
        if inner == None and boundary == None:
            raise InvalidArguments('domain solutions or preconstructed domain operators with imprinted solution are needed')
        
        if boundary == None:
            raise InvalidArguments('Only inner domain operator is given, either a boundary solution or operator is needed')
        
        elif inner == None:
            raise InvalidArguments('Only boundary domain operator is given, either a inner solution or operator is needed')
        
        return inner + boundary
    
    else:
        if boundary != None or inner != None:
            raise InvalidArguments('to much arguments are given')     
        
        return innerdom(n, d, inner_solution, dtype = dtype, device = device)+boundarydom(n, d, boundary_solution, dtype = dtype, device = device)


def centralfd(n, d, derivative = None, dtype = tn.float64, device = None):
    """
    Construct a d-dimensional central difference operator with n mesh points in each direction.
    
    Args:
        n (int): the meshpoints.
        d (int): the dimensions.
        dtype (torch.dtype, optional): the dtype of the returned tensor. Defaults to tn.float64.
        device (torch.device, optional): the device where the TT cores are created (None means CPU). Defaults to None.

    Raises:
        InvalidArguments: d must be greater than 0.

    Returns:
        torchtt.TT: The d-dimensional central difference operator.
    """
    cfd_1 = -1 * tn.diag(tn.ones(n - 1, dtype = dtype, device = device), -1) + tn.diag(tn.ones(n - 1, dtype = dtype, device = device), 1)
    cfd_1[0, 1] = 0
    cfd_1[-1, -2] = 0
    cfd_1 /= 1 / (2 * n)
    cfd_1 = tntt.TT(cfd_1, [(n, n)])
       
    
    if d == 0:
        raise InvalidArguments('d must be greater than 0')
        
    elif d == 1:
        return cfd_1
    
    else:
        cfd_tt = tntt.zeros([(n, n)] * d)
        
    if derivative == None:
        
        for i in range(1, d - 1):
            cfd_tt = cfd_tt + tntt.eye([n] * i) ** cfd_1 ** tntt.eye([n] * (d - 2))
            cfd_tt = cfd_tt.round(1e-14)
            
        cfd_tt = cfd_tt + cfd_1 ** tntt.eye([n] * (d - 1)) +  tntt.eye([n] * (d - 1)) ** cfd_1
        
    else:
        
        if isinstance(derivative, list) and len(derivative) == d:
            
            for i in range(1, d - 1):
                
                if derivative[i] == 1:
                    cfd_tt = cfd_tt + tntt.eye([n] * i) ** cfd_1 ** tntt.eye([n] * (d - 2))
                    cfd_tt = cfd_tt.round(1e-14)
                    
            if derivative[0] == 1:
                cfd_tt = cfd_tt + cfd_1 ** tntt.eye([n] * (d - 1))
                
            if derivative[d - 1] == 1:
                cfd_tt = cfd_tt + tntt.eye([n] * (d - 1)) ** cfd_1
                
        else: 
            raise InvalidArguments('Shape must be a list and have the length d.')
 
    return cfd_tt.round(1e-14)

    
def forwardfd(n, d, derivative = None, dtype = tn.float64, device = None):
    """
    Construct a d-dimensional forward difference operator with n mesh points in each direction.
    
    Args:
        n (int): the meshpoints.
        d (int): the dimensions.
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
        
    if derivative == None:
        
        for i in range(1, d - 1):
            ffd_tt = ffd_tt + tntt.eye([n] * i) ** ffd_1 ** tntt.eye([n] * (d - 2))
            ffd_tt = ffd_tt.round(1e-14)
            
        ffd_tt = ffd_tt + ffd_1 ** tntt.eye([n] * (d - 1)) +  tntt.eye([n] * (d - 1)) ** ffd_1
        
    else:
        
        if isinstance(derivative, list) and len(derivative) == d:
            
            for i in range(1, d - 1):
                
                if derivative[i] == 1:
                    ffd_tt = ffd_tt + tntt.eye([n] * i) ** ffd_1 ** tntt.eye([n] * (d - 2))
                    ffd_tt = ffd_tt.round(1e-14)
                    
            if derivative[0] == 1:
                ffd_tt = ffd_tt + ffd_1 ** tntt.eye([n] * (d - 1))
                
            if derivative[d - 1] == 1:
                ffd_tt = ffd_tt + tntt.eye([n] * (d - 1)) ** ffd_1
                
        else: 
            raise InvalidArguments('Shape must be a list and have the length d.')
    
    return ffd_tt.round(1e-14)


def backwardfd(n, d, derivative = None, dtype = tn.float64, device = None):
    """
    Construct a d-dimensional backward difference operator with n mesh points in each direction.
    
    Args:
        n (int): the meshpoints.
        d (int): the dimensions.
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
        
    if derivative == None:
        
        for i in range(1, d - 1):
            bfd_tt = bfd_tt + tntt.eye([n] * i) ** bfd_1 ** tntt.eye([n] * (d - 2))
            bfd_tt = bfd_tt.round(1e-14)
            
        bfd_tt = bfd_tt + bfd_1 ** tntt.eye([n] * (d - 1)) +  tntt.eye([n] * (d - 1)) ** bfd_1
        
    else:
        
        if isinstance(derivative, list) and len(derivative) == d:
            
            for i in range(1, d - 1):
                
                if derivative[i] == 1:
                    bfd_tt = bfd_tt + tntt.eye([n] * i) ** bfd_1 ** tntt.eye([n] * (d - 2))
                    bfd_tt = bfd_tt.round(1e-14)
                    
            if derivative[0] == 1:
                bfd_tt = bfd_tt + bfd_1 ** tntt.eye([n] * (d - 1))
                
            if derivative[d - 1] == 1:
                bfd_tt = bfd_tt + tntt.eye([n] * (d - 1)) ** bfd_1
                
        else: 
            raise InvalidArguments('Shape must be a list and have the length d.')
  
    return bfd_tt.round(1e-14)


def backward2fd(n, d, derivative = None, order = 1, dtype = tn.float64, device = None):
    """
    Construct a d-dimensional backward difference operator of order of approximation 2 and with the possibility of taking the second order of derivative.
    With n mesh points in each direction.
    
    Args:
        n (int): the meshpoints.
        d (int): the dimensions.
        order (int): order of derivative. Choose between 1 and 2
        dtype (torch.dtype, optional): the dtype of the returned tensor. Defaults to tn.float64.
        device (torch.device, optional): the device where the TT cores are created (None means CPU). Defaults to None.

    Raises:
        InvalidArguments: d must be greater than 0.

    Returns:
        torchtt.TT: The d-dimensional 2nd order backward difference operator.
    """
    if order == 1:
        b2fd_1 = -4 * tn.diag(tn.ones(n - 1, dtype = dtype, device = device), -1) + 3 * tn.eye(n, dtype = dtype, device = device) + tn.diag(tn.ones(n - 2, dtype = dtype, device = device), -2)
        b2fd_1[0, 1] = 0
        b2fd_1[-1, -2] = 0
        b2fd_1 /= 1 / (2 * n)
        b2fd_1 = tntt.TT(b2fd_1, [(n, n)])
        
    elif order == 2:
        b2fd_1 = -2 * tn.diag(tn.ones(n - 1, dtype = dtype, device = device), -1) + tn.eye(n, dtype = dtype, device = device) + tn.diag(tn.ones(n - 2, dtype = dtype, device = device), -2)
        b2fd_1[0, 1] = 0
        b2fd_1[-1, -2] = 0
        b2fd_1 /= 1 / n ** 2
        b2fd_1 = tntt.TT(b2fd_1, [(n, n)])
        
    else:
        raise InvalidArguments('order cant be higher than 2')
    
    if d == 0:
        raise InvalidArguments('d must be greater than 0')
        
    elif d == 1:
        return b2fd_1
    
    else:
        b2fd_tt = tntt.zeros([(n, n)] * d)
        
    if derivative == None:
        
        for i in range(1, d - 1):
            b2fd_tt = b2fd_tt + tntt.eye([n] * i) ** b2fd_1 ** tntt.eye([n] * (d - 2))
            b2fd_tt = b2fd_tt.round(1e-14)
            
        b2fd_tt = b2fd_tt + b2fd_1 ** tntt.eye([n] * (d - 1)) +  tntt.eye([n] * (d - 1)) ** b2fd_1
        
    else:
        
        if isinstance(derivative, list) and len(derivative) == d:
            
            for i in range(1, d - 1):
                
                if derivative[i] == 1:
                    b2fd_tt = b2fd_tt + tntt.eye([n] * i) ** b2fd_1 ** tntt.eye([n] * (d - 2))
                    b2fd_tt = b2fd_tt.round(1e-14)
                    
            if derivative[0] == 1:
                b2fd_tt = b2fd_tt + b2fd_1 ** tntt.eye([n] * (d - 1))
                
            if derivative[d - 1] == 1:
                b2fd_tt = b2fd_tt + tntt.eye([n] * (d - 1)) ** b2fd_1
                
        else: 
            raise InvalidArguments('Shape must be a list and have the length d.')
    
    return b2fd_tt.round(1e-14)


def forward2fd(n, d, derivative = None, order = 1, dtype = tn.float64, device = None):
    """
    Construct a d-dimensional forward difference operator of order of approximation 2 and with the possibility of taking the second order of derivative.
    With n mesh points in each direction.
    
    Args:
        n (int): the meshpoints.
        d (int): the dimensions.
        order (int): order of derivative. Choose between 1 and 2
        dtype (torch.dtype, optional): the dtype of the returned tensor. Defaults to tn.float64.
        device (torch.device, optional): the device where the TT cores are created (None means CPU). Defaults to None.

    Raises:
        InvalidArguments: d must be greater than 0.

    Returns:
        torchtt.TT: The d-dimensional 2nd order forward difference operator.
    """
    if order == 1:
        f2fd_1 = 4 * tn.diag(tn.ones(n - 1, dtype = dtype, device = device), 1) - 3 * tn.eye(n, dtype = dtype, device = device) - tn.diag(tn.ones(n - 2, dtype = dtype, device = device), 2)
        f2fd_1[0, 1] = 0
        f2fd_1[-1, -2] = 0
        f2fd_1 /= 1 / (2 * n)
        f2fd_1 = tntt.TT(f2fd_1, [(n, n)])
        
    elif order == 2:
        f2fd_1 = -2 * tn.diag(tn.ones(n - 1, dtype = dtype, device = device), 1) + tn.eye(n, dtype = dtype, device = device) + tn.diag(tn.ones(n - 2, dtype = dtype, device = device), 2)
        f2fd_1[0, 1] = 0
        f2fd_1[-1, -2] = 0
        f2fd_1 /= 1 / n ** 2
        f2fd_1 = tntt.TT(f2fd_1, [(n, n)])
        
    else:
        raise InvalidArguments('order cant be higher than 2')
    
    if d == 0:
        raise InvalidArguments('d must be greater than 0')
        
    elif d == 1:
        return f2fd_1
    
    else:
        f2fd_tt = tntt.zeros([(n, n)] * d)
        
    if derivative == None:
        
        for i in range(1, d - 1):
            f2fd_tt = f2fd_tt + tntt.eye([n] * i) ** f2fd_1 ** tntt.eye([n] * (d - 2))
            f2fd_tt = f2fd_tt.round(1e-14)
            
        f2fd_tt = f2fd_tt + f2fd_1 ** tntt.eye([n] * (d - 1)) +  tntt.eye([n] * (d - 1)) ** f2fd_1
        
    else:
        
        if isinstance(derivative, list) and len(derivative) == d:
            
            for i in range(1, d - 1):
                
                if derivative[i] == 1:
                    f2fd_tt = f2fd_tt + tntt.eye([n] * i) ** f2fd_1 ** tntt.eye([n] * (d - 2))
                    f2fd_tt = f2fd_tt.round(1e-14)
                    
            if derivative[0] == 1:
                f2fd_tt = f2fd_tt + f2fd_1 ** tntt.eye([n] * (d - 1))
                
            if derivative[d - 1] == 1:
                f2fd_tt = f2fd_tt + tntt.eye([n] * (d - 1)) ** f2fd_1
                
        else: 
            raise InvalidArguments('Shape must be a list and have the length d.')
  
    return f2fd_tt.round(1e-14)

