"""
Finite difference operators and tools.

@author: Zyntec
"""


import torch as tn
import torchtt as tntt
import numpy as np
from torchtt.errors import *

def isEquidistant(xn, dtype = tn.float64, device = None):
    if isinstance(xn, list) and isinstance(xn[0], tntt.TT):
        d = len(xn)

        for i in range(d):
            slices = [1] * d
            slices[i] = slice(None)
            
            vector= xn[i][tuple(slices)].numpy()
            shifted_vector = np.roll(vector, shift=1, axis=0)
            
            stepsize=shifted_vector-vector
            stepsize[0]=stepsize[1]
        
            stepsize= np.abs(stepsize)
            stepsize = np.round(stepsize,14)
    
            # Prüfe, ob alle Entfernungen gleich sind
            if np.all(stepsize == stepsize[0])==False:
                return (False)
        return(True)
    else:
        raise InvalidArguments('Input must be a list of TT.Tensors.')
        
def getstepsize(xn, dtype = tn.float64, device = None):
    
    if isinstance(xn, list) and isinstance(xn[0], tntt.TT):
        d = len(xn)
        n = xn[0].shape[0]
        
        if isEquidistant(xn)==True:
            slices = [1] * d
            slices[0] = slice(None)
            
            vector= xn[0][tuple(slices)].numpy()
            shifted_vector = np.roll(vector, shift=1, axis=0)
            
            stepsize=shifted_vector-vector
            stepsize[0]=stepsize[1]
            
            stepsize= np.abs(stepsize)
            stepsize = np.round(stepsize,15)
            return stepsize[0]
        else:
            steplist = []
            for i in range(d):
                slices = [1] * d
                slices[i] = slice(None)
                
                vector= xn[i][tuple(slices)].numpy()
                shifted_vector = np.roll(vector, shift=1, axis=0)
                
                stepsize=shifted_vector-vector
                
                
                stepsize= np.abs(stepsize)
                stepsize = np.round(stepsize,15)
                steplist.append(stepsize)
                
            return steplist
            
    else:
        raise InvalidArguments('Input must be a list of TT.Tensors.')       

def set_boundary_conditions(operator, dimensions = 0, Boundary_conditions= 'Zero', dtype = tn.float64, device = None):
    
    Bcs= ['Zero', 'Dirichlet', 'Neumann', 'Periodic']
    if isinstance(operator, tn.Tensor):
        
        if Boundary_conditions == 'Zero':
            operator[0, :] = 0
            operator[-1, :] = 0
           
        elif Boundary_conditions == 'Dirichlet':
            operator[0, :] = 0
            operator[-1, :] = 0
            if dimensions != 0:
                operator[0, 0] = -1 *(dimensions*2)
                operator[-1, -1] = -1 *(dimensions*2)
            else:
                operator[0, 0] = -1
                operator[-1, -1] = -1
        else: 
            raise InvalidArguments('Wrong boundary conditions, choose between, Zero, Dirichlet, Neumann or Periodic')
    
    else: 
        raise InvalidArguments('operator must be a pytorch tensor')
    
    return operator
            
def operator_to_higherdim(operator, meshpoints, dimensions, stepsize = None, derivative = None, dtype = tn.float64, device = None):
    
    n = meshpoints
    d = dimensions
    op_tt = tntt.zeros([(n, n)] * d)
    
    if stepsize == None:    
        if derivative == None:
            
            for i in range(1, d - 1):
                op_tt = op_tt + tntt.eye([n] * i) ** operator ** tntt.eye([n] * (d - 1-i))
                op_tt = op_tt.round(1e-14)
                
            op_tt = op_tt + operator ** tntt.eye([n] * (d - 1)) +  tntt.eye([n] * (d - 1)) ** operator
            
        else:
            
            if isinstance(derivative, list) and len(derivative) == d:
                
                for i in range(1, d - 1):
                    
                    if derivative[i] == 1:
                        op_tt = op_tt + tntt.eye([n] * i) ** operator ** tntt.eye([n] * (d - 1 - i))
                        op_tt = op_tt.round(1e-14)
                        
                if derivative[0] == 1:
                    op_tt = op_tt + operator ** tntt.eye([n] * (d - 1))
                    
                if derivative[d - 1] == 1:
                    op_tt = op_tt + tntt.eye([n] * (d - 1)) ** operator
                    
            else: 
                raise InvalidArguments('Shape must be a list and have the length d.')
    else:
        
        if isinstance(stepsize, list) and len(stepsize) == d:
            if derivative == None:
                h = stepsize
                h_1=[]
                
                weight=[]
                
                for i in range(d):

                    h_1.append(np.roll(h[i], -1))
                    h_1[i][-1]=h_1[i][-2]
                    h[i][0]=h[i][1]
                    
                    weight.append((1*tn.eye(n, dtype = dtype, device = device)*(h[i]+h_1[i]) + 2*tn.diag(tn.ones(n - 1, dtype = dtype, device = device)*h[i][1:], 1) + 2*tn.diag(tn.ones(n - 1, dtype = dtype, device = device)*h_1[i][:-1], -1))/(h[i]*h_1[i]*(h[i]+h_1[i])))
                     
                
                for i in range(1, d - 1):
                    
                    op_tt = op_tt + tntt.eye([n] * i) ** (operator*tntt.TT(weight[i], [(n, n)])) ** tntt.eye([n] * (d - 1-i))
                    op_tt = op_tt.round(1e-14)
                    
                op_tt = op_tt + (operator*tntt.TT(weight[0], [(n, n)])) ** tntt.eye([n] * (d - 1)) +  tntt.eye([n] * (d - 1)) ** (operator*tntt.TT(weight[-1], [(n, n)]))
                
            else:
                
                if isinstance(derivative, list) and len(derivative) == d:
                    
                    for i in range(1, d - 1):
                        
                        if derivative[i] == 1:
                            op_tt = op_tt + tntt.eye([n] * i) ** operator ** tntt.eye([n] * (d - 1 - i))
                            op_tt = op_tt.round(1e-14)
                            
                    if derivative[0] == 1:
                        op_tt = op_tt + operator ** tntt.eye([n] * (d - 1))
                        
                    if derivative[d - 1] == 1:
                        op_tt = op_tt + tntt.eye([n] * (d - 1)) ** operator
                        
                else: 
                    raise InvalidArguments('Shape must be a list and have the length d.')
                
    return op_tt.round(1e-14)
    
def laplacian(xn, boundarycondition = "Zero", derivative = None, dtype = tn.float64, device = None):
    """
    Construct a d-dimensional Laplacian operator with n mesh points in each direction.
    
    Args:
        xn (list[tntt.Tensor): List of the mesh tensors in the TT format.
        derivative (list[int]): List of ones and zeros. Where every one leads to the derivative in this specific direction.
        dtype (torch.dtype, optional): the dtype of the returned tensor. Defaults to tn.float64.
        device (torch.device, optional): the device where the TT cores are created (None means CPU). Defaults to None.

    Raises:
        InvalidArguments: d must be greater than 0.

    Returns:
        torchtt.TT: The d-dimensional Laplacian operator.
    """
    if isinstance(xn, list) and isinstance(xn[0], tntt.TT):
        d = len(xn)
        n = xn[0].shape[0]
        
        L_1 = -2 * tn.eye(n, dtype = dtype, device = device) + tn.diag(tn.ones(n - 1, dtype = dtype, device = device), -1) + tn.diag(tn.ones(n - 1, dtype = dtype, device = device), 1)
        L_1 = set_boundary_conditions(L_1, d, Boundary_conditions = boundarycondition)
        
        if isEquidistant(xn)==True:
            h = getstepsize(xn)
            L_1 = L_1 * 1 / (h ** 2)
            L_1 = tntt.TT(L_1, [(n, n)])
            
            if d == 0:
                raise InvalidArguments('d must be greater than 0.')
                
            elif d == 1:
                return L_1
            
            else:          
                L_tt = operator_to_higherdim(L_1, n, d, derivative = derivative)
        else:
            
            
            ''' doesnt work at the moment'''
            h = getstepsize(xn)
            
        
            if d == 0:
                raise InvalidArguments('d must be greater than 0.')
            
            elif d == 1:
                h_1=[]
                h_1.append(np.roll(h[0], -1))
                h_1[0][-1]=h_1[0][-2]
                h[0][0]=h[0][1]
                weight= (1*tn.eye(n, dtype = dtype, device = device)*(h[0]+h_1[0]) + 2*tn.diag(tn.ones(n - 1, dtype = dtype, device = device)*h[0][1:], 1) + 2*tn.diag(tn.ones(n - 1, dtype = dtype, device = device)*h_1[0][:-1], -1))/(h[0]*h_1[0]*(h[0]+h_1[0]))
                L_1 = tntt.TT(L_1, [(n, n)])
                L_1=L_1*tntt.TT(weight, [(n, n)])
                return L_1
        
            else: 
                L_1 = tntt.TT(L_1, [(n, n)])
                L_tt = operator_to_higherdim(L_1, n, d, h, derivative = derivative)
            
    else: 
        raise InvalidArguments('Shape must be a list of TT.Tensors.')
       
    return L_tt.round(1e-14)


def boundarydom(xn, ref_solution = None, boundarycondition = 'Zero', dtype = tn.float64, device = None):
    """
    Construct a d-dimensional boundary domain operator with n mesh points in each direction.
    If a reference solution is given, it is automatically inserted into the boundary domain operator.
    
    Args:
        xn (list[tntt.Tensor): List of the mesh tensors in the TT format.
        ref_solution (tntt.tensor): the reference solution which is given for the boundary
        dtype (torch.dtype, optional): the dtype of the returned tensor. Defaults to tn.float64.
        device (torch.device, optional): the device where the TT cores are created (None means CPU). Defaults to None.

    Raises:
        InvalidArguments: d must be greater than 0.

    Returns:
        torchtt.TT: The d-dimensional boundary domain operator.
    """
    if isinstance(xn, list) and isinstance(xn[0], tntt.TT):
        d = len(xn)
        n = xn[0].shape[0]
        
        bd_1 = tn.zeros(n,n, dtype = dtype, device = device)
        bd_1 = set_boundary_conditions(bd_1, d, Boundary_conditions= boundarycondition)

        
        if isEquidistant(xn)==True:
            h = getstepsize(xn)
            bd_1 = bd_1 * ( (1) / (h ** 2))
            bd_1 = tntt.TT(bd_1, [(n, n)])
            
            if d == 0:
                raise InvalidArguments('d must be greater than 0')
                
            elif d == 1:
                bd_tt = bd_1
                
            else:          
                bd_tt = operator_to_higherdim(bd_1, n, d)
        else:
            
            ''' doesnt work at the moment'''
            h = getstepsize(xn)
            if d == 0:
                raise InvalidArguments('d must be greater than 0')
                
            elif d == 1:
                h_1=[]
                h_1.append(np.roll(h[0], -1))
                h_1[0][-1]=h_1[0][-2]
                h[0][0]=h[0][1]
                weight= (1*tn.eye(n, dtype = dtype, device = device)*(h[0]+h_1[0]) + 2*tn.diag(tn.ones(n - 1, dtype = dtype, device = device)*h[0][1:], -1) + 2*tn.diag(tn.ones(n - 1, dtype = dtype, device = device)*h_1[0][:-1], 1))/(h[0]*h_1[0]*(h[0]+h_1[0]))
                bd_1 = tntt.TT(bd_1, [(n, n)])
                bd_1=bd_1*tntt.TT(weight, [(n, n)])    
                bd_tt = bd_1
            else: 
                bd_1 = tntt.TT(bd_1, [(n, n)])
                bd_tt = operator_to_higherdim(bd_1, n, d, h)
            
        if ref_solution != None:
            bd_tt = (bd_tt @ ref_solution).round(1e-14)
            
    else: 
        raise InvalidArguments('Shape must be a list of TT.Tensors.')
        
    return bd_tt


def innerdom(xn, ref_solution = None, dtype = tn.float64, device = None):   
    """
    Construct a d-dimensional inner domain operator with n mesh points in each direction.
    If a reference solution is given, it is automatically inserted into the boundary domain operator.
    
    Args:
        xn (list[tntt.Tensor): List of the mesh tensors in the TT format.
        ref_solution (tntt.tensor): the reference solution which is given for the inner domain
        dtype (torch.dtype, optional): the dtype of the returned tensor. Defaults to tn.float64.
        device (torch.device, optional): the device where the TT cores are created (None means CPU). Defaults to None.

    Raises:
        InvalidArguments: d must be greater than 0.

    Returns:
        torchtt.TT: The d-dimensional inner domain operator.
    """
    if isinstance(xn, list) and isinstance(xn[0], tntt.TT):
        d = len(xn)
        n = xn[0].shape[0]
    
    
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
                i_tt =  i_tt ** i_1 
        
        
                
        if ref_solution != None:
            i_tt = (i_tt @ ref_solution).round(1e-14)    
    
    else: 
        raise InvalidArguments('Shape must be a list of TT.Tensors.')    
        
    return i_tt
 
           
def righthandside(xn, inner_solution = None, boundary_solution = None, inner = None, boundary = None, boundarycondition = "Zero",dtype = tn.float64, device = None):
    """
    Construct a the d-dimensional righthandside of the equation Ax=b with either given an inner solution and boundary_solution or with
    a preconstructed inner and boundary domain operator.
    If a inner solution and boundary solution is given, it will automatical contruct both operators insert the given solutions and construct
    the righthandside. If only one of the two operators is given, the solution to create the other operator must be provided.
    
    Args:
        xn (list[tntt.Tensor): List of the mesh tensors in the TT format.
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
    if isinstance(xn, list) and isinstance(xn[0], tntt.TT):
        
        if inner_solution != None and boundary_solution == None:       
            if boundary == None:
                raise InvalidArguments('only innersolution is given. boundarydomain solution or boundarydomain is needed')
            
            elif boundary != None and inner != None:
                raise InvalidArguments('to much arguments are given')    
            
            return innerdom(xn, inner_solution, dtype = dtype, device = device) + boundary
        
        elif boundary_solution != None and inner_solution == None:
            if inner == None:
                raise InvalidArguments('only boundarysolution is given. innerdomain solution or Innerdomain is needed')
            
            elif boundary != None and inner != None:
                raise InvalidArguments('to much arguments are given')     
            
            return boundarydom(xn, boundary_solution, boundarycondition = boundarycondition, dtype = dtype, device = device) + inner
        
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
                
            return innerdom(xn, inner_solution, dtype = dtype, device = device)+boundarydom(xn, boundary_solution, boundarycondition = boundarycondition, dtype = dtype, device = device)
        
    else: 
        raise InvalidArguments('Shape must be a list of TT.Tensors.')    
         
           

''' doesnt Work at the moment


def centralfd(xn, boundarycondition = 'Zero', derivative = None, order = 1, dtype = tn.float64, device = None):
    """
    Construct a d-dimensional central difference operator with n mesh points in each direction.
    
    Args:
        xn (list[tntt.Tensor): List of the mesh tensors in the TT format.
        derivative (list[int]): List of ones and zeros. Where every one leads to the derivative in this specific direction.
        dtype (torch.dtype, optional): the dtype of the returned tensor. Defaults to tn.float64.
        device (torch.device, optional): the device where the TT cores are created (None means CPU). Defaults to None.

    Raises:
        InvalidArguments: d must be greater than 0.

    Returns:
        torchtt.TT: The d-dimensional central difference operator.
    """
    if isinstance(xn, list) and isinstance(xn[0], tntt.TT):
        d = len(xn)
        n = xn[0].shape[0]
        h = getstepsize(xn)
        
        if order == 1:
            cfd_1 = -1 * tn.diag(tn.ones(n - 1, dtype = dtype, device = device), -1) + tn.diag(tn.ones(n - 1, dtype = dtype, device = device), 1)
            cfd_1 = set_boundary_conditions(cfd_1, d, Boundary_conditions = boundarycondition)
            cfd_1 = cfd_1 / (2 * h)
            cfd_1 = tntt.TT(cfd_1, [(n, n)])
           
        elif order == 2:
            cfd_1 = laplacian(xn, boundarycondition = boundarycondition, derivative = derivative, dtype = dtype, device = dtype)
            
        if d == 0:
            raise InvalidArguments('d must be greater than 0')
            
        elif d == 1:
            return cfd_1
        
        else:          
            cfd_tt = operator_to_higherdim(cfd_1, n, d, derivative = derivative)
            
    else: 
        raise InvalidArguments('Shape must be a list of TT.Tensors.')    
        
    return cfd_tt.round(1e-14)

    
def forwardfd(xn, boundarycondition = 'Zero', derivative = None, order = 1, dtype = tn.float64, device = None):
    """
    Construct a d-dimensional forward difference operator with n mesh points in each direction.
    
    Args:
        xn (list[tntt.Tensor): List of the mesh tensors in the TT format.
        derivative (list[int]): List of ones and zeros. Where every one leads to the derivative in this specific direction.
        dtype (torch.dtype, optional): the dtype of the returned tensor. Defaults to tn.float64.
        device (torch.device, optional): the device where the TT cores are created (None means CPU). Defaults to None.

    Raises:
        InvalidArguments: d must be greater than 0.

    Returns:
        torchtt.TT: The d-dimensional forward difference operator.
    """
    if isinstance(xn, list) and isinstance(xn[0], tntt.TT):
        d = len(xn)
        n = xn[0].shape[0]
        h = getstepsize(xn)
        
        if order == 1:
            ffd_1 = -1 * tn.eye(n, dtype = dtype, device = device) + tn.diag(tn.ones(n - 1, dtype = dtype, device = device), 1)
            ffd_1 = set_boundary_conditions(ffd_1, d, Boundary_conditions = boundarycondition)
            ffd_1 = ffd_1 * 1 / h
            ffd_1 = tntt.TT(ffd_1, [(n, n)])
        
        elif order == 2:
            ffd_1 = -2 * tn.diag(tn.ones(n - 1, dtype = dtype, device = device), 1) + tn.eye(n, dtype = dtype, device = device) + tn.diag(tn.ones(n - 2, dtype = dtype, device = device), 2)
            ffd_1 = set_boundary_conditions(ffd_1, d, Boundary_conditions = boundarycondition)
            ffd_1 = ffd_1* 1 / n ** 2
            ffd_1 = tntt.TT(ffd_1, [(n, n)])
        
        if d == 0:
            raise InvalidArguments('d must be greater than 0')
            
        elif d == 1:
            return ffd_1
        else:          
            ffd_tt = operator_to_higherdim(ffd_1, n, d, derivative = derivative)            
    else: 
        raise InvalidArguments('Shape must be a list of TT.Tensors.')    
        
    return ffd_tt.round(1e-14)


def backwardfd(xn, boundarycondition = 'Zero', derivative = None, order = 1, dtype = tn.float64, device = None):
    """
    Construct a d-dimensional backward difference operator with n mesh points in each direction.
    
    Args:
        xn (list[tntt.Tensor): List of the mesh tensors in the TT format.
        derivative (list[int]): List of ones and zeros. Where every one leads to the derivative in this specific direction.
        dtype (torch.dtype, optional): the dtype of the returned tensor. Defaults to tn.float64.
        device (torch.device, optional): the device where the TT cores are created (None means CPU). Defaults to None.

    Raises:
        InvalidArguments: d must be greater than 0.

    Returns:
        torchtt.TT: The d-dimensional backward difference operator.
    """
    if isinstance(xn, list) and isinstance(xn[0], tntt.TT):
        d = len(xn)
        n = xn[0].shape[0]
        h = getstepsize(xn)
        
        if order == 1:
            bfd_1 = -1 * tn.diag(tn.ones(n - 1, dtype = dtype, device = device), -1) + tn.eye(n, dtype = dtype, device = device)
            bfd_1 = set_boundary_conditions(bfd_1, d, Boundary_conditions = boundarycondition)
            bfd_1 = bfd_1 * 1 / h
            bfd_1 = tntt.TT(bfd_1, [(n, n)])
        
        elif order == 2:
            bfd_1 = -2 * tn.diag(tn.ones(n - 1, dtype = dtype, device = device), -1) + tn.eye(n, dtype = dtype, device = device) + tn.diag(tn.ones(n - 2, dtype = dtype, device = device), -2)
            bfd_1 = set_boundary_conditions(bfd_1, d, Boundary_conditions = boundarycondition)
            bfd_1 = bfd_1 * 1 / h ** 2
            bfd_1 = tntt.TT(bfd_1, [(n, n)])
        
        if d == 0:
            raise InvalidArguments('d must be greater than 0')
            
        elif d == 1:
            return bfd_1
        
        else:          
            bfd_tt = operator_to_higherdim(bfd_1, n, d, derivative = derivative)            
    else: 
        raise InvalidArguments('Shape must be a list of TT.Tensors.')    
        
    return bfd_tt.round(1e-14)


def backward2fd(xn, boundarycondition = 'Zero', derivative = None, order = 1, dtype = tn.float64, device = None):
    """
    Construct a d-dimensional backward difference operator of order of approximation 2 and with the possibility of taking the second order of derivative.
    With n mesh points in each direction.
    
    Args:
        xn (list[tntt.Tensor): List of the mesh tensors in the TT format.
        derivative (list[int]): List of ones and zeros. Where every one leads to the derivative in this specific direction.
        order (int): order of derivative. Choose between 1 and 2
        dtype (torch.dtype, optional): the dtype of the returned tensor. Defaults to tn.float64.
        device (torch.device, optional): the device where the TT cores are created (None means CPU). Defaults to None.

    Raises:
        InvalidArguments: d must be greater than 0.

    Returns:
        torchtt.TT: The d-dimensional 2nd order backward difference operator.
    """
    if isinstance(xn, list) and isinstance(xn[0], tntt.TT):
        d = len(xn)
        n = xn[0].shape[0]
        h = getstepsize(xn)
    
        if order == 1:
            b2fd_1 = -4 * tn.diag(tn.ones(n - 1, dtype = dtype, device = device), -1) + 3 * tn.eye(n, dtype = dtype, device = device) + tn.diag(tn.ones(n - 2, dtype = dtype, device = device), -2)
            b2fd_1 = set_boundary_conditions(b2fd_1, d, Boundary_conditions = boundarycondition)
            b2fd_1 = b2fd_1 * 1 / (2 * h)
            b2fd_1 = tntt.TT(b2fd_1, [(n, n)])
            
        elif order == 2:
            b2fd_1 = -5 * tn.diag(tn.ones(n - 1, dtype = dtype, device = device), -1) + 2 * tn.eye(n, dtype = dtype, device = device) + 4*tn.diag(tn.ones(n - 2, dtype = dtype, device = device), -2) - tn.diag(tn.ones(n - 3, dtype = dtype, device = device), -3)
            b2fd_1 = set_boundary_conditions(b2fd_1, d, Boundary_conditions = boundarycondition)
            b2fd_1 = b2fd_1 * 1 / h ** 3
            b2fd_1 = tntt.TT(b2fd_1, [(n, n)])
            
        else:
            raise InvalidArguments('order cant be higher than 2')
        
        if d == 0:
            raise InvalidArguments('d must be greater than 0')
            
        elif d == 1:
            return b2fd_1
        
        else:          
            b2fd_tt = operator_to_higherdim(b2fd_1, n, d, derivative = derivative)            
    else: 
        raise InvalidArguments('Shape must be a list of TT.Tensors.')    
        
    return b2fd_tt.round(1e-14)


def forward2fd(xn, boundarycondition = 'Zero', derivative = None, order = 1, dtype = tn.float64, device = None):
    """
    Construct a d-dimensional forward difference operator of order of approximation 2 and with the possibility of taking the second order of derivative.
    With n mesh points in each direction.
    
    Args:
        xn (list[tntt.Tensor): List of the mesh tensors in the TT format.
        derivative (list[int]): List of ones and zeros. Where every one leads to the derivative in this specific direction.
        order (int): order of derivative. Choose between 1 and 2
        dtype (torch.dtype, optional): the dtype of the returned tensor. Defaults to tn.float64.
        device (torch.device, optional): the device where the TT cores are created (None means CPU). Defaults to None.

    Raises:
        InvalidArguments: d must be greater than 0.

    Returns:
        torchtt.TT: The d-dimensional 2nd order forward difference operator.
    """
    if isinstance(xn, list) and isinstance(xn[0], tntt.TT):
        d = len(xn)
        n = xn[0].shape[0]
        h = getstepsize(xn)
    
        if order == 1:
            f2fd_1 = 4 * tn.diag(tn.ones(n - 1, dtype = dtype, device = device), 1) - 3 * tn.eye(n, dtype = dtype, device = device) - tn.diag(tn.ones(n - 2, dtype = dtype, device = device), 2)
            f2fd_1 = set_boundary_conditions(f2fd_1, d, Boundary_conditions = boundarycondition)
            f2fd_1 = f2fd_1 * 1 / (2 * h)
            f2fd_1 = tntt.TT(f2fd_1, [(n, n)])
            
        elif order == 2:
            
            f2fd_1 = -5 * tn.diag(tn.ones(n - 1, dtype = dtype, device = device), 1) + 2*tn.eye(n, dtype = dtype, device = device) + 4* tn.diag(tn.ones(n - 2, dtype = dtype, device = device), 2) - tn.diag(tn.ones(n - 3, dtype = dtype, device = device), 3)
            f2fd_1 = set_boundary_conditions(f2fd_1, d, Boundary_conditions = boundarycondition)
            f2fd_1 = f2fd_1* 1 / n ** 3
            f2fd_1 = tntt.TT(f2fd_1, [(n, n)])
                      
        else:
            raise InvalidArguments('order cant be higher than 2')
        
        if d == 0:
            raise InvalidArguments('d must be greater than 0')
            
        elif d == 1:
            return f2fd_1
        
        else:          
            f2fd_tt = operator_to_higherdim(f2fd_1, n, d, derivative = derivative)
    else: 
        raise InvalidArguments('Shape must be a list of TT.Tensors.')    
        
    return f2fd_tt.round(1e-14)
'''

