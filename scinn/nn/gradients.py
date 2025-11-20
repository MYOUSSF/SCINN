"""Automatic differentiation utilities for computing gradients."""

__all__ = ["jacobian", "hessian", "grad", "divergence", "laplacian"]

import torch


def grad(y, x, create_graph=True, retain_graph=True):
    """Compute gradient dy/dx.
    
    Args:
        y: Output tensor of shape (batch_size, output_dim)
        x: Input tensor of shape (batch_size, input_dim)
        create_graph: Whether to create computation graph for higher-order derivatives
        retain_graph: Whether to retain the computation graph
        
    Returns:
        Gradient tensor of shape (batch_size, output_dim, input_dim)
    """
    if y.shape[0] == 0:
        return torch.zeros((0,) + x.shape[1:], dtype=y.dtype, device=y.device)
    
    grad_outputs = torch.ones_like(y)
    gradients = torch.autograd.grad(
        outputs=y,
        inputs=x,
        grad_outputs=grad_outputs,
        create_graph=create_graph,
        retain_graph=retain_graph,
        only_inputs=True
    )[0]
    
    return gradients


def jacobian(y, x, create_graph=True):
    """Compute Jacobian matrix.
    
    Args:
        y: Output tensor (batch_size, output_dim)
        x: Input tensor (batch_size, input_dim)
        
    Returns:
        Jacobian tensor of shape (batch_size, output_dim, input_dim)
    """
    batch_size = y.shape[0]
    output_dim = y.shape[1] if y.dim() > 1 else 1
    input_dim = x.shape[1] if x.dim() > 1 else 1
    
    jac = torch.zeros(batch_size, output_dim, input_dim, device=y.device)
    
    for i in range(output_dim):
        if y.dim() > 1:
            y_i = y[:, i:i+1]
        else:
            y_i = y
        
        grad_y_i = grad(y_i, x, create_graph=create_graph)
        
        if grad_y_i.dim() == 1:
            grad_y_i = grad_y_i.unsqueeze(1)
        
        jac[:, i, :] = grad_y_i
    
    return jac


def hessian(y, x):
    """Compute Hessian matrix (for scalar output).
    
    Args:
        y: Scalar output tensor (batch_size, 1)
        x: Input tensor (batch_size, input_dim)
        
    Returns:
        Hessian tensor of shape (batch_size, input_dim, input_dim)
    """
    batch_size = x.shape[0]
    input_dim = x.shape[1] if x.dim() > 1 else 1
    
    # First derivatives
    dy_dx = grad(y, x, create_graph=True)
    
    # Second derivatives
    hess = torch.zeros(batch_size, input_dim, input_dim, device=y.device)
    
    for i in range(input_dim):
        if dy_dx.dim() == 1:
            dy_dx_i = dy_dx
        else:
            dy_dx_i = dy_dx[:, i:i+1]
        
        d2y_dx2 = grad(dy_dx_i, x, create_graph=True)
        
        if d2y_dx2.dim() == 1:
            d2y_dx2 = d2y_dx2.unsqueeze(1)
        
        hess[:, i, :] = d2y_dx2
    
    return hess


def divergence(y, x):
    """Compute divergence of a vector field.
    
    Args:
        y: Vector field tensor (batch_size, dim)
        x: Input tensor (batch_size, dim)
        
    Returns:
        Divergence tensor (batch_size, 1)
    """
    batch_size = y.shape[0]
    dim = y.shape[1]
    
    div = torch.zeros(batch_size, 1, device=y.device)
    
    for i in range(dim):
        y_i = y[:, i:i+1]
        dy_dx = grad(y_i, x, create_graph=True)
        div += dy_dx[:, i:i+1]
    
    return div


def laplacian(y, x):
    """Compute Laplacian (trace of Hessian).
    
    Args:
        y: Scalar output tensor (batch_size, 1)
        x: Input tensor (batch_size, input_dim)
        
    Returns:
        Laplacian tensor (batch_size, 1)
    """
    # First derivatives
    dy_dx = grad(y, x, create_graph=True)
    
    # Sum of second derivatives (Laplacian)
    laplace = torch.zeros_like(y)
    
    input_dim = x.shape[1] if x.dim() > 1 else 1
    for i in range(input_dim):
        if dy_dx.dim() == 1:
            dy_dx_i = dy_dx
        else:
            dy_dx_i = dy_dx[:, i:i+1]
        
        d2y_dx2 = grad(dy_dx_i, x, create_graph=True)
        
        if d2y_dx2.dim() == 1:
            laplace += d2y_dx2.unsqueeze(1)
        else:
            laplace += d2y_dx2[:, i:i+1]
    
    return laplace
