"""Initial and boundary conditions with simplified API."""

__all__ = ["DirichletBC", "NeumannBC", "RobinBC", "PeriodicBC", "IC"]

import torch
import numpy as np


class BC:
    """Base class for boundary conditions."""
    
    def __init__(self, geometry, func, component=0, sampling="boundary"):
        """
        Args:
            geometry: Geometry object
            func: Function that defines the BC value, func(x) -> value
            component: Which component of the solution this BC applies to
            sampling: "boundary" (default) or "domain". Where to sample points.
        """
        self.geometry = geometry
        self.func = func
        self.component = component
        self.sampling = sampling
    
    def sample_points(self, n, random="pseudo"):
        """Sample points on the boundary or domain."""
        if self.sampling == "boundary":
            return self.geometry.random_boundary_points(n, random)
        elif self.sampling == "domain":
            return self.geometry.random_points(n, random)
        else:
            raise ValueError(f"Unknown sampling method: {self.sampling}")


class DirichletBC(BC):
    """Dirichlet boundary condition: u = func(x) on boundary.
    
    Args:
        geometry: Geometry object defining the domain
        func: Function that returns the boundary value, func(x) -> value
        component: Which output component this BC applies to (default: 0)
    
    Example:
        # u = 0 on boundary
        bc = DirichletBC(geometry, lambda x: 0)
        
        # u = sin(x) on boundary  
        bc = DirichletBC(geometry, lambda x: np.sin(x[:, 0]))
    """
    
    def __init__(self, geometry, func, component=0, sampling="boundary"):
        super().__init__(geometry, func, component, sampling)
        self.type = "dirichlet"
    
    def error(self, x, u):
        """Compute the error: u - func(x)."""
        target = self.func(x)
        if isinstance(target, (int, float)):
            target = torch.full_like(u[:, self.component:self.component+1], target)
        elif isinstance(target, np.ndarray):
            target = torch.from_numpy(target).float().to(u.device)
            if target.dim() == 1:
                target = target.unsqueeze(1)
        
        return u[:, self.component:self.component+1] - target


class NeumannBC(BC):
    """Neumann boundary condition: du/dn = func(x) on boundary.
    
    Args:
        geometry: Geometry object defining the domain
        func: Function that returns the normal derivative value
        component: Which output component this BC applies to
    
    Example:
        # du/dn = 0 on boundary (homogeneous Neumann)
        bc = NeumannBC(geometry, lambda x: 0)
    """
    
    def __init__(self, geometry, func, component=0, sampling="boundary"):
        super().__init__(geometry, func, component, sampling)
        self.type = "neumann"
    
    def error(self, x, u, grad_u):
        """Compute the error in the normal derivative.
        
        Args:
            x: Boundary points (numpy array)
            u: Solution values at x
            grad_u: Gradient of u at x
        """
        # Get outward normal
        normal = self.geometry.boundary_normal(x)
        normal_torch = torch.from_numpy(normal).float().to(u.device)
        
        # Compute normal derivative
        du_dn = torch.sum(grad_u * normal_torch, dim=1, keepdim=True)
        
        # Get target value
        target = self.func(x)
        if isinstance(target, (int, float)):
            target = torch.full_like(du_dn, target)
        elif isinstance(target, np.ndarray):
            target = torch.from_numpy(target).float().to(u.device)
            if target.dim() == 1:
                target = target.unsqueeze(1)
        
        return du_dn - target


class RobinBC(BC):
    """Robin boundary condition: alpha*u + beta*du/dn = func(x) on boundary.
    
    Args:
        geometry: Geometry object
        func: Function that returns RHS value
        alpha, beta: Coefficients (can be constants or functions)
        component: Which output component this BC applies to
    
    Example:
        # u + du/dn = 1 on boundary
        bc = RobinBC(geometry, lambda x: 1, alpha=1, beta=1)
    """
    
    def __init__(self, geometry, func, alpha=1, beta=1, component=0, sampling="boundary"):
        super().__init__(geometry, func, component, sampling)
        self.alpha = alpha
        self.beta = beta
        self.type = "robin"
    
    def error(self, x, u, grad_u):
        """Compute the Robin BC error."""
        # Get coefficients
        if callable(self.alpha):
            alpha = torch.from_numpy(self.alpha(x)).float().to(u.device)
        else:
            alpha = self.alpha
        
        if callable(self.beta):
            beta = torch.from_numpy(self.beta(x)).float().to(u.device)
        else:
            beta = self.beta
        
        # Get normal derivative
        normal = self.geometry.boundary_normal(x)
        normal_torch = torch.from_numpy(normal).float().to(u.device)
        du_dn = torch.sum(grad_u * normal_torch, dim=1, keepdim=True)
        
        # Compute LHS
        lhs = alpha * u[:, self.component:self.component+1] + beta * du_dn
        
        # Get RHS
        target = self.func(x)
        if isinstance(target, (int, float)):
            target = torch.full_like(lhs, target)
        elif isinstance(target, np.ndarray):
            target = torch.from_numpy(target).float().to(u.device)
            if target.dim() == 1:
                target = target.unsqueeze(1)
        
        return lhs - target


class PeriodicBC(BC):
    """Periodic boundary condition.
    
    Args:
        geometry: Geometry object
        component_x: Which spatial dimension is periodic
        component_u: Which output component this BC applies to
    
    Example:
        # u(x=0) = u(x=1) for x-direction
        bc = PeriodicBC(geometry, component_x=0)
    """
    
    def __init__(self, geometry, component_x, component_u=0):
        super().__init__(geometry, lambda x: 0, component_u)
        self.component_x = component_x
        self.type = "periodic"
    
    def sample_point_pairs(self, n, random="pseudo"):
        """Sample pairs of corresponding periodic points."""
        # This is geometry-specific; for now, use a simple approach
        # For intervals: pair left and right boundaries
        # This needs to be customized based on the geometry
        raise NotImplementedError("Periodic BC sampling not yet implemented")


class IC:
    """Initial condition: u(t=t0, x) = func(x).
    
    Args:
        geometry: Spatial geometry
        func: Function that returns initial values, func(x) -> value
        component: Which output component this IC applies to
    
    Example:
        # u(t=0, x) = sin(pi*x)
        ic = IC(geometry, lambda x: np.sin(np.pi * x[:, 0]))
    """
    
    def __init__(self, geometry, func, component=0, sampling="domain"):
        self.geometry = geometry
        self.func = func
        self.component = component
        self.sampling = sampling
        self.type = "initial"
    
    def sample_points(self, n, random="pseudo"):
        """Sample points in the spatial domain."""
        if self.sampling == "domain":
            return self.geometry.random_points(n, random)
        elif self.sampling == "boundary":
            return self.geometry.random_boundary_points(n, random)
        else:
            raise ValueError(f"Unknown sampling method: {self.sampling}")
    
    def error(self, x, u):
        """Compute the IC error: u - func(x)."""
        target = self.func(x)
        if isinstance(target, (int, float)):
            target = torch.full_like(u[:, self.component:self.component+1], target)
        elif isinstance(target, np.ndarray):
            target = torch.from_numpy(target).float().to(u.device)
            if target.dim() == 1:
                target = target.unsqueeze(1)
        
        return u[:, self.component:self.component+1] - target
