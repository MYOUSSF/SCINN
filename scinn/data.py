"""Data class for organizing PINNs problem setup."""

__all__ = ["Data"]

import numpy as np
import torch


class Data:
    """Container for PINN problem data.
    
    Args:
        geometry: Computational domain (Geometry object)
        pde: PDE equation(s) as a function or list of functions
        bcs: List of boundary conditions
        ics: List of initial conditions (for time-dependent problems)
        num_domain: Number of collocation points in domain
        num_boundary: Number of points on each boundary
        num_initial: Number of initial condition points
        num_test: Number of test points for evaluation
        measurement_data: Dictionary containing measurement data with keys:
            - 'x': numpy array of shape (n_measurements, input_dim) containing input coordinates
            - 'u': numpy array of shape (n_measurements, output_dim) containing measured values
        sampler: Sampling method ("pseudo", "LHS", "Halton", "Sobol")
        
    Example:
        from scinn import geometry, icbc
        
        # Define domain
        geom = geometry.Rectangle(0, 1, 0, 1)
        
        # Define PDE (Poisson equation: -Δu = f)
        def pde(x, u):
            return laplacian(u, x) + 1
        
        # Define BCs
        bc = icbc.DirichletBC(geom, lambda x: 0)
        
        # Add measurement data (optional)
        measurements = {
            'x': np.array([[0.5, 0.5], [0.3, 0.7]]),  # measurement locations
            'u': np.array([[0.2], [0.15]])             # measured values
        }
        
        # Create data object
        data = Data(geom, pde, [bc], num_domain=1000, num_boundary=100,
                    measurement_data=measurements)
    """
    
    def __init__(self, 
                 geometry,
                 pde,
                 bcs=None,
                 ics=None,
                 num_domain=1000,
                 num_boundary=100,
                 num_initial=100,
                 num_test=1000,
                 measurement_data=None,
                 sampler="pseudo"):
        
        self.geometry = geometry
        self.pde = pde if isinstance(pde, list) else [pde]
        self.bcs = bcs if bcs is not None else []
        self.ics = ics if ics is not None else []
        self.num_domain = num_domain
        self.num_boundary = num_boundary
        self.num_initial = num_initial
        self.num_test = num_test
        self.sampler = sampler
        
        # Store measurement data
        self.measurement_data = self._validate_measurement_data(measurement_data)
        
        # Pre-sample some points (can be regenerated during training)
        self.resample()
    
    def _validate_measurement_data(self, measurement_data):
        """Validate and process measurement data.
        
        Args:
            measurement_data: Dictionary with 'x' and 'u' keys, or None
            
        Returns:
            Validated measurement data dictionary or None
        """
        if measurement_data is None:
            return None
        
        if not isinstance(measurement_data, dict):
            raise TypeError("measurement_data must be a dictionary with keys 'x' and 'u'")
        
        if 'x' not in measurement_data or 'u' not in measurement_data:
            raise ValueError("measurement_data must contain both 'x' and 'u' keys")
        
        x_data = np.array(measurement_data['x'])
        u_data = np.array(measurement_data['u'])
        
        # Ensure proper shapes
        if x_data.ndim == 1:
            x_data = x_data.reshape(-1, 1)
        if u_data.ndim == 1:
            u_data = u_data.reshape(-1, 1)
        
        if x_data.shape[0] != u_data.shape[0]:
            raise ValueError(f"Number of measurement points ({x_data.shape[0]}) "
                           f"must match number of values ({u_data.shape[0]})")
        
        # Check if measurement points are within the geometry
        n_outside = np.sum(~self.geometry.inside(x_data))
        if n_outside > 0:
            print(f"Warning: {n_outside} measurement points are outside the domain")
        
        print(f"Loaded {x_data.shape[0]} measurement points with {u_data.shape[1]} component(s)")
        
        return {
            'x': x_data,
            'u': u_data
        }
    
    def resample(self):
        """Resample all collocation points."""
        # Domain points for PDE residual
        self.train_x_domain = self.geometry.random_points(
            self.num_domain, random=self.sampler
        )
        
        # Boundary points for BCs
        self.train_x_bc = []
        for bc in self.bcs:
            x_bc = bc.sample_points(self.num_boundary, random=self.sampler)
            self.train_x_bc.append(x_bc)
        
        # Initial condition points
        self.train_x_ic = []
        for ic in self.ics:
            x_ic = ic.sample_points(self.num_initial, random=self.sampler)
            
            # For time-dependent problems: if IC geometry has lower dimension than full geometry,
            # append zeros to match the full geometry dimension (e.g., append t=0 for space-time)
            if x_ic.shape[1] < self.geometry.dim:
                # Append zeros for the missing dimensions (typically time)
                n_missing = self.geometry.dim - x_ic.shape[1]
                zeros = np.zeros((x_ic.shape[0], n_missing))
                x_ic = np.column_stack((x_ic, zeros))
            
            self.train_x_ic.append(x_ic)
        
        # Test points for evaluation
        self.test_x = self.geometry.random_points(
            self.num_test, random=self.sampler
        )
    
    def get_domain_points(self):
        """Get domain collocation points."""
        return self.train_x_domain
    
    def get_boundary_points(self, bc_idx=None):
        """Get boundary points for specific BC or all BCs."""
        if bc_idx is not None:
            return self.train_x_bc[bc_idx]
        return self.train_x_bc
    
    def get_initial_points(self, ic_idx=None):
        """Get initial condition points."""
        if ic_idx is not None:
            return self.train_x_ic[ic_idx]
        return self.train_x_ic
    
    def get_test_points(self):
        """Get test points."""
        return self.test_x
    
    def get_measurement_data(self):
        """Get measurement data points and values.
        
        Returns:
            Dictionary with 'x' and 'u' keys, or None if no measurements
        """
        return self.measurement_data
    
    def has_measurements(self):
        """Check if measurement data is available."""
        return self.measurement_data is not None
    
    def to_tensor(self, x, device='cpu', requires_grad=True):
        """Convert numpy array to PyTorch tensor."""
        if isinstance(x, torch.Tensor):
            tensor = x.to(device)
        else:
            tensor = torch.from_numpy(x).float().to(device)
        
        if requires_grad:
            tensor.requires_grad_(True)
        
        return tensor