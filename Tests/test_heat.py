"""
Test script for solving the 1D heat equation:
    du/dt = α d²u/dx² for x ∈ [0,1], t ∈ [0,1]
    u(0,t) = u(1,t) = 0 (Dirichlet BCs)
    u(x,0) = sin(πx) (Initial condition)
"""


import os
import numpy as np
import torch
from scinn import geometry, nn, icbc, Data, Solver


def main():
    print("=" * 70)
    print("Test 2: 1D Heat Equation")
    print("=" * 70)
    
    # Define domain: space × time
    x_domain = geometry.Interval(0, 1)
    t_domain = geometry.TimeDomain(0, 1)
    geom = x_domain * t_domain  # Cross product: [0,1] × [0,1]
    
    # Thermal diffusivity
    alpha = 0.1
    
    # Define PDE: du/dt - α*d²u/dx² = 0
    def heat_pde(xt, u):
        """Heat equation residual"""
        # xt[:, 0] is x, xt[:, 1] is t
        du_dt = nn.grad(u, xt)[:, 1:2]  # ∂u/∂t
        
        # Compute second derivative w.r.t. x
        u_x = nn.grad(u, xt)[:, 0:1]  # ∂u/∂x
        u_xx = nn.grad(u_x, xt)[:, 0:1]  # ∂²u/∂x²
        
        residual = du_dt - alpha * u_xx
        return residual
    
    # Boundary conditions: u(0,t) = 0 and u(1,t) = 0
    # We need to define these as 2D geometries (lines in space-time)
    
    # Left boundary: x=0, t in [0,1]
    geom_left = geometry.Interval(0, 0) * t_domain
    bc_left = icbc.DirichletBC(geom_left, lambda x: 0.0, sampling="domain")
    
    # Right boundary: x=1, t in [0,1]
    geom_right = geometry.Interval(1, 1) * t_domain
    bc_right = icbc.DirichletBC(geom_right, lambda x: 0.0, sampling="domain")
    
    # Initial condition: u(x,0) = sin(πx)
    # Initial domain: x in [0,1], t=0
    geom_ic = x_domain * geometry.Interval(0, 0)
    ic = icbc.IC(geom_ic, lambda x: np.sin(np.pi * x[:, 0]), sampling="domain")
    
    # Create data object
    data = Data(
        geometry=geom,
        pde=heat_pde,
        bcs=[bc_left, bc_right],
        ics=[ic],
        num_domain=5000,
        num_boundary=100,
        num_initial=100,
        num_test=1000
    )
    
    # Create neural network
    model = nn.FNN(
        layer_sizes=[2, 50, 50, 50, 1],  # Input: (x, t), Output: u
        activation='tanh'
    )
    
    # Create solver
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    solver = Solver(
        model=model,
        data=data,
        lr=1e-3,
        loss_weights={'pde': 1.0, 'bc': 50.0, 'ic': 50.0},
        device=device
    )
    
    # Train
    solver.train(epochs=5000, print_every=500, resample_every=1000)
    
    # Test predictions at different times
    print("\nPredictions at x=0.5 for different times:")
    test_times = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    test_points = np.column_stack([np.full_like(test_times, 0.5), test_times])
    predictions = solver.predict(test_points)
    
    for i, t in enumerate(test_times):
        u_pred = predictions[i, 0]
        # Exact solution: u(x,t) = sin(πx) * exp(-π²αt)
        u_exact = np.sin(np.pi * 0.5) * np.exp(-np.pi**2 * alpha * t)
        error = abs(u_pred - u_exact)
        print(f"  t={t:.2f}: u_pred={u_pred:.6f}, u_exact={u_exact:.6f}, error={error:.6f}")
    
    # Save model
    model_path = os.path.join(os.path.dirname(__file__), '..', 'model_heat.pt')
    solver.save(model_path)
    print("\nTest 2 completed successfully!")


if __name__ == "__main__":
    main()
