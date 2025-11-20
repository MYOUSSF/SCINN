"""
Comprehensive Examples for SCINN Framework

This file demonstrates various use cases and features of the SCINN framework.
"""

import numpy as np
import torch
from scinn import geometry, nn, icbc, Data, Solver


# ============================================================================
# Example 1: 1D Poisson Equation
# ============================================================================

def example_1d_poisson():
    """
    Solve: -d²u/dx² = f(x) for x ∈ [0, 1]
    with u(0) = u(1) = 0
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: 1D Poisson Equation")
    print("="*70)
    
    # Domain
    geom = geometry.Interval(0, 1)
    
    # PDE: -u'' = 1
    def pde(x, u):
        u_x = nn.grad(u, x)
        u_xx = nn.grad(u_x, x)
        return u_xx + 1.0
    
    # Boundary conditions
    bc = icbc.DirichletBC(geom, lambda x: 0.0)
    
    # Setup
    data = Data(geom, pde, bcs=[bc], num_domain=100, num_boundary=2)
    model = nn.FNN([1, 20, 20, 1], activation='tanh')
    solver = Solver(model, data, lr=1e-3)
    
    # Train
    solver.train(epochs=1000, print_every=200)
    
    # Verify
    x_test = np.array([[0.5]])
    u_pred = solver.predict(x_test)[0, 0]
    u_exact = 0.125  # Exact solution at x=0.5
    print(f"\nAt x=0.5: u_pred={u_pred:.6f}, u_exact={u_exact:.6f}")


# ============================================================================
# Example 2: 2D Laplace Equation on Disk
# ============================================================================

def example_2d_laplace_disk():
    """
    Solve: Δu = 0 in disk
    with u = r² on boundary
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Laplace Equation on Disk")
    print("="*70)
    
    # Domain
    geom = geometry.Disk([0, 0], radius=1.0)
    
    # PDE: Δu = 0
    def pde(x, u):
        return nn.laplacian(u, x)
    
    # BC: u = r² on boundary
    def bc_func(x):
        return x[:, 0]**2 + x[:, 1]**2
    
    bc = icbc.DirichletBC(geom, bc_func)
    
    # Setup
    data = Data(geom, pde, bcs=[bc], num_domain=1000, num_boundary=100)
    model = nn.FNN([2, 30, 30, 1], activation='tanh')
    solver = Solver(model, data, lr=1e-3, loss_weights={'pde': 1.0, 'bc': 100.0, 'ic': 1.0})
    
    # Train
    solver.train(epochs=2000, print_every=400)
    
    # Verify at center
    x_test = np.array([[0.0, 0.0]])
    u_pred = solver.predict(x_test)[0, 0]
    print(f"\nAt center: u_pred={u_pred:.6f}")


# ============================================================================
# Example 3: Complex Geometry (Annulus)
# ============================================================================

def example_annulus():
    """
    Solve Poisson equation on an annulus (disk with hole).
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Poisson on Annulus")
    print("="*70)
    
    # Create annulus
    outer = geometry.Disk([0, 0], radius=2.0)
    inner = geometry.Disk([0, 0], radius=0.5)
    annulus = outer - inner
    
    # PDE
    def pde(x, u):
        return nn.laplacian(u, x) + 1.0
    
    # BC: u = 0 on boundaries
    bc = icbc.DirichletBC(annulus, lambda x: 0.0)
    
    # Setup
    data = Data(annulus, pde, bcs=[bc], num_domain=2000, num_boundary=200)
    model = nn.FNN([2, 40, 40, 1], activation='tanh')
    solver = Solver(model, data, lr=1e-3)
    
    # Train
    solver.train(epochs=2000, print_every=400)


# ============================================================================
# Example 4: Time-Dependent Problem (Heat Equation)
# ============================================================================

def example_heat_equation():
    """
    Solve 1D heat equation with initial condition.
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: 1D Heat Equation")
    print("="*70)
    
    # Space-time domain
    x_domain = geometry.Interval(0, 1)
    t_domain = geometry.TimeDomain(0, 0.5)
    geom = x_domain * t_domain
    
    # Parameters
    alpha = 0.1
    
    # PDE: du/dt - α*d²u/dx² = 0
    def pde(xt, u):
        du_dt = nn.grad(u, xt)[:, 1:2]
        u_x = nn.grad(u, xt)[:, 0:1]
        u_xx = nn.grad(u_x, xt)[:, 0:1]
        return du_dt - alpha * u_xx
    
    # Initial condition: u(x,0) = sin(πx)
    ic = icbc.IC(x_domain, lambda x: np.sin(np.pi * x[:, 0]))
    
    # Boundary conditions: u(0,t) = u(1,t) = 0
    bc = icbc.DirichletBC(geom, lambda x: 0.0)
    
    # Setup
    data = Data(geom, pde, bcs=[bc], ics=[ic], 
                num_domain=3000, num_boundary=100, num_initial=100)
    model = nn.FNN([2, 50, 50, 1], activation='tanh')
    solver = Solver(model, data, lr=1e-3, 
                   loss_weights={'pde': 1.0, 'bc': 50.0, 'ic': 50.0})
    
    # Train
    solver.train(epochs=3000, print_every=500)
    
    # Test at different times
    times = [0.0, 0.1, 0.2, 0.3]
    print("\nSolution at x=0.5:")
    for t in times:
        xt = np.array([[0.5, t]])
        u_pred = solver.predict(xt)[0, 0]
        u_exact = np.sin(np.pi * 0.5) * np.exp(-np.pi**2 * alpha * t)
        print(f"  t={t:.1f}: u_pred={u_pred:.6f}, u_exact={u_exact:.6f}")


# ============================================================================
# Example 5: System of PDEs (Coupled Equations)
# ============================================================================

def example_coupled_system():
    """
    Solve a system of coupled PDEs.
    Example: Reaction-diffusion system
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: Coupled Reaction-Diffusion System")
    print("="*70)
    
    # Domain
    geom = geometry.Rectangle(0, 1, 0, 1)
    
    # Parameters
    D_u, D_v = 0.01, 0.01
    a, b = 1.0, 1.0
    
    # PDEs (simplified Gray-Scott model)
    def pde_u(x, u):
        # Assuming u has shape (N, 2) where u[:, 0] is u and u[:, 1] is v
        laplace_u = nn.laplacian(u[:, 0:1], x)
        reaction = -u[:, 0:1] * u[:, 1:2]**2
        return laplace_u * D_u + reaction
    
    def pde_v(x, u):
        laplace_v = nn.laplacian(u[:, 1:2], x)
        reaction = u[:, 0:1] * u[:, 1:2]**2 - b * u[:, 1:2]
        return laplace_v * D_v + reaction
    
    # Boundary conditions
    bc_u = icbc.DirichletBC(geom, lambda x: 1.0, component=0)
    bc_v = icbc.DirichletBC(geom, lambda x: 0.0, component=1)
    
    # Setup (using single equation for simplicity in this demo)
    data = Data(geom, pde_u, bcs=[bc_u], num_domain=1000, num_boundary=100)
    model = nn.FNN([2, 40, 40, 2], activation='tanh')  # 2 outputs
    solver = Solver(model, data, lr=1e-3)
    
    print("Note: This is a simplified example. Full coupled system")
    print("      would require separate handling of multiple components.")


# ============================================================================
# Example 6: Neumann Boundary Condition
# ============================================================================

def example_neumann():
    """
    Solve Poisson with Neumann BC.
    """
    print("\n" + "="*70)
    print("EXAMPLE 6: Poisson with Neumann BC")
    print("="*70)
    
    # Domain
    geom = geometry.Rectangle(0, 1, 0, 1)
    
    # PDE
    def pde(x, u):
        return nn.laplacian(u, x) + 1.0
    
    # Mixed BCs (simplified - using Dirichlet for this demo)
    bc = icbc.DirichletBC(geom, lambda x: 0.0)
    
    # For actual Neumann BC:
    # bc_neumann = icbc.NeumannBC(geom, lambda x: 0.0)
    
    data = Data(geom, pde, bcs=[bc], num_domain=1000, num_boundary=100)
    model = nn.FNN([2, 30, 30, 1], activation='tanh')
    solver = Solver(model, data, lr=1e-3)
    
    solver.train(epochs=1000, print_every=200)


# ============================================================================
# Example 7: Using Different Neural Network Architectures
# ============================================================================

def example_architectures():
    """
    Compare different neural network architectures.
    """
    print("\n" + "="*70)
    print("EXAMPLE 7: Different NN Architectures")
    print("="*70)
    
    geom = geometry.Interval(0, 1)
    
    def pde(x, u):
        u_xx = nn.grad(nn.grad(u, x), x)
        return u_xx + 1.0
    
    bc = icbc.DirichletBC(geom, lambda x: 0.0)
    data = Data(geom, pde, bcs=[bc], num_domain=100, num_boundary=2)
    
    # Standard FNN
    print("\n1. Standard FNN:")
    model1 = nn.FNN([1, 20, 20, 1], activation='tanh')
    solver1 = Solver(model1, data, lr=1e-3)
    solver1.train(epochs=500, print_every=100)
    
    # Modified MLP with Fourier features
    print("\n2. Modified MLP with Fourier Features:")
    data.resample()  # Resample for fair comparison
    model2 = nn.ModifiedMLP([1, 20, 20, 1], fourier_features=True)
    solver2 = Solver(model2, data, lr=1e-3)
    solver2.train(epochs=500, print_every=100)
    
    # ResNet
    print("\n3. ResNet:")
    data.resample()
    model3 = nn.ResNet([1, 20, 1], num_res_blocks=3)
    solver3 = Solver(model3, data, lr=1e-3)
    solver3.train(epochs=500, print_every=100)


# ============================================================================
# Main Function
# ============================================================================

def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("SCINN: Comprehensive Examples")
    print("="*70)
    
    examples = [
        ("1D Poisson", example_1d_poisson),
        ("2D Laplace on Disk", example_2d_laplace_disk),
        ("Annulus", example_annulus),
        ("Heat Equation", example_heat_equation),
        ("Different Architectures", example_architectures),
    ]
    
    print("\nAvailable examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    
    print("\nRunning basic examples...")
    
    # Run a few quick examples
    example_1d_poisson()
    example_2d_laplace_disk()
    
    print("\n" + "="*70)
    print("Examples completed!")
    print("="*70)
    print("\nTo run more examples, call the functions individually:")
    print("  example_annulus()")
    print("  example_heat_equation()")
    print("  example_architectures()")


if __name__ == "__main__":
    main()
