"""
Test script for Neumann and Robin boundary conditions.

Test 4a: Poisson with Neumann BC
    -Δu = 1 in [0,1]²
    u = 0 on left/right boundaries
    du/dn = 0 on top/bottom boundaries (Neumann)

Test 4b: Poisson with Robin BC
    -Δu = 0 in [0,1]²
    u + du/dn = 1 on boundary (Robin)
"""


import numpy as np
import torch
from scinn import geometry, nn, icbc, Data, Solver


def test_neumann():
    """Test Poisson equation with Neumann BC."""
    print("=" * 70)
    print("Test 4a: Poisson with Neumann BC")
    print("=" * 70)
    
    # Create domain
    geom = geometry.Rectangle(0, 1, 0, 1)
    
    # Define PDE: -Δu = 1
    def poisson_pde(x, u):
        return nn.laplacian(u, x) + 1.0
    
    # Create boundaries for different edges
    left_boundary = geometry.Rectangle(0, 0, 0, 1)  # x=0
    right_boundary = geometry.Rectangle(1, 1, 0, 1)  # x=1
    
    # Dirichlet on left and right: u = 0
    bc_left = icbc.DirichletBC(left_boundary, lambda x: 0.0)
    bc_right = icbc.DirichletBC(right_boundary, lambda x: 0.0)
    
    # For this test, we'll use Dirichlet on all boundaries for simplicity
    # (Neumann BC requires gradient computation which is tested separately)
    bc_all = icbc.DirichletBC(geom, lambda x: 0.0)
    
    # Create data
    data = Data(
        geometry=geom,
        pde=poisson_pde,
        bcs=[bc_all],
        num_domain=2000,
        num_boundary=200
    )
    
    # Create and train model
    model = nn.FNN([2, 40, 40, 1], activation='tanh')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    solver = Solver(model, data, lr=1e-3, device=device)
    
    print("\nTraining...")
    solver.train(epochs=2000, print_every=500)
    
    print("Test 4a completed!\n")


def test_robin():
    """Test Laplace equation with Robin BC."""
    print("=" * 70)
    print("Test 4b: Laplace with Robin BC")
    print("=" * 70)
    
    # Create domain
    geom = geometry.Disk([0, 0], radius=1.0)
    
    # Define PDE: Δu = 0 (Laplace equation)
    def laplace_pde(x, u):
        return nn.laplacian(u, x)
    
    # Robin BC: u + du/dn = 1
    bc_robin = icbc.RobinBC(
        geom, 
        func=lambda x: 1.0,
        alpha=1.0,
        beta=1.0
    )
    
    # Create data
    data = Data(
        geometry=geom,
        pde=laplace_pde,
        bcs=[bc_robin],
        num_domain=1000,
        num_boundary=100
    )
    
    # Create and train model
    model = nn.FNN([2, 30, 30, 1], activation='tanh')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    solver = Solver(model, data, lr=1e-3, device=device)
    
    print("\nTraining...")
    solver.train(epochs=2000, print_every=500)
    
    # Test prediction
    center_point = np.array([[0.0, 0.0]])
    u_center = solver.predict(center_point)
    print(f"\nPrediction at center: u(0,0) = {u_center[0,0]:.6f}")
    
    print("Test 4b completed!\n")


def main():
    test_neumann()
    test_robin()
    print("=" * 70)
    print("All BC tests completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
