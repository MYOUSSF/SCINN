"""
Test script for solving the 2D Poisson equation:
    -Δu = 1 in Ω = [0,1] × [0,1]
    u = 0 on ∂Ω

Exact solution: u(x,y) = x(1-x)y(1-y) / 2 (for f=1)
"""


import numpy as np
import torch
import matplotlib.pyplot as plt
from scinn import geometry, nn, icbc, Data, Solver


def main():
    print("=" * 70)
    print("Test 1: 2D Poisson Equation")
    print("=" * 70)
    
    # Define domain
    geom = geometry.Rectangle(0, 1, 0, 1)
    
    # Define PDE: -Δu = 1
    def poisson_pde(x, u):
        """PDE residual: Δu + 1 = 0"""
        laplacian_u = nn.laplacian(u, x)
        return laplacian_u + 1.0
    
    # Define boundary condition: u = 0 on boundary
    bc = icbc.DirichletBC(geom, lambda x: 0.0)
    
    # Create data object
    data = Data(
        geometry=geom,
        pde=poisson_pde,
        bcs=[bc],
        num_domain=2000,
        num_boundary=200,
        num_test=1000
    )
    
    # Create neural network
    model = nn.FNN(
        layer_sizes=[2, 50, 50, 50, 1],
        activation='tanh'
    )
    
    # Create solver
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    solver = Solver(
        model=model,
        data=data,
        lr=1e-3,
        loss_weights={'pde': 1.0, 'bc': 100.0, 'ic': 0.0},
        device=device
    )
    
    # Train
    solver.train(epochs=5000, print_every=500)
    
    # Test predictions
    test_points = geom.random_points(100)
    predictions = solver.predict(test_points)
    
    print("\nSample predictions:")
    for i in range(min(5, len(test_points))):
        x, y = test_points[i]
        u_pred = predictions[i, 0]
        print(f"  Point ({x:.3f}, {y:.3f}): u = {u_pred:.6f}")
    
    # Save model
    solver.save('model_poisson.pt')
    print("\nTest 1 completed successfully!")
    

if __name__ == "__main__":
    main()
