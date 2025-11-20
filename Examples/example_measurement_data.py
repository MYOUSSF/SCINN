"""
Example: Using Measurement Data with PINNs

This example demonstrates how to incorporate known measurement data into PINN training.
We'll solve a 2D Poisson equation with some known measurements.

Problem: -Δu = 1 on [0,1]×[0,1] with u = 0 on boundary

We'll generate synthetic measurements and train a PINN to fit both the PDE and the data.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import cm

# Import the SCINN framework
import scinn
from scinn import geometry, icbc
from scinn.nn import FNN, laplacian


def analytical_solution(x):
    """Analytical solution for -Δu = 1 with u = 0 on boundary of [0,1]²
    
    This is a simplified approximation for demonstration.
    """
    # Using separation of variables, the exact solution involves infinite series
    # For simplicity, we use a polynomial approximation
    return x[:, 0] * (1 - x[:, 0]) * x[:, 1] * (1 - x[:, 1])


def generate_measurement_data(n_points=50, noise_level=0.01):
    """Generate synthetic measurement data with optional noise.
    
    Args:
        n_points: Number of measurement points
        noise_level: Standard deviation of Gaussian noise
        
    Returns:
        Dictionary with 'x' and 'u' keys
    """
    # Generate random points in the domain
    x_measurements = np.random.rand(n_points, 2)
    
    # Compute analytical solution at these points
    u_measurements = analytical_solution(x_measurements).reshape(-1, 1)
    
    # Add noise
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, u_measurements.shape)
        u_measurements += noise
    
    print(f"Generated {n_points} measurement points with noise level {noise_level}")
    
    return {
        'x': x_measurements,
        'u': u_measurements
    }


def example_with_measurements():
    """Example: PINN with measurement data"""
    
    print("="*80)
    print("Example: PINN with Measurement Data")
    print("="*80)
    
    # 1. Define geometry
    geom = geometry.Rectangle(0, 1, 0, 1)
    
    # 2. Define PDE: -Δu = 1
    def pde(x, u):
        return laplacian(u, x) + 1.0
    
    # 3. Define boundary conditions: u = 0 on boundary
    bc = icbc.DirichletBC(geom, lambda x: 0.0)
    
    # 4. Generate measurement data
    measurements = generate_measurement_data(n_points=100, noise_level=0.005)
    
    # 5. Create Data object with measurements
    data = scinn.Data(
        geometry=geom,
        pde=pde,
        bcs=[bc],
        num_domain=2000,
        num_boundary=200,
        measurement_data=measurements  # Add measurements here
    )
    
    # 6. Create neural network
    model = FNN(
        layer_sizes=[2, 50, 50, 50, 1],
        activation='tanh'
    )
    
    # 7. Create solver with data loss weight
    # Higher data weight means the model will fit measurements more closely
    solver = scinn.Solver(
        model=model,
        data=data,
        lr=1e-3,
        loss_weights={
            'pde': 1.0,
            'bc': 10.0,
            'ic': 1.0,
            'data': 50.0  # Weight for measurement data fitting
        }
    )
    
    # 8. Train
    solver.train(epochs=5000, print_every=500)
    
    # 9. Evaluate at measurement locations
    print("\n" + "="*80)
    print("Evaluation at Measurement Locations:")
    print("="*80)
    results = solver.evaluate_at_measurements()
    
    return solver, data, results


def example_without_measurements():
    """Example: Standard PINN without measurement data (for comparison)"""
    
    print("\n" + "="*80)
    print("Example: Standard PINN (No Measurement Data)")
    print("="*80)
    
    # Same setup but without measurements
    geom = geometry.Rectangle(0, 1, 0, 1)
    
    def pde(x, u):
        return laplacian(u, x) + 1.0
    
    bc = icbc.DirichletBC(geom, lambda x: 0.0)
    
    data = scinn.Data(
        geometry=geom,
        pde=pde,
        bcs=[bc],
        num_domain=2000,
        num_boundary=200
    )
    
    model = FNN(
        layer_sizes=[2, 50, 50, 50, 1],
        activation='tanh'
    )
    
    solver = scinn.Solver(
        model=model,
        data=data,
        lr=1e-3
    )
    
    solver.train(epochs=5000, print_every=500)
    
    return solver, data


def visualize_results(solver, data, results=None):
    """Visualize the solution and measurement points"""
    
    # Create a grid for visualization
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x, y)
    xy = np.column_stack([X.ravel(), Y.ravel()])
    
    # Predict
    u_pred = solver.predict(xy).reshape(X.shape)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Solution
    ax1 = axes[0]
    contour = ax1.contourf(X, Y, u_pred, levels=20, cmap='viridis')
    plt.colorbar(contour, ax=ax1)
    
    if results is not None:
        # Overlay measurement points
        x_meas = results['x']
        ax1.scatter(x_meas[:, 0], x_meas[:, 1], c='red', s=20, 
                   marker='o', label='Measurements', edgecolors='white', linewidths=0.5)
        ax1.legend()
    
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('PINN Solution')
    ax1.set_aspect('equal')
    
    # Plot 2: Training history
    ax2 = axes[1]
    history = solver.get_history()
    epochs = range(1, len(history['loss']) + 1)
    
    ax2.semilogy(epochs, history['loss_pde'], label='PDE Loss', alpha=0.7)
    ax2.semilogy(epochs, history['loss_bc'], label='BC Loss', alpha=0.7)
    if data.has_measurements():
        ax2.semilogy(epochs, history['loss_data'], label='Data Loss', alpha=0.7)
    ax2.semilogy(epochs, history['loss'], label='Total Loss', linewidth=2, color='black')
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training History')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pinn_with_measurements.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved as 'pinn_with_measurements.png'")


def time_dependent_example():
    """Example: Time-dependent problem with measurement data
    
    Problem: Heat equation ∂u/∂t = α∇²u on [0,1]×[0,T]
    """
    
    print("\n" + "="*80)
    print("Example: Time-Dependent Problem with Measurements")
    print("="*80)
    
    # Spatial domain [0, 1] and time domain [0, 0.5]
    spatial_geom = geometry.Interval(0, 1)
    time_geom = geometry.TimeDomain(0, 0.5)
    geom = spatial_geom * time_geom  # Cross product for space-time
    
    # Heat equation with α = 0.1
    alpha = 0.1
    
    def heat_pde(xt, u):
        """Heat equation: ∂u/∂t - α∂²u/∂x² = 0"""
        from scinn.nn import grad
        
        # xt has shape (batch, 2) where xt[:, 0] is x and xt[:, 1] is t
        u_t = grad(u, xt)[:, 1:2]  # ∂u/∂t
        u_x = grad(u, xt)[:, 0:1]  # ∂u/∂x
        u_xx = grad(u_x, xt)[:, 0:1]  # ∂²u/∂x²
        
        return u_t - alpha * u_xx
    
    # Boundary conditions: u(0,t) = u(1,t) = 0
    bc_left = icbc.DirichletBC(geom, lambda xt: 0.0)
    bc_right = icbc.DirichletBC(geom, lambda xt: 0.0)
    
    # Initial condition: u(x,0) = sin(π*x)
    ic = icbc.IC(spatial_geom, lambda x: np.sin(np.pi * x[:, 0]))
    
    # Generate measurement data at various time points
    n_meas = 50
    x_meas = np.random.rand(n_meas, 1)
    t_meas = np.random.rand(n_meas, 1) * 0.5
    xt_meas = np.column_stack([x_meas, t_meas])
    
    # Analytical solution: u(x,t) = exp(-α*π²*t)*sin(π*x)
    u_meas = np.exp(-alpha * np.pi**2 * t_meas) * np.sin(np.pi * x_meas)
    
    measurements = {
        'x': xt_meas,
        'u': u_meas
    }
    
    # Create data object
    data = scinn.Data(
        geometry=geom,
        pde=heat_pde,
        bcs=[bc_left, bc_right],
        ics=[ic],
        num_domain=5000,
        num_boundary=200,
        num_initial=200,
        measurement_data=measurements
    )
    
    # Create model
    model = FNN([2, 64, 64, 64, 1], activation='tanh')
    
    # Create solver
    solver = scinn.Solver(
        model=model,
        data=data,
        lr=1e-3,
        loss_weights={
            'pde': 1.0,
            'bc': 10.0,
            'ic': 10.0,
            'data': 20.0
        }
    )
    
    # Train
    solver.train(epochs=3000, print_every=300)
    
    # Evaluate
    results = solver.evaluate_at_measurements()
    
    return solver, data, results


if __name__ == "__main__":
    # Run examples
    
    # Example 1: PINN with measurements
    solver_with_data, data_with_meas, results = example_with_measurements()
    
    # Example 2: PINN without measurements (for comparison)
    solver_no_data, data_no_meas = example_without_measurements()
    
    # Example 3: Time-dependent problem
    # solver_time, data_time, results_time = time_dependent_example()
    
    # Visualize (requires matplotlib)
    try:
        visualize_results(solver_with_data, data_with_meas, results)
    except Exception as e:
        print(f"\nVisualization skipped: {e}")
    
    print("\n" + "="*80)
    print("Examples completed!")
    print("="*80)
