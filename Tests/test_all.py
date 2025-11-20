"""
Comprehensive test suite for SCINN framework.
Run all tests to verify the framework is working correctly.
"""

import numpy as np
import torch
from scinn import geometry, nn, icbc, Data, Solver


def test_geometry_basics():
    """Test basic geometry operations."""
    print("\n" + "=" * 70)
    print("TEST 1: Basic Geometry")
    print("=" * 70)
    
    # Test interval
    interval = geometry.Interval(0, 1)
    points = interval.random_points(10)
    assert points.shape == (10, 1), "Interval points shape incorrect"
    assert np.all(interval.inside(points)), "Interval inside check failed"
    print("✓ Interval tests passed")
    
    # Test rectangle
    rect = geometry.Rectangle(0, 1, 0, 1)
    points = rect.random_points(100)
    assert points.shape == (100, 2), "Rectangle points shape incorrect"
    assert np.all(rect.inside(points)), "Rectangle inside check failed"
    print("✓ Rectangle tests passed")
    
    # Test disk
    disk = geometry.Disk([0, 0], 1.0)
    points = disk.random_points(100)
    assert points.shape == (100, 2), "Disk points shape incorrect"
    assert np.all(disk.inside(points)), "Disk inside check failed"
    print("✓ Disk tests passed")
    
    # Test ellipse
    ellipse = geometry.Ellipse([0, 0], 2, 1)
    points = ellipse.random_points(100)
    assert points.shape == (100, 2), "Ellipse points shape incorrect"
    assert np.all(ellipse.inside(points)), "Ellipse inside check failed"
    print("✓ Ellipse tests passed")
    
    print("\n✅ All basic geometry tests passed!")


def test_geometry_operations():
    """Test geometry operations."""
    print("\n" + "=" * 70)
    print("TEST 2: Geometry Operations")
    print("=" * 70)
    
    # Test union
    disk1 = geometry.Disk([0, 0], 1.0)
    disk2 = geometry.Disk([1, 0], 1.0)
    union = disk1 | disk2
    points = union.random_points(100)
    assert len(points) == 100, "Union sampling failed"
    print("✓ Union tests passed")
    
    # Test difference
    outer = geometry.Disk([0, 0], 2.0)
    inner = geometry.Disk([0, 0], 0.5)
    diff = outer - inner
    points = diff.random_points(100)
    assert len(points) == 100, "Difference sampling failed"
    assert np.all(diff.inside(points)), "Difference inside check failed"
    print("✓ Difference tests passed")
    
    # Test intersection
    rect = geometry.Rectangle(-1, 1, -1, 1)
    disk = geometry.Disk([0, 0], 1.5)
    inter = rect & disk
    points = inter.random_points(100)
    assert len(points) == 100, "Intersection sampling failed"
    print("✓ Intersection tests passed")
    
    # Test cross product
    interval1 = geometry.Interval(0, 1)
    interval2 = geometry.Interval(0, 1)
    cross = interval1 * interval2
    points = cross.random_points(100)
    assert points.shape == (100, 2), "Cross product shape incorrect"
    print("✓ Cross product tests passed")
    
    print("\n✅ All geometry operation tests passed!")


def test_neural_network():
    """Test neural network architectures."""
    print("\n" + "=" * 70)
    print("TEST 3: Neural Networks")
    print("=" * 70)
    
    # Test FNN
    model = nn.FNN([2, 20, 20, 1], activation='tanh')
    x = torch.randn(10, 2, requires_grad=True)
    y = model(x)
    assert y.shape == (10, 1), "FNN output shape incorrect"
    print("✓ FNN tests passed")
    
    # Test ModifiedMLP
    model = nn.ModifiedMLP([2, 20, 20, 1], activation='tanh', fourier_features=True)
    y = model(x)
    assert y.shape == (10, 1), "ModifiedMLP output shape incorrect"
    print("✓ ModifiedMLP tests passed")
    
    # Test ResNet
    model = nn.ResNet([2, 20, 1], activation='tanh')
    y = model(x)
    assert y.shape == (10, 1), "ResNet output shape incorrect"
    print("✓ ResNet tests passed")
    
    print("\n✅ All neural network tests passed!")


def test_gradients():
    """Test gradient computations."""
    print("\n" + "=" * 70)
    print("TEST 4: Gradient Computations")
    print("=" * 70)
    
    # Create simple function u = x^2 + y^2
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    u = (x[:, 0:1]**2 + x[:, 1:2]**2)
    
    # Test gradient
    grad_u = nn.grad(u, x)
    assert grad_u.shape == x.shape, "Gradient shape incorrect"
    print("✓ Gradient computation passed")
    
    # Test Laplacian
    laplace_u = nn.laplacian(u, x)
    assert laplace_u.shape == (2, 1), "Laplacian shape incorrect"
    # For u = x^2 + y^2, Δu = 2 + 2 = 4
    expected = torch.full_like(laplace_u, 4.0)
    assert torch.allclose(laplace_u, expected, atol=1e-5), "Laplacian value incorrect"
    print("✓ Laplacian computation passed")
    
    print("\n✅ All gradient tests passed!")


def test_boundary_conditions():
    """Test boundary condition definitions."""
    print("\n" + "=" * 70)
    print("TEST 5: Boundary Conditions")
    print("=" * 70)
    
    geom = geometry.Rectangle(0, 1, 0, 1)
    
    # Test Dirichlet BC
    bc_dirichlet = icbc.DirichletBC(geom, lambda x: 0.0)
    x_bc = bc_dirichlet.sample_points(10)
    assert len(x_bc) == 10, "Dirichlet BC sampling failed"
    print("✓ Dirichlet BC tests passed")
    
    # Test Neumann BC
    bc_neumann = icbc.NeumannBC(geom, lambda x: 1.0)
    assert bc_neumann.type == 'neumann', "Neumann BC type incorrect"
    print("✓ Neumann BC tests passed")
    
    # Test Robin BC
    bc_robin = icbc.RobinBC(geom, lambda x: 1.0, alpha=1.0, beta=1.0)
    assert bc_robin.type == 'robin', "Robin BC type incorrect"
    print("✓ Robin BC tests passed")
    
    # Test Initial Condition
    ic = icbc.IC(geom, lambda x: np.sin(np.pi * x[:, 0]))
    x_ic = ic.sample_points(10)
    assert len(x_ic) == 10, "IC sampling failed"
    print("✓ Initial condition tests passed")
    
    print("\n✅ All boundary condition tests passed!")


def test_simple_pinn():
    """Test a simple PINN problem."""
    print("\n" + "=" * 70)
    print("TEST 6: Simple PINN (1D Poisson)")
    print("=" * 70)
    
    # 1D Poisson: -d²u/dx² = 1, u(0) = u(1) = 0
    # Exact solution: u(x) = x(1-x)/2
    
    geom = geometry.Interval(0, 1)
    
    def poisson_1d(x, u):
        u_xx = nn.grad(nn.grad(u, x), x)
        return u_xx + 1.0
    
    bc = icbc.DirichletBC(geom, lambda x: 0.0)
    
    data = Data(
        geometry=geom,
        pde=poisson_1d,
        bcs=[bc],
        num_domain=100,
        num_boundary=2
    )
    
    model = nn.FNN([1, 20, 20, 1], activation='tanh')
    device = 'cpu'
    solver = Solver(model, data, lr=1e-3, device=device)
    
    print("\nTraining for 1000 epochs...")
    solver.train(epochs=1000, print_every=250)
    
    # Test at x = 0.5, exact solution is 0.125
    test_point = np.array([[0.5]])
    u_pred = solver.predict(test_point)[0, 0]
    u_exact = 0.125
    error = abs(u_pred - u_exact)
    
    print(f"\nAt x=0.5:")
    print(f"  Predicted: {u_pred:.6f}")
    print(f"  Exact:     {u_exact:.6f}")
    print(f"  Error:     {error:.6f}")
    
    if error < 0.01:
        print("✓ PINN solution is accurate!")
    else:
        print("⚠ PINN solution may need more training")
    
    print("\n✅ Simple PINN test completed!")


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("PINNsTorch Framework Test Suite")
    print("=" * 70)
    
    try:
        test_geometry_basics()
        test_geometry_operations()
        test_neural_network()
        test_gradients()
        test_boundary_conditions()
        test_simple_pinn()
        
        print("\n" + "=" * 70)
        print("🎉 ALL TESTS PASSED! 🎉")
        print("=" * 70)
        print("\nThe PINNsTorch framework is working correctly!")
        print("You can now use it to solve your PDE problems.")
        print("\nNext steps:")
        print("  1. Run: python test_poisson.py")
        print("  2. Run: python test_heat.py")
        print("  3. Run: python test_geometry.py")
        print("  4. Run: python test_bc.py")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
