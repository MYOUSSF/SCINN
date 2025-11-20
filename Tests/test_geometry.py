"""
Test script for complex geometries:
    1. Disk with hole (annulus)
    2. Union of shapes
    3. Intersection of shapes
"""


import numpy as np
import torch
from scinn import geometry, nn, icbc, Data, Solver


def test_annulus():
    """Test Poisson equation on an annulus (disk with hole)."""
    print("=" * 70)
    print("Test 3a: Annulus (Disk - Disk)")
    print("=" * 70)
    
    # Create annulus: outer disk - inner disk
    outer_disk = geometry.Disk([0, 0], radius=2.0)
    inner_disk = geometry.Disk([0, 0], radius=0.5)
    annulus = outer_disk - inner_disk
    
    # Test point sampling
    print("\nTesting point sampling:")
    domain_points = annulus.random_points(100)
    boundary_points = annulus.random_boundary_points(100)
    
    print(f"  Domain points shape: {domain_points.shape}")
    print(f"  Boundary points shape: {boundary_points.shape}")
    
    # Verify points are inside
    inside_mask = annulus.inside(domain_points)
    print(f"  Points inside domain: {np.sum(inside_mask)}/{len(domain_points)}")
    
    # Verify boundary points are on boundary
    on_boundary_mask = annulus.on_boundary(boundary_points)
    print(f"  Points on boundary: {np.sum(on_boundary_mask)}/{len(boundary_points)}")
    
    # Define simple PDE: -Δu = 1
    def laplace_pde(x, u):
        return nn.laplacian(u, x) + 1.0
    
    # BCs: u = 0 on outer boundary, u = 1 on inner boundary
    # Note: This is simplified; ideally we'd separate inner/outer boundaries
    bc = icbc.DirichletBC(annulus, lambda x: 0.0)
    
    # Create data and model
    data = Data(
        geometry=annulus,
        pde=laplace_pde,
        bcs=[bc],
        num_domain=2000,
        num_boundary=200
    )
    
    model = nn.FNN([2, 40, 40, 1], activation='tanh')
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    solver = Solver(model, data, lr=1e-3, device=device)
    
    # Quick training
    solver.train(epochs=2000, print_every=500)
    
    print("Test 3a completed!\n")


def test_union():
    """Test union of two circles."""
    print("=" * 70)
    print("Test 3b: Union of Two Circles")
    print("=" * 70)
    
    # Create two overlapping disks
    disk1 = geometry.Disk([0, 0], radius=1.0)
    disk2 = geometry.Disk([1.5, 0], radius=1.0)
    union = disk1 | disk2  # Using | operator
    
    # Test point sampling
    domain_points = union.random_points(100)
    boundary_points = union.random_boundary_points(100)
    
    print(f"\nDomain points shape: {domain_points.shape}")
    print(f"Boundary points shape: {boundary_points.shape}")
    
    # Verify points
    inside_mask = union.inside(domain_points)
    on_boundary_mask = union.on_boundary(boundary_points)
    
    print(f"Points inside: {np.sum(inside_mask)}/{len(domain_points)}")
    print(f"Points on boundary: {np.sum(on_boundary_mask)}/{len(boundary_points)}")
    
    print("Test 3b completed!\n")


def test_intersection():
    """Test intersection of rectangle and disk."""
    print("=" * 70)
    print("Test 3c: Intersection of Rectangle and Disk")
    print("=" * 70)
    
    # Create shapes
    rect = geometry.Rectangle(-1, 1, -1, 1)
    disk = geometry.Disk([0, 0], radius=1.5)
    intersection = rect & disk  # Using & operator
    
    # Test point sampling
    domain_points = intersection.random_points(100)
    boundary_points = intersection.random_boundary_points(100)
    
    print(f"\nDomain points shape: {domain_points.shape}")
    print(f"Boundary points shape: {boundary_points.shape}")
    
    # Verify points
    inside_mask = intersection.inside(domain_points)
    on_boundary_mask = intersection.on_boundary(boundary_points)
    
    print(f"Points inside: {np.sum(inside_mask)}/{len(domain_points)}")
    print(f"Points on boundary: {np.sum(on_boundary_mask)}/{len(boundary_points)}")
    
    print("Test 3c completed!\n")


def test_ellipse():
    """Test ellipse geometry."""
    print("=" * 70)
    print("Test 3d: Ellipse")
    print("=" * 70)
    
    # Create ellipse
    ellipse = geometry.Ellipse(center=[0, 0], semimajor=3, semiminor=2)
    
    # Test specific points
    test_points = np.array([
        [3, 0],    # On boundary (major axis)
        [0, 2],    # On boundary (minor axis)
        [0, 0],    # Center (inside)
        [1, 1],    # Inside
        [4, 0]     # Outside
    ])
    
    print("\nPoint classification:")
    inside = ellipse.inside(test_points)
    on_boundary = ellipse.on_boundary(test_points)
    
    for i, pt in enumerate(test_points):
        status = "on boundary" if on_boundary[i] else ("inside" if inside[i] else "outside")
        print(f"  Point {pt}: {status}")
    
    # Test boundary normals
    boundary_pts = ellipse.random_boundary_points(5)
    normals = ellipse.boundary_normal(boundary_pts)
    
    print("\nBoundary normals:")
    for i in range(len(boundary_pts)):
        print(f"  Point {boundary_pts[i]}: normal = {normals[i]}")
    
    print("Test 3d completed!\n")


def main():
    test_annulus()
    test_union()
    test_intersection()
    test_ellipse()
    print("=" * 70)
    print("All geometry tests completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
