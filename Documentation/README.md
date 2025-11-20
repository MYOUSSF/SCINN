# SCINN

A lightweight, clean, and intuitive Physics-Informed Neural Networks (PINNs) framework built with PyTorch.

## Features

- **Clean API**: Simple and intuitive interface for defining PDEs, geometries, and boundary conditions
- **Flexible Geometries**: Built-in shapes (Interval, Rectangle, Disk, Ellipse) with operations (Union, Intersection, Difference, Cross Product)
- **Multiple BC Types**: Dirichlet, Neumann, Robin, and Initial Conditions
- **PyTorch-based**: Leverages PyTorch's automatic differentiation
- **Modular Design**: Easy to extend with custom components

## Installation

```bash
# The framework is ready to use - just ensure you have the dependencies
pip install torch numpy
pip install scikit-optimize  # Optional, for advanced sampling
```

## Quick Start

### Example 1: 2D Poisson Equation

```python
import torch
from scinn import geometry, nn, icbc, Data, Solver

# Define domain
geom = geometry.Rectangle(0, 1, 0, 1)

# Define PDE: -Δu = 1
def poisson_pde(x, u):
    return nn.laplacian(u, x) + 1.0

# Define boundary condition: u = 0 on boundary
bc = icbc.DirichletBC(geom, lambda x: 0.0)

# Create data object
data = Data(
    geometry=geom,
    pde=poisson_pde,
    bcs=[bc],
    num_domain=2000,
    num_boundary=200
)

# Create neural network
model = nn.FNN([2, 50, 50, 1], activation='tanh')

# Create solver and train
solver = Solver(model, data, lr=1e-3)
solver.train(epochs=5000)

# Make predictions
test_points = geom.random_points(100)
predictions = solver.predict(test_points)
```

### Example 2: Heat Equation

```python
# Space-time domain
x_domain = geometry.Interval(0, 1)
t_domain = geometry.TimeDomain(0, 1)
geom = x_domain * t_domain  # Cross product

# Heat equation: du/dt = α*d²u/dx²
def heat_pde(xt, u):
    du_dt = nn.grad(u, xt)[:, 1:2]
    u_x = nn.grad(u, xt)[:, 0:1]
    u_xx = nn.grad(u_x, xt)[:, 0:1]
    return du_dt - 0.1 * u_xx

# Initial condition: u(x,0) = sin(πx)
ic = icbc.IC(x_domain, lambda x: np.sin(np.pi * x[:, 0]))

# Boundary conditions: u(0,t) = u(1,t) = 0
bc = icbc.DirichletBC(geom, lambda x: 0.0)

data = Data(geom, heat_pde, bcs=[bc], ics=[ic])
model = nn.FNN([2, 50, 50, 1], activation='tanh')
solver = Solver(model, data, lr=1e-3)
solver.train(epochs=5000)
```

## Geometry Operations

Create complex geometries using simple operations:

```python
# Union (|)
union = disk1 | disk2

# Difference (-)
annulus = outer_disk - inner_disk

# Intersection (&)
shape = rectangle & disk

# Cross Product (*)
spacetime = space * time
```

## Boundary Conditions

### Dirichlet BC
```python
# u = 0 on boundary
bc = icbc.DirichletBC(geometry, lambda x: 0.0)

# u = sin(x) on boundary
bc = icbc.DirichletBC(geometry, lambda x: np.sin(x[:, 0]))
```

### Neumann BC
```python
# du/dn = 0 on boundary (no flux)
bc = icbc.NeumannBC(geometry, lambda x: 0.0)
```

### Robin BC
```python
# u + du/dn = 1 on boundary
bc = icbc.RobinBC(geometry, lambda x: 1.0, alpha=1.0, beta=1.0)
```

### Initial Condition
```python
# u(x, t=0) = f(x)
ic = icbc.IC(geometry, lambda x: np.sin(np.pi * x[:, 0]))
```

## Neural Network Architectures

```python
# Standard feedforward network
model = nn.FNN([2, 50, 50, 1], activation='tanh')

# Modified MLP with Fourier features
model = nn.ModifiedMLP([2, 50, 1], fourier_features=True)

# Residual network
model = nn.ResNet([2, 50, 1], num_res_blocks=4)
```

## Gradient Utilities

```python
# First-order gradient
du_dx = nn.grad(u, x)

# Laplacian
laplace_u = nn.laplacian(u, x)

# Jacobian
J = nn.jacobian(u, x)

# Hessian
H = nn.hessian(u, x)

# Divergence
div = nn.divergence(u, x)
```

## Training Options

```python
solver = Solver(
    model=model,
    data=data,
    lr=1e-3,
    loss_weights={'pde': 1.0, 'bc': 100.0, 'ic': 50.0},
    device='cuda'  # or 'cpu'
)

# Train with resampling
solver.train(
    epochs=10000,
    print_every=500,
    resample_every=1000  # Resample collocation points
)

# Save/load model
solver.save('model.pt')
solver.load('model.pt')
```

## Available Geometries

- **1D**: `Interval`, `Circle` (1D manifold), `TimeDomain`
- **2D**: `Rectangle`, `Disk`, `Ellipse`
- **Operations**: `Union` (|), `Intersection` (&), `Difference` (-), `CrossProduct` (*)

## Project Structure

```
scinn/
├── geometry/
│   ├── geometry.py      # Base shapes
│   ├── ops.py          # Geometry operations
│   └── sampler.py      # Point sampling methods
├── nn/
│   ├── fnn.py          # Neural network architectures
│   └── gradients.py    # Automatic differentiation utilities
├── icbc/
│   └── conditions.py   # Boundary and initial conditions
├── data.py             # Data container
└── solver.py           # Training solver
```

## Testing

Run the comprehensive test suite:

```bash
python test_all.py          # Run all unit tests
python test_poisson.py      # Test 2D Poisson equation
python test_heat.py         # Test 1D heat equation
python test_geometry.py     # Test complex geometries
python test_bc.py          # Test boundary conditions
```

## Comparison with DeepXDE

### Improvements:
- **Cleaner API**: More intuitive syntax without `on_boundary` and `isclose()` checks
- **Better geometry handling**: Simplified operations with operator overloading
- **PyTorch-only**: No TensorFlow dependency
- **Modular**: Easy to extend and customize
- **Modern**: Uses Python 3.6+ features

### Example Comparison:

**DeepXDE:**
```python
def boundary(x, on_boundary):
    return on_boundary and np.isclose(x[0], 0)
```

**SCINN:**
```python
bc = DirichletBC(geometry, lambda x: 0.0)
```

## License

MIT License

## References

- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal of Computational Physics.
