# SCINN Quick Start Guide

## Installation

1. **Install PyTorch** (if not already installed):
```bash
pip install torch
```

2. **Install SCINN**:
```bash
# From the directory containing setup.py
pip install -e .

# Or just add the directory to your Python path
export PYTHONPATH="/path/to/scinn:$PYTHONPATH"
```

3. **Optional dependencies**:
```bash
pip install scikit-optimize  # For advanced sampling
pip install matplotlib       # For visualization
```

## Testing

Run the test suite to verify everything works:

```bash
# Comprehensive test suite
python test_all.py

# Individual tests
python test_poisson.py    # 2D Poisson equation
python test_heat.py       # 1D heat equation
python test_geometry.py   # Complex geometries
python test_bc.py        # Boundary conditions

# Examples
python examples.py        # Run example problems
```

## First Example

```python
import torch
from scinn import geometry, nn, icbc, Data, Solver

# 1. Define the domain
geom = geometry.Rectangle(0, 1, 0, 1)

# 2. Define the PDE: -Δu = 1
def poisson_pde(x, u):
    return nn.laplacian(u, x) + 1.0

# 3. Define boundary condition: u = 0
bc = icbc.DirichletBC(geom, lambda x: 0.0)

# 4. Create data object
data = Data(
    geometry=geom,
    pde=poisson_pde,
    bcs=[bc],
    num_domain=2000,
    num_boundary=200
)

# 5. Create neural network
model = nn.FNN([2, 50, 50, 1], activation='tanh')

# 6. Create solver and train
solver = Solver(model, data, lr=1e-3)
solver.train(epochs=5000, print_every=500)

# 7. Make predictions
test_points = geom.random_points(100)
predictions = solver.predict(test_points)
```

## Key Concepts

### 1. Geometries
```python
# Basic shapes
interval = geometry.Interval(0, 1)
rectangle = geometry.Rectangle(0, 1, 0, 1)
disk = geometry.Disk([0, 0], radius=1.0)
ellipse = geometry.Ellipse([0, 0], semimajor=2, semiminor=1)

# Operations
union = disk1 | disk2           # Union
diff = disk1 - disk2            # Difference (annulus)
inter = rect & disk             # Intersection
spacetime = space * time        # Cross product
```

### 2. PDEs
Define PDEs as functions that return the residual:

```python
# Poisson: -Δu = f
def poisson(x, u):
    return nn.laplacian(u, x) + f(x)

# Heat: du/dt - α*d²u/dx² = 0
def heat(xt, u):
    du_dt = nn.grad(u, xt)[:, 1:2]
    u_x = nn.grad(u, xt)[:, 0:1]
    u_xx = nn.grad(u_x, xt)[:, 0:1]
    return du_dt - alpha * u_xx

# Wave: d²u/dt² - c²*d²u/dx² = 0
def wave(xt, u):
    u_t = nn.grad(u, xt)[:, 1:2]
    u_tt = nn.grad(u_t, xt)[:, 1:2]
    u_x = nn.grad(u, xt)[:, 0:1]
    u_xx = nn.grad(u_x, xt)[:, 0:1]
    return u_tt - c**2 * u_xx
```

### 3. Boundary Conditions
```python
# Dirichlet: u = g on boundary
bc = icbc.DirichletBC(geom, lambda x: g(x))

# Neumann: du/dn = g on boundary
bc = icbc.NeumannBC(geom, lambda x: g(x))

# Robin: α*u + β*du/dn = g on boundary
bc = icbc.RobinBC(geom, lambda x: g(x), alpha=1, beta=1)

# Initial condition: u(x, t=0) = g(x)
ic = icbc.IC(geom, lambda x: g(x))
```

### 4. Training
```python
solver = Solver(
    model=model,
    data=data,
    lr=1e-3,                                    # Learning rate
    loss_weights={'pde': 1.0, 'bc': 100.0},   # Loss weights
    device='cuda'                               # Use GPU if available
)

solver.train(
    epochs=10000,
    print_every=500,
    resample_every=1000  # Resample points during training
)
```

## Tips for Good Results

1. **Loss Weights**: BC losses are typically weighted higher (10-100x) than PDE loss
2. **Network Architecture**: Start with 3-4 hidden layers of 50 neurons each
3. **Activation**: `tanh` works well for most problems
4. **Collocation Points**: Use 1000-5000 domain points, 100-500 boundary points
5. **Learning Rate**: Start with 1e-3, reduce if training is unstable
6. **Resampling**: Resample points every 1000-2000 epochs for better coverage

## Troubleshooting

**Poor convergence?**
- Increase boundary condition loss weight
- Add more collocation points
- Try different activation functions
- Use learning rate scheduling

**Numerical instability?**
- Reduce learning rate
- Use gradient clipping
- Normalize input/output data

**Slow training?**
- Use GPU: `device='cuda'`
- Reduce number of collocation points
- Use simpler network architecture

## Next Steps

1. Read the full README.md
2. Run all test scripts
3. Explore examples.py
4. Try your own PDE problems!

## Getting Help

- Check the README.md for detailed documentation
- Look at test scripts for working examples
- Examine examples.py for more complex use cases
