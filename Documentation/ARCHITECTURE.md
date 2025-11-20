# Scinn Architecture

## Component Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         SCINN                               │
│                    Physics-Informed Neural Networks              │
└─────────────────────────────────────────────────────────────────┘
                                │
                ┌───────────────┼───────────────┐
                │               │               │
                ▼               ▼               ▼
        ┌──────────────┐ ┌──────────┐  ┌──────────────┐
        │  Geometry    │ │   ICBC   │  │   Neural     │
        │   Module     │ │  Module  │  │   Networks   │
        └──────────────┘ └──────────┘  └──────────────┘
                │               │               │
                ▼               ▼               ▼
        ┌──────────────┐ ┌──────────┐  ┌──────────────┐
        │ • Interval   │ │•Dirichlet│  │• FNN         │
        │ • Rectangle  │ │• Neumann │  │• ModifiedMLP │
        │ • Disk       │ │• Robin   │  │• ResNet      │
        │ • Ellipse    │ │• IC      │  │• Gradients   │
        └──────────────┘ └──────────┘  └──────────────┘
                │               │               │
                └───────────────┼───────────────┘
                                ▼
                        ┌───────────────┐
                        │  Data Object  │
                        │   (Problem    │
                        │   Container)  │
                        └───────────────┘
                                │
                                ▼
                        ┌───────────────┐
                        │    Solver     │
                        │   (Training)  │
                        └───────────────┘
                                │
                                ▼
                        ┌───────────────┐
                        │  Predictions  │
                        └───────────────┘
```

## Workflow

### 1. Problem Definition
```
User defines:
  ├── Domain geometry
  ├── PDE equations
  ├── Boundary conditions
  └── Initial conditions (if time-dependent)
```

### 2. Data Setup
```
Data object combines:
  ├── Geometry → samples collocation points
  ├── PDEs → defines residual computation
  ├── BCs → defines boundary constraints
  └── ICs → defines initial constraints
```

### 3. Model Creation
```
Neural Network:
  ├── Input: spatial (and time) coordinates
  ├── Hidden layers: nonlinear transformations
  └── Output: solution values
```

### 4. Training Loop
```
For each epoch:
  ├── Sample points from domain
  ├── Sample points from boundaries
  ├── Forward pass through network
  ├── Compute PDE residuals using autodiff
  ├── Compute BC errors
  ├── Compute IC errors (if applicable)
  ├── Combine losses with weights
  └── Backpropagate and update weights
```

### 5. Prediction
```
Trained model can:
  ├── Evaluate solution at any point
  ├── Compute derivatives
  └── Generate solution fields
```

## Module Interactions

### Geometry Module
```python
geometry.Interval(a, b)
    ↓
    • inside(x): checks if points are in domain
    • on_boundary(x): checks if points are on boundary
    • random_points(n): samples interior points
    • random_boundary_points(n): samples boundary points
    • boundary_normal(x): computes outward normals
```

### Operations
```python
geom1 | geom2   → Union
geom1 & geom2   → Intersection
geom1 - geom2   → Difference
geom1 * geom2   → CrossProduct
```

### ICBC Module
```python
DirichletBC(geom, func)
    ↓
    • sample_points(n): gets boundary points
    • error(x, u): computes u - func(x)

NeumannBC(geom, func)
    ↓
    • sample_points(n): gets boundary points
    • error(x, u, grad_u): computes du/dn - func(x)

IC(geom, func)
    ↓
    • sample_points(n): gets initial points
    • error(x, u): computes u - func(x)
```

### Neural Network Module
```python
FNN([dim_in, h1, h2, ..., dim_out])
    ↓
    • forward(x): computes u(x)
    • Parameters are trainable

Gradients:
    grad(u, x)      → ∂u/∂x
    laplacian(u, x) → Δu
    hessian(u, x)   → ∂²u/∂x²
```

### Data Module
```python
Data(geometry, pde, bcs, ics)
    ↓
    • Manages all problem data
    • Samples collocation points
    • Provides train/test splits
    • Converts numpy ↔ torch
```

### Solver Module
```python
Solver(model, data)
    ↓
    • compute_pde_loss(): evaluates PDE residuals
    • compute_bc_loss(): evaluates BC errors
    • compute_ic_loss(): evaluates IC errors
    • train(): main training loop
    • predict(): makes predictions
    • save()/load(): model persistence
```

## Data Flow Example: Poisson Equation

```
1. Define geometry:
   geom = Rectangle(0, 1, 0, 1)
        ↓
   geom.random_points(1000) → x_domain ∈ ℝ^(1000×2)
   geom.random_boundary_points(100) → x_bc ∈ ℝ^(100×2)

2. Define PDE:
   def pde(x, u):
       return laplacian(u, x) + 1.0
        ↓
   For x_domain:
   u = model(x_domain) → u ∈ ℝ^(1000×1)
   residual = pde(x_domain, u) → r ∈ ℝ^(1000×1)
   loss_pde = mean(r²)

3. Define BC:
   bc = DirichletBC(geom, lambda x: 0)
        ↓
   For x_bc:
   u_bc = model(x_bc) → u_bc ∈ ℝ^(100×1)
   error_bc = u_bc - 0
   loss_bc = mean(error_bc²)

4. Total loss:
   loss = w_pde * loss_pde + w_bc * loss_bc
        ↓
   loss.backward() → compute gradients
   optimizer.step() → update weights
```

## File Organization

```
scinn/
│
├── __init__.py              # Package entry point
│   └── Exports: geometry, nn, icbc, Data, Solver
│
├── geometry/
│   ├── __init__.py          # Geometry module entry
│   ├── geometry.py          # Base shapes and Geometry class
│   ├── ops.py              # Union, Intersection, Difference, CrossProduct
│   └── sampler.py          # Point sampling strategies
│
├── nn/
│   ├── __init__.py          # Neural network module entry
│   ├── fnn.py              # FNN, ModifiedMLP, ResNet architectures
│   └── gradients.py        # grad, laplacian, hessian, etc.
│
├── icbc/
│   ├── __init__.py          # ICBC module entry
│   └── conditions.py       # BC and IC classes
│
├── data.py                  # Data container class
└── solver.py                # Solver class for training

tests/
├── test_all.py             # Comprehensive test suite
├── test_poisson.py         # Poisson equation example
├── test_heat.py           # Heat equation example
├── test_geometry.py       # Geometry tests
└── test_bc.py             # BC tests

docs/
├── README.md              # Full documentation
├── QUICKSTART.md          # Quick start guide
└── PROJECT_SUMMARY.md     # This summary
```

## Key Design Decisions

### 1. Geometry Operations via Operators
Instead of: `geom1.union(geom2)`
Use: `geom1 | geom2`

**Rationale**: More intuitive, follows set theory notation

### 2. Lambdas for BCs
Instead of: Complex callback with `on_boundary` parameter
Use: Simple `lambda x: value`

**Rationale**: Cleaner, no boolean logic needed

### 3. Automatic Differentiation
Instead of: Manual derivative implementation
Use: PyTorch autograd

**Rationale**: Flexible, accurate, automatic higher-order derivatives

### 4. Data Container
Instead of: Separate objects for geometry, PDE, BCs
Use: Single Data object

**Rationale**: Cleaner organization, easier to pass around

### 5. Loss Weights Dictionary
Instead of: Multiple parameters
Use: `{'pde': 1.0, 'bc': 100.0, 'ic': 50.0}`

**Rationale**: Clear, extensible, easy to modify

## Extensibility

### Adding New Geometry
```python
class MyGeometry(Geometry):
    def __init__(self, ...):
        super().__init__(dim=...)
    
    def inside(self, x): ...
    def on_boundary(self, x): ...
    def random_points(self, n): ...
    def random_boundary_points(self, n): ...
```

### Adding New BC Type
```python
class MyBC(BC):
    def __init__(self, geometry, func):
        super().__init__(geometry, func)
        self.type = "custom"
    
    def error(self, x, u, ...):
        # Compute BC error
        return error
```

### Adding New Network
```python
class MyNetwork(nn.Module):
    def __init__(self, ...):
        super().__init__()
        # Define layers
    
    def forward(self, x):
        # Forward pass
        return output
```

## Performance Characteristics

- **Memory**: O(n_points * n_params)
- **Computation**: O(n_points * n_layers) per epoch
- **Gradient**: Automatic via PyTorch (backpropagation)
- **Scalability**: Linear in number of points
- **GPU**: Full support via PyTorch

## Comparison Summary

| Feature | DeepXDE | Scinn |
|---------|---------|------------|
| Backend | TF/PyTorch | PyTorch only |
| BC Syntax | Complex callbacks | Simple lambdas |
| Geometry Ops | Methods | Operators |
| Gradient API | Custom | Native PyTorch |
| Code Style | Functional | Object-oriented |
| Python Version | 3.6+ | 3.7+ |
| Dependencies | Many | Minimal |
