# Measurement Data Support in SCINN

This document describes how to incorporate known measurement data into your Physics-Informed Neural Networks (PINNs) using the SCINN framework.

## Overview

The measurement data feature allows you to train PINNs that simultaneously:
1. Satisfy the governing PDE
2. Match boundary/initial conditions
3. **Fit known measurement data** (new feature)

This is particularly useful when you have experimental or observational data that you want to incorporate into your physics-based model.

## Key Changes

### 1. Modified `Data` Class

The `Data` class now accepts an optional `measurement_data` parameter:

```python
from scinn import Data, geometry, icbc

# Create measurement data dictionary
measurements = {
    'x': x_coordinates,  # numpy array of shape (n_measurements, input_dim)
    'u': u_values        # numpy array of shape (n_measurements, output_dim)
}

# Create Data object with measurements
data = Data(
    geometry=geom,
    pde=pde_function,
    bcs=[bc1, bc2],
    num_domain=1000,
    num_boundary=100,
    measurement_data=measurements  # Add your measurements here
)
```

#### Input Format

- `x`: Coordinates where measurements were taken
  - For 2D spatial problems: shape `(n_measurements, 2)` with columns `[x, y]`
  - For 1D time-dependent: shape `(n_measurements, 2)` with columns `[x, t]`
  - For 2D time-dependent: shape `(n_measurements, 3)` with columns `[x, y, t]`

- `u`: Measured values at those coordinates
  - For scalar problems: shape `(n_measurements, 1)`
  - For vector problems: shape `(n_measurements, n_components)`

### 2. Modified `Solver` Class

The `Solver` class now includes a `'data'` loss component:

```python
from scinn import Solver

solver = Solver(
    model=model,
    data=data,
    lr=1e-3,
    loss_weights={
        'pde': 1.0,    # Weight for PDE residual
        'bc': 10.0,    # Weight for boundary conditions
        'ic': 10.0,    # Weight for initial conditions
        'data': 50.0   # Weight for measurement data (NEW)
    }
)
```

#### Loss Weight Tuning

The `'data'` weight controls how strongly the model fits the measurements:
- **Higher values** (e.g., 50-100): Model will fit measurements more closely
- **Lower values** (e.g., 1-10): Model will prioritize PDE satisfaction
- **Recommended starting point**: 10-50 times the PDE weight

## Usage Examples

### Example 1: Simple 2D Problem with Measurements

```python
import numpy as np
from scinn import geometry, icbc, Data, Solver
from scinn.nn import FNN, laplacian

# 1. Define domain
geom = geometry.Rectangle(0, 1, 0, 1)

# 2. Define PDE: -Δu = 1
def pde(x, u):
    return laplacian(u, x) + 1.0

# 3. Boundary conditions
bc = icbc.DirichletBC(geom, lambda x: 0.0)

# 4. Measurement data (from experiments or sensors)
x_measurements = np.array([
    [0.3, 0.3],
    [0.5, 0.5],
    [0.7, 0.7],
    # ... more measurement points
])

u_measurements = np.array([
    [0.045],
    [0.063],
    [0.045],
    # ... corresponding values
])

measurements = {
    'x': x_measurements,
    'u': u_measurements
}

# 5. Create Data object
data = Data(
    geometry=geom,
    pde=pde,
    bcs=[bc],
    num_domain=2000,
    num_boundary=200,
    measurement_data=measurements
)

# 6. Create and train model
model = FNN([2, 50, 50, 50, 1], activation='tanh')

solver = Solver(
    model=model,
    data=data,
    lr=1e-3,
    loss_weights={'pde': 1.0, 'bc': 10.0, 'data': 50.0}
)

solver.train(epochs=5000, print_every=500)
```

### Example 2: Time-Dependent Problem with Measurements

```python
import numpy as np
from scinn import geometry, icbc, Data, Solver
from scinn.nn import FNN, grad

# Spatial and temporal domains
spatial_geom = geometry.Interval(0, 1)
time_geom = geometry.TimeDomain(0, 1.0)
geom = spatial_geom * time_geom  # Space-time domain

# Heat equation: ∂u/∂t = α∂²u/∂x²
alpha = 0.1

def heat_pde(xt, u):
    u_t = grad(u, xt)[:, 1:2]    # ∂u/∂t
    u_x = grad(u, xt)[:, 0:1]    # ∂u/∂x
    u_xx = grad(u_x, xt)[:, 0:1] # ∂²u/∂x²
    return u_t - alpha * u_xx

# Boundary and initial conditions
bc = icbc.DirichletBC(geom, lambda xt: 0.0)
ic = icbc.IC(spatial_geom, lambda x: np.sin(np.pi * x[:, 0]))

# Measurement data at various (x, t) points
xt_measurements = np.array([
    [0.25, 0.1],  # x=0.25, t=0.1
    [0.50, 0.2],  # x=0.50, t=0.2
    [0.75, 0.3],  # x=0.75, t=0.3
    # ... more points
])

u_measurements = np.array([
    [0.567],
    [0.412],
    [0.289],
    # ... corresponding values
])

measurements = {
    'x': xt_measurements,
    'u': u_measurements
}

# Create data and train
data = Data(
    geometry=geom,
    pde=heat_pde,
    bcs=[bc],
    ics=[ic],
    num_domain=5000,
    num_boundary=200,
    num_initial=200,
    measurement_data=measurements
)

model = FNN([2, 64, 64, 64, 1], activation='tanh')

solver = Solver(
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

solver.train(epochs=3000, print_every=300)
```

### Example 3: Multi-Component Problem

For problems with multiple output components (e.g., velocity fields):

```python
# Measurement data for 2D velocity field
measurements = {
    'x': np.array([
        [0.5, 0.5],  # Location
        [0.3, 0.7],
        # ...
    ]),
    'u': np.array([
        [0.2, 0.3],  # [u_x, u_y] at first location
        [0.1, 0.4],  # [u_x, u_y] at second location
        # ...
    ])
}

# Model with 2 outputs
model = FNN([2, 64, 64, 64, 2], activation='tanh')
```

## Evaluating Fit to Measurements

After training, you can evaluate how well the model fits the measurement data:

```python
# Train the model
solver.train(epochs=5000)

# Evaluate at measurement locations
results = solver.evaluate_at_measurements()

print(f"MSE:       {results['mse']:.6e}")
print(f"MAE:       {results['mae']:.6e}")
print(f"Max Error: {results['max_error']:.6e}")

# Access individual predictions
x_meas = results['x']          # Measurement locations
u_true = results['u_true']     # True measured values
u_pred = results['u_pred']     # Model predictions
errors = results['error']       # u_pred - u_true
```

## Best Practices

### 1. Loss Weight Selection

Start with these guidelines and adjust based on results:
- PDE weight: 1.0 (baseline)
- BC weight: 10-100 (boundary conditions are usually important)
- IC weight: 10-100 (initial conditions are usually important)
- Data weight: 10-100 (depends on measurement confidence)

If measurements are:
- **Very accurate**: Use higher data weight (50-100)
- **Noisy**: Use moderate data weight (10-30)
- **Sparse**: Use moderate data weight and more collocation points

### 2. Data Preprocessing

```python
# Normalize measurements if needed
x_mean, x_std = x_measurements.mean(axis=0), x_measurements.std(axis=0)
u_mean, u_std = u_measurements.mean(), u_measurements.std()

x_norm = (x_measurements - x_mean) / x_std
u_norm = (u_measurements - u_mean) / u_std

# Remember to denormalize predictions!
```

### 3. Handling Noisy Data

If measurements contain noise:
- Use moderate data weights
- Consider adding more collocation points
- Monitor the PDE residual to ensure physics is still satisfied

### 4. Training Monitoring

Watch the individual loss components:
```python
history = solver.get_history()

import matplotlib.pyplot as plt
plt.semilogy(history['loss_pde'], label='PDE')
plt.semilogy(history['loss_bc'], label='BC')
plt.semilogy(history['loss_data'], label='Data')
plt.legend()
plt.show()
```

All loss components should decrease. If one dominates, adjust the weights.

## API Reference

### Data Class

```python
Data(
    geometry,
    pde,
    bcs=None,
    ics=None,
    num_domain=1000,
    num_boundary=100,
    num_initial=100,
    num_test=1000,
    measurement_data=None,  # NEW
    sampler="pseudo"
)
```

**Parameters:**
- `measurement_data` (dict, optional): Dictionary with keys:
  - `'x'`: numpy array of shape `(n_measurements, input_dim)`
  - `'u'`: numpy array of shape `(n_measurements, output_dim)`

**Methods:**
- `has_measurements()`: Returns True if measurement data is available
- `get_measurement_data()`: Returns the measurement data dictionary

### Solver Class

```python
Solver(
    model,
    data,
    optimizer=None,
    lr=1e-3,
    loss_weights=None,  # Can include 'data' key
    device='cpu'
)
```

**Parameters:**
- `loss_weights` (dict, optional): Dictionary with keys `'pde'`, `'bc'`, `'ic'`, `'data'`

**Methods:**
- `compute_data_loss()`: Compute MSE between predictions and measurements
- `evaluate_at_measurements()`: Evaluate model at measurement locations

## Troubleshooting

### Issue: Data loss not decreasing
- **Solution**: Increase the data weight or check data format

### Issue: PDE residual not decreasing
- **Solution**: Decrease data weight or increase PDE weight

### Issue: Model overfits to measurements
- **Solution**: Decrease data weight, increase number of collocation points

### Issue: Predictions don't match measurements
- **Solution**: 
  - Increase data weight
  - Check that measurement points are inside the domain
  - Verify measurement data format (correct shape and dtype)

## Complete Working Example

See `example_measurement_data.py` for complete runnable examples.

## Notes

- Measurement data points don't need to be on a regular grid
- You can have measurements at different time steps for time-dependent problems
- The framework automatically validates that measurement points are within the domain (warns if not)
- Measurement data is NOT resampled during training (unlike collocation points)
