# SCINN - Scientific Computing with Informed Neural Networks

A lightweight, intuitive Physics-Informed Neural Networks (PINNs) framework built with PyTorch.

## 🚀 Quick Start

```bash
# Navigate to Installation folder
cd Installation

# Install the package
pip install -e ..

# Or install dependencies manually
pip install torch numpy
```

## 📁 Project Structure

```
scinn/
├── Documentation/          📚 Complete documentation
│   ├── README.md          → Full API reference
│   ├── QUICKSTART.md      → Get started in 5 minutes
│   ├── ARCHITECTURE.md    → System design
│   └── PROJECT_SUMMARY.md → Overview & comparison
│
├── Installation/          🔧 Setup files
│   ├── setup.py          → Package installer
│   └── requirements.txt  → Dependencies
│
├── scinn/                 📦 Main library
│   ├── __init__.py
│   ├── data.py           → Problem data container
│   ├── solver.py         → Training solver
│   ├── geometry/         → Geometric domains
│   ├── nn/               → Neural networks
│   └── icbc/             → Boundary conditions
│
├── Tests/                 ✅ Test suite
│   ├── test_all.py       → Comprehensive tests
│   ├── test_poisson.py   → 2D Poisson
│   ├── test_heat.py      → Heat equation
│   ├── test_geometry.py  → Complex geometries
│   └── test_bc.py        → Boundary conditions
│
└── Examples/              📖 Working examples
    └── examples.py       → 7 example problems
```

## 🧪 Testing

```bash
# Run from Tests directory
cd Tests
python test_all.py        # Full test suite
python test_heat.py       # Heat equation example
```

## 📖 Documentation

Start with these files in the `Documentation/` folder:

1. **QUICKSTART.md** - Get running in 5 minutes
2. **README.md** - Complete API documentation
3. **ARCHITECTURE.md** - System design and diagrams
4. **PROJECT_SUMMARY.md** - Overview and comparison with DeepXDE

## 💡 Simple Example

```python
from scinn import geometry, nn, icbc, Data, Solver

# Define geometry and PDE
geom = geometry.Rectangle(0, 1, 0, 1)
def pde(x, u): return nn.laplacian(u, x) + 1.0

# Set boundary condition
bc = icbc.DirichletBC(geom, lambda x: 0.0)

# Create data and model
data = Data(geom, pde, bcs=[bc])
model = nn.FNN([2, 50, 50, 1])

# Train
solver = Solver(model, data)
solver.train(epochs=5000)
```

## ✨ Key Features

- **Clean API** - No complex callbacks
- **Operator Overloading** - `geom1 | geom2`, `geom1 - geom2`
- **Easy Gradients** - `nn.grad()`, `nn.laplacian()`, `nn.hessian()`
- **Multiple BCs** - Dirichlet, Neumann, Robin, Initial conditions
- **PyTorch Only** - No TensorFlow dependency

## 📦 Installation Options

### Option 1: Editable Install (Recommended)
```bash
cd Installation
pip install -e ..
```

### Option 2: Direct Install
```bash
cd Installation
python setup.py install
```

### Option 3: Just Dependencies
```bash
pip install -r Installation/requirements.txt
# Then add project root to PYTHONPATH
```

## 🎯 Example Problems

The `Examples/` folder contains:
1. 1D Poisson equation
2. 2D Poisson on rectangle
3. 2D Laplace on disk
4. Poisson on annulus
5. Heat equation (time-dependent)
6. Neumann BCs
7. Robin BCs

## 📊 Requirements

- Python 3.7+
- PyTorch 2.0+
- NumPy 1.20+
- Optional: scikit-optimize (for advanced sampling)

## 🛠️ Development

```bash
# Run tests
cd Tests
python test_all.py

# Run examples
cd Examples
python examples.py
```

## 📚 Learn More

Visit the `Documentation/` folder for:
- Detailed tutorials
- API reference
- Architecture diagrams
- Comparison with other frameworks

## 🤝 Contributing

This is a clean, modular framework designed for:
- Research in scientific computing
- Educational purposes
- Rapid prototyping of PDE solvers

Feel free to extend with:
- New geometry types
- Additional neural architectures
- Custom boundary conditions

## 📄 License

MIT License

---

**Getting Help?**
- Check `Documentation/QUICKSTART.md` for setup issues
- See `Documentation/README.md` for API details
- Review `Examples/examples.py` for code patterns
