"""Solver for training Physics-Informed Neural Networks."""

__all__ = ["Solver"]

import torch
import torch.nn as nn
import numpy as np
from .nn.gradients import grad


class Solver:
    """PINN Solver for training neural networks with physics constraints.
    
    Args:
        model: Neural network model
        data: Data object containing geometry, PDEs, boundary conditions, and measurements
        optimizer: PyTorch optimizer (default: Adam)
        lr: Learning rate
        loss_weights: Dictionary of loss weights {'pde': w1, 'bc': w2, 'ic': w3, 'data': w4}
        device: Device to run on ('cpu' or 'cuda')
        
    Example:
        model = FNN([2, 50, 50, 1])
        solver = Solver(model, data, lr=1e-3, 
                       loss_weights={'pde': 1.0, 'bc': 1.0, 'ic': 1.0, 'data': 10.0})
        solver.train(epochs=10000)
    """
    
    def __init__(self, 
                 model, 
                 data,
                 optimizer=None,
                 lr=1e-3,
                 loss_weights=None,
                 device='cpu'):
        
        self.model = model.to(device)
        self.data = data
        self.device = device
        
        # Setup optimizer
        if optimizer is None:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        else:
            self.optimizer = optimizer
        
        # Loss weights
        if loss_weights is None:
            self.loss_weights = {'pde': 1.0, 'bc': 1.0, 'ic': 1.0, 'data': 1.0}
        else:
            # Ensure all keys exist with default values
            default_weights = {'pde': 1.0, 'bc': 1.0, 'ic': 1.0, 'data': 1.0}
            default_weights.update(loss_weights)
            self.loss_weights = default_weights
        
        # Training history
        self.history = {
            'loss': [],
            'loss_pde': [],
            'loss_bc': [],
            'loss_ic': [],
            'loss_data': []
        }
    
    def compute_pde_loss(self, x):
        """Compute PDE residual loss at collocation points.
        
        Args:
            x: Domain points (numpy array or tensor)
            
        Returns:
            PDE loss (scalar tensor)
        """
        x_tensor = self.data.to_tensor(x, device=self.device, requires_grad=True)
        u = self.model(x_tensor)
        
        # Compute PDE residuals
        total_loss = 0.0
        for pde_fn in self.data.pde:
            residual = pde_fn(x_tensor, u)
            total_loss += torch.mean(residual ** 2)
        
        return total_loss
    
    def compute_bc_loss(self):
        """Compute boundary condition loss.
        
        Returns:
            BC loss (scalar tensor)
        """
        if len(self.data.bcs) == 0:
            return torch.tensor(0.0, device=self.device)
        
        total_loss = 0.0
        
        for i, bc in enumerate(self.data.bcs):
            x_bc = self.data.get_boundary_points(i)
            x_bc_tensor = self.data.to_tensor(x_bc, device=self.device, requires_grad=True)
            u_bc = self.model(x_bc_tensor)
            
            if bc.type == 'dirichlet':
                error = bc.error(x_bc, u_bc)
                total_loss += torch.mean(error ** 2)
            
            elif bc.type in ['neumann', 'robin']:
                # Compute gradient for Neumann/Robin BCs
                grad_u = grad(u_bc, x_bc_tensor)
                error = bc.error(x_bc, u_bc, grad_u)
                total_loss += torch.mean(error ** 2)
        
        return total_loss
    
    def compute_ic_loss(self):
        """Compute initial condition loss.
        
        Returns:
            IC loss (scalar tensor)
        """
        if len(self.data.ics) == 0:
            return torch.tensor(0.0, device=self.device)
        
        total_loss = 0.0
        
        for i, ic in enumerate(self.data.ics):
            x_ic = self.data.get_initial_points(i)
            x_ic_tensor = self.data.to_tensor(x_ic, device=self.device, requires_grad=False)
            u_ic = self.model(x_ic_tensor)
            
            error = ic.error(x_ic, u_ic)
            total_loss += torch.mean(error ** 2)
        
        return total_loss
    
    def compute_data_loss(self):
        """Compute data fitting loss from measurement points.
        
        Returns:
            Data loss (scalar tensor)
        """
        if not self.data.has_measurements():
            return torch.tensor(0.0, device=self.device)
        
        measurement_data = self.data.get_measurement_data()
        
        # Convert measurement data to tensors
        x_data = self.data.to_tensor(measurement_data['x'], 
                                     device=self.device, 
                                     requires_grad=False)
        u_data = self.data.to_tensor(measurement_data['u'], 
                                     device=self.device, 
                                     requires_grad=False)
        
        # Predict at measurement locations
        u_pred = self.model(x_data)
        
        # Compute MSE between predictions and measurements
        loss = torch.mean((u_pred - u_data) ** 2)
        
        return loss
    
    def compute_total_loss(self):
        """Compute total weighted loss.
        
        Returns:
            Total loss and individual loss components
        """
        # Domain points for PDE
        x_domain = self.data.get_domain_points()
        loss_pde = self.compute_pde_loss(x_domain)
        
        # Boundary conditions
        loss_bc = self.compute_bc_loss()
        
        # Initial conditions
        loss_ic = self.compute_ic_loss()
        
        # Measurement data
        loss_data = self.compute_data_loss()
        
        # Total weighted loss
        total_loss = (
            self.loss_weights['pde'] * loss_pde +
            self.loss_weights['bc'] * loss_bc +
            self.loss_weights['ic'] * loss_ic +
            self.loss_weights['data'] * loss_data
        )
        
        return total_loss, loss_pde, loss_bc, loss_ic, loss_data
    
    def train_step(self):
        """Perform one training step.
        
        Returns:
            Dictionary of loss values
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Compute losses
        total_loss, loss_pde, loss_bc, loss_ic, loss_data = self.compute_total_loss()
        
        # Backward pass
        total_loss.backward()
        self.optimizer.step()
        
        return {
            'loss': total_loss.item(),
            'loss_pde': loss_pde.item(),
            'loss_bc': loss_bc.item(),
            'loss_ic': loss_ic.item(),
            'loss_data': loss_data.item()
        }
    
    def train(self, epochs, print_every=100, resample_every=None):
        """Train the PINN model.
        
        Args:
            epochs: Number of training epochs
            print_every: Print loss every N epochs
            resample_every: Resample collocation points every N epochs (None = no resampling)
        """
        print(f"Starting training for {epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters())}")
        print(f"Loss weights: {self.loss_weights}")
        if self.data.has_measurements():
            print(f"Training with {self.data.get_measurement_data()['x'].shape[0]} measurement points")
        print("-" * 80)
        
        for epoch in range(epochs):
            # Resample points if specified
            if resample_every is not None and epoch % resample_every == 0 and epoch > 0:
                self.data.resample()
            
            # Training step
            losses = self.train_step()
            
            # Record history
            for key, value in losses.items():
                self.history[key].append(value)
            
            # Print progress
            if (epoch + 1) % print_every == 0 or epoch == 0:
                print(f"Epoch {epoch+1:6d} | "
                      f"Loss: {losses['loss']:.6e} | "
                      f"PDE: {losses['loss_pde']:.6e} | "
                      f"BC: {losses['loss_bc']:.6e} | "
                      f"IC: {losses['loss_ic']:.6e} | "
                      f"Data: {losses['loss_data']:.6e}")
        
        print("-" * 80)
        print("Training completed!")
    
    def predict(self, x):
        """Make predictions at given points.
        
        Args:
            x: Points for prediction (numpy array)
            
        Returns:
            Predictions (numpy array)
        """
        self.model.eval()
        with torch.no_grad():
            x_tensor = self.data.to_tensor(x, device=self.device, requires_grad=False)
            u = self.model(x_tensor)
            return u.cpu().numpy()
    
    def evaluate_at_measurements(self):
        """Evaluate model predictions at measurement locations.
        
        Returns:
            Dictionary with 'x', 'u_true', 'u_pred', and 'error' keys
        """
        if not self.data.has_measurements():
            print("No measurement data available")
            return None
        
        measurement_data = self.data.get_measurement_data()
        x_data = measurement_data['x']
        u_true = measurement_data['u']
        
        # Predict
        u_pred = self.predict(x_data)
        
        # Compute error metrics
        error = u_pred - u_true
        mse = np.mean(error ** 2)
        mae = np.mean(np.abs(error))
        max_error = np.max(np.abs(error))
        
        print(f"\nMeasurement Error Metrics:")
        print(f"  MSE:       {mse:.6e}")
        print(f"  MAE:       {mae:.6e}")
        print(f"  Max Error: {max_error:.6e}")
        
        return {
            'x': x_data,
            'u_true': u_true,
            'u_pred': u_pred,
            'error': error,
            'mse': mse,
            'mae': mae,
            'max_error': max_error
        }
    
    def save(self, filepath):
        """Save model and optimizer state.
        
        Args:
            filepath: Path to save the model
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'loss_weights': self.loss_weights
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load model and optimizer state.
        
        Args:
            filepath: Path to load the model from
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        self.loss_weights = checkpoint['loss_weights']
        print(f"Model loaded from {filepath}")
    
    def get_history(self):
        """Get training history.
        
        Returns:
            Dictionary of loss histories
        """
        return self.history