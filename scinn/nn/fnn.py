"""Neural network models for PINNs."""

__all__ = ["FNN", "ModifiedMLP"]

import torch
import torch.nn as nn
import numpy as np


class FNN(nn.Module):
    """Fully connected neural network (Feedforward Neural Network).
    
    Args:
        layer_sizes: List of layer sizes [input_dim, hidden1, hidden2, ..., output_dim]
        activation: Activation function ('tanh', 'relu', 'sigmoid', 'sin', 'gelu')
        initializer: Weight initialization method
        
    Example:
        model = FNN([2, 50, 50, 50, 1], activation='tanh')
    """
    
    def __init__(self, layer_sizes, activation='tanh', initializer='xavier_normal'):
        super().__init__()
        
        self.layer_sizes = layer_sizes
        self.activation_name = activation
        self.initializer = initializer
        
        # Build layers
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
        
        # Set activation function
        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'sin':
            self.activation = lambda x: torch.sin(x)
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for layer in self.layers[:-1]:  # All hidden layers
            if self.initializer == 'xavier_normal':
                nn.init.xavier_normal_(layer.weight)
            elif self.initializer == 'xavier_uniform':
                nn.init.xavier_uniform_(layer.weight)
            elif self.initializer == 'kaiming_normal':
                nn.init.kaiming_normal_(layer.weight)
            elif self.initializer == 'kaiming_uniform':
                nn.init.kaiming_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        
        # Initialize output layer with smaller weights
        nn.init.xavier_normal_(self.layers[-1].weight, gain=0.1)
        nn.init.zeros_(self.layers[-1].bias)
    
    def forward(self, x):
        """Forward pass through the network."""
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        return x


class ModifiedMLP(nn.Module):
    """Modified MLP with Fourier features for better performance.
    
    This architecture uses a modified input layer to help with learning
    high-frequency functions, based on the paper:
    "Fourier Features Let Networks Learn High Frequency Functions"
    
    Args:
        layer_sizes: List of layer sizes
        activation: Activation function
        fourier_features: Whether to use Fourier features
        sigma: Scale parameter for Fourier features
    """
    
    def __init__(self, layer_sizes, activation='tanh', 
                 fourier_features=False, sigma=1.0):
        super().__init__()
        
        self.fourier_features = fourier_features
        self.sigma = sigma
        
        if fourier_features:
            # Random Fourier feature mapping
            self.B = torch.randn(layer_sizes[0], layer_sizes[1] // 2) * sigma
            actual_input_size = layer_sizes[1]
        else:
            actual_input_size = layer_sizes[0]
        
        # Build network
        self.layers = nn.ModuleList()
        sizes = [actual_input_size] + layer_sizes[1:]
        
        for i in range(len(sizes) - 1):
            self.layers.append(nn.Linear(sizes[i], sizes[i+1]))
        
        # Activation
        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sin':
            self.activation = lambda x: torch.sin(x)
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for layer in self.layers:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        if self.fourier_features:
            # Apply Fourier feature mapping
            x_proj = 2 * np.pi * x @ self.B.to(x.device)
            x = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        return x


class ResNet(nn.Module):
    """Residual neural network for PINNs.
    
    Args:
        layer_sizes: List of layer sizes
        activation: Activation function
        num_res_blocks: Number of residual blocks
    """
    
    def __init__(self, layer_sizes, activation='tanh', num_res_blocks=4):
        super().__init__()
        
        self.input_layer = nn.Linear(layer_sizes[0], layer_sizes[1])
        
        # Residual blocks
        self.res_blocks = nn.ModuleList()
        for _ in range(num_res_blocks):
            self.res_blocks.append(ResidualBlock(layer_sizes[1], activation))
        
        self.output_layer = nn.Linear(layer_sizes[1], layer_sizes[-1])
        
        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sin':
            self.activation = lambda x: torch.sin(x)
        else:
            self.activation = nn.Tanh()
    
    def forward(self, x):
        x = self.activation(self.input_layer(x))
        
        for block in self.res_blocks:
            x = block(x)
        
        x = self.output_layer(x)
        return x


class ResidualBlock(nn.Module):
    """Residual block for ResNet."""
    
    def __init__(self, hidden_size, activation='tanh'):
        super().__init__()
        
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        
        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sin':
            self.activation = lambda x: torch.sin(x)
        else:
            self.activation = nn.Tanh()
    
    def forward(self, x):
        residual = x
        out = self.activation(self.linear1(x))
        out = self.linear2(out)
        out = out + residual
        out = self.activation(out)
        return out
