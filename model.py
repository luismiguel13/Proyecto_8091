import torch
from torch import nn
import numpy as np


class WireActivation(nn.Module):
    """
    Real-valued WIRE: Re{ e^{j * omega * x} * exp(-(x/sigma)^2) } = cos(omega * x) * exp(-(x/sigma)^2)
    Note: This is the real part of the complex Gabor-based WIRE activation; sigma maps inversely to the 'spread' in some formulations.
    """
    def __init__(self, omega: float = 1.0, sigma: float = 1.0):
        super().__init__()
        self.omega = float(omega)
        self.sigma = float(sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cos(self.omega * x) * torch.exp(- (x / self.sigma) ** 2)

class SineActivation(nn.Module):
    """
    Sinusoidal activation as in SIREN-like networks: sin(w0 * x).
    """
    def __init__(self, w0: float = 1.0):
        super().__init__()
        self.w0 = float(w0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.w0 * x)


class GaussianFourierFeatures(torch.nn.Module):
    def __init__(self, input_dim, num_features, sigma=10.0):
        super().__init__()
        self.num_features = num_features
        self.sigma = sigma
        
        # Matriz de frecuencias aleatorias
        self.register_buffer(
            'freqs', 
            torch.randn(num_features, input_dim) * sigma
        )
        
    def forward(self, coords):
        """
        coords: coordenadas normalizadas [N, input_dim]
        """
        # Aplicar transformación Fourier
        proj = 2 * np.pi * coords @ self.freqs.T
        return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)

# ---- CMLP ----
class CMLP(nn.Module):
    """
    Coordinate MLP adjustable for single- or multi-output regression.
    - in_features: 2 (x,y) or 3 (x,y,z)
    - out_features: 1 (single-output) or K (multi-output)
    - hidden_layers: number of hidden layers (n)
    - hidden_units: neurons per hidden layer (m)
    - activation: 'wire', 'sine', or 'relu'
    - wire_omega, wire_sigma: parameters for WIRE
    - sine_w0: frequency scale for sinusoidal activation
    """
    def __init__(
        self,
        in_features: int = 3,
        out_features: int = 1,
        hidden_layers: int = 5,
        hidden_units: int = 512,
        activation: str = 'wire',
        wire_omega: float = 1.0,
        wire_sigma: float = 1.0,
        sine_w0: float = 1.0,
        input_transform: bool = True,
        final_activation: nn.Module | None = None,
    ):
        super().__init__()
        assert hidden_layers >= 1, "Use at least 1 hidden layer"
        layers = []

        # Select activation module factory
        def make_act():
            if activation.lower() == 'wire':
                return WireActivation(omega=wire_omega, sigma=wire_sigma)
            elif activation.lower() == 'sine':
                return SineActivation(w0=sine_w0)
            elif activation.lower() == 'relu':
                return nn.ReLU()
            else:
                raise ValueError(f"Unsupported activation: {activation}")

        # Input layer
        if input_transform == True:
            layers.append(GaussianFourierFeatures(
                input_dim=2, num_features=128, sigma=10.0
            ))
            layers.append(nn.Linear(256, hidden_units))
        else:
            layers.append(nn.Linear(in_features, hidden_units))

        layers.append(make_act())

        # Hidden layers
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_units, hidden_units))
            layers.append(make_act())

        # Output layer
        layers.append(nn.Linear(hidden_units, out_features))
        if final_activation is not None:
            layers.append(final_activation)

        self.net = nn.Sequential(*layers)

        # Optional: Xavier initialization for linear layers
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        coords: shape [B, in_features] with normalized coordinates.
        returns: shape [B, out_features]
        """
        return self.net(coords)
