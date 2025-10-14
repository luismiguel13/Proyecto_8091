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


class AnisotropicPositionalEncoding(nn.Module):
    """
    Anisotropic Positional Encoding para coordenadas.
    
    Args:
        M: número de frecuencias para eje x (tiempo)
        N: número de frecuencias para eje y (receptor)  
        K: número de frecuencias para eje z (fuente)
        sampling: 'linear' o 'exponential'
    """
    def __init__(self, M, N, sampling='linear'):
        super().__init__()
        self.M = M
        self.N = N
        self.sampling = sampling
        
        # Generar frecuencias para cada eje
        self.freq_x = self._generate_frequencies(M)
        self.freq_y = self._generate_frequencies(N)
        
    
    def _generate_frequencies(self, U):
        """Genera vector de frecuencias según tipo de muestreo"""
        if self.sampling == 'linear':
            # π_i = i*π
            freqs = torch.arange(1, U + 1, dtype=torch.float32) * np.pi
        elif self.sampling == 'exponential':
            # π_i = 2^(i-1) * π
            freqs = (2.0 ** torch.arange(0, U, dtype=torch.float32)) * np.pi
        else:
            raise ValueError("sampling debe ser 'linear' o 'exponential'")
        return freqs
    
    def encode_coordinate(self, coord, frequencies):
        """
        Codifica una coordenada con funciones sinusoidales.
        
        Args:
            coord: tensor de forma (batch_size, 1)
            frequencies: tensor de frecuencias (U,)
        Returns:
            tensor de forma (batch_size, 2*U)
        """
        productos = coord * frequencies.unsqueeze(0)
        
        # Aplicar cos y sin
        cos_encoding = torch.cos(productos)
        sin_encoding = torch.sin(productos)
        
        # Concatenar: [cos(π_1*v), sin(π_1*v), ..., cos(π_U*v), sin(π_U*v)]
        encoding = torch.cat([cos_encoding, sin_encoding], dim=1)
        
        return encoding
    
    def forward(self, coords):
        """
        Args:
            coords: tensor de forma (batch_size, 3) con columnas [x, y, z]
                   donde x=tiempo, y=receptor, z=fuente
                   Las coordenadas deben estar normalizadas en [0, 1]
        Returns:
            tensor de forma (batch_size, 2*(M+N+K))
        """
        x = coords[:, 0:1]  
        y = coords[:, 1:2]  
        
        # Codificar cada coordenada
        encoding_x = self.encode_coordinate(x, self.freq_x)  # (batch_size, 2*M)
        encoding_y = self.encode_coordinate(y, self.freq_y)  # (batch_size, 2*N)
        
        # Concatenar todas las codificaciones
        full_encoding = torch.cat([encoding_x, encoding_y], dim=1)
        
        return full_encoding

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
        input_transform: str = "FF",
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
            elif activation.lower() == 'sigmoid':
                return nn.Sigmoid()
            elif activation.lower() == 'tanh':
                return nn.Tanh()
            else:
                raise ValueError(f"Unsupported activation: {activation}")

        # Input layer
        if input_transform == "GFF":
            layers.append(GaussianFourierFeatures(
                input_dim=2, num_features=128, sigma=10.0
            ))
            layers.append(nn.Linear(256, hidden_units))

        if input_transform == "FF":
            layers.append(AnisotropicPositionalEncoding(M=8, N=5, sampling="linear"))        
            # Dimensión de entrada del MLP
            input_dim =  2 * (8 + 5)
            layers.append(nn.Linear(input_dim, hidden_units))

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
