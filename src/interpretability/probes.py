# Two probe types:
#   LinearProbe : single linear layer (W @ x + b)
#   MLPProbe    : one hidden layer (Linear -> ReLU -> Linear)
#
# Both support regression (output_dim=1) and classification (output_dim=K).

import torch
import torch.nn as nn


class LinearProbe(nn.Module):
    """
    Single linear layer probe.

    If this achieves high accuracy, the target property is linearly
    decodable from the representation — i.e., there exists a direction
    in activation space that corresponds to the property.

    Parameters
    ----------
    input_dim : int
        Dimension of the input representation (e.g., 256 for z_con).
    output_dim : int
        1 for regression, K for K-class classification.
    """

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class MLPProbe(nn.Module):
    """
    One-hidden-layer MLP probe.

    If this succeeds where LinearProbe fails, the property is encoded
    nonlinearly — present but not as a simple direction.

    Parameters
    ----------
    input_dim : int
        Dimension of the input representation.
    hidden_dim : int
        Hidden layer width. Default 128.
    output_dim : int
        1 for regression, K for K-class classification.
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---- Factory ----------------------------------------------------------------

def create_probe(
    probe_type: str,
    input_dim: int,
    output_dim: int,
    hidden_dim: int = 128,
) -> nn.Module:
    """
    Factory function to create a probe by name.

    Parameters
    ----------
    probe_type : str
        "linear" or "mlp".
    input_dim : int
        Representation dimension.
    output_dim : int
        1 for regression, K for classification.
    hidden_dim : int
        Hidden layer width for MLP probe. Ignored for linear.
    """
    if probe_type == "linear":
        return LinearProbe(input_dim, output_dim)
    elif probe_type == "mlp":
        return MLPProbe(input_dim, output_dim, hidden_dim)
    else:
        raise ValueError(f"Unknown probe type: {probe_type!r}. Use 'linear' or 'mlp'.")
