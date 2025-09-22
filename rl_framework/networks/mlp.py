"""Network building utilities."""
from __future__ import annotations

from typing import Iterable, List, Sequence, Type

import torch
from torch import nn


def build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_sizes: Sequence[int],
    activation: Type[nn.Module] = nn.ReLU,
    output_activation: Type[nn.Module] | None = None,
) -> nn.Sequential:
    """Construct a multilayer perceptron."""

    layers: List[nn.Module] = []
    last_dim = input_dim
    for hidden_dim in hidden_sizes:
        layers.append(nn.Linear(last_dim, hidden_dim))
        layers.append(activation())
        last_dim = hidden_dim
    layers.append(nn.Linear(last_dim, output_dim))
    if output_activation is not None:
        layers.append(output_activation())
    return nn.Sequential(*layers)
