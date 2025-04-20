"""
Adaptive Neural Network Module (Improved Version)
Author: Dexter Garcia
Description: Modular, efficient feedforward neural net for regression or classification,
with support for parameter compression and forecasting.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple


class AdaptiveNeuralNet(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_layers: List[int] = [64, 32],
        activation=nn.ReLU,
        output_size: int = 1,
        task_type: str = "regression"  # or "classification"
    ):
        super(AdaptiveNeuralNet, self).__init__()
        self.task_type = task_type.lower()
        layers = []
        current_size = input_size

        for hidden_size in hidden_layers:
            layers.append(nn.Linear(current_size, hidden_size))
            layers.append(activation())
            current_size = hidden_size

        layers.append(nn.Linear(current_size, output_size))

        if self.task_type == "classification":
            layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def train_model(
    model: nn.Module,
    data: torch.Tensor,
    targets: torch.Tensor,
    epochs: int = 30,
    lr: float = 0.001,
    verbose: bool = True
) -> None:
    if hasattr(model, 'task_type') and model.task_type == "classification":
        criterion = nn.BCELoss()
    else:
        criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(data).squeeze()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        if verbose and (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")


def forecast(model: nn.Module, last_sequence: torch.Tensor, n_steps: int) -> List[float]:
    model.eval()
    preds = []
    current_seq = last_sequence.clone().detach()

    with torch.no_grad():
        for _ in range(n_steps):
            output = model(current_seq.unsqueeze(0)).squeeze().item()
            preds.append(output)
            current_seq = torch.roll(current_seq, shifts=-1, dims=0)
            current_seq[-1] = output

    return preds


def compress_parameters(model: nn.Module, threshold: float = 0.01) -> None:
    with torch.no_grad():
        for param in model.parameters():
            param.abs_()
            mask = param < threshold
            param[mask] = 0


def count_nonzero_params(model: nn.Module) -> int:
    return sum((p != 0).sum().item() for p in model.parameters())


def count_total_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    import numpy as np

    # Example: Forecasting a sine wave
    seq_length = 10
    data = np.array([np.sin(x * 0.1) for x in range(100)])

    X = []
    y = []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    model = AdaptiveNeuralNet(input_size=seq_length, hidden_layers=[32, 16], task_type="regression")
    train_model(model, X, y, epochs=50)

    print("Forecasting next 5 steps:")
    prediction = forecast(model, X[-1], n_steps=5)
    print(prediction)

    compress_parameters(model, threshold=0.05)
    print("Compressed parameter count:", count_nonzero_params(model))
    print("Total parameters:", count_total_params(model))
