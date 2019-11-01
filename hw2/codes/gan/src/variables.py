import torch

from torch import nn


class LinearGenerator(nn.Module):
    def __init__(self, noise_dim=2, output_dim=2):
        super().__init__()
        self.W = nn.Parameter(torch.randn(noise_dim, output_dim))
        self.b = nn.Parameter(2 * torch.randn(output_dim))

    def forward(self, z):
        """
        Evaluate on a sample. The variable z contains one sample per row
        """
        # TODO


class LinearDualVariable(nn.Module):
    def __init__(self, input_dim=2):
        super().__init__()
        self.v = nn.Parameter(torch.randn(input_dim))

    def forward(self, x):
        """
        Evaluate on a sample. The variable x contains one sample per row
        """
        # TODO

    def enforce_lipschitz(self):
        """Enforce the 1-Lipschitz condition of the function"""
        with torch.no_grad():
            # TODO

