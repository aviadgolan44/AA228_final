import torch
import torch.nn as nn


class DQN(nn.Module):
    """
    Creates double-headed Deep Q Neural Network.

    Fields:
    input_dim - number of inputs for state
    output_dim - number of actions possible
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)

        # output layers for x and y
        self.fcx = nn.Linear(32, output_dim)
        self.fcy = nn.Linear(32, output_dim)

    def forward(self, x):
        """
        Computes forward pass of model.

        Inputs:
        x - input vector
        """
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fcx(x), self.fcy(x)
