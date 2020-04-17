import torch.nn as nn
import torch.nn.functional as F

from torch import tanh
from MADDPG.config import *
from MADDPG.utils import hidden_init


class Actor(nn.Module):
    """Policy updates through the Actor (Policy) Model (towards the direction suggested by the Critic)"""

    def __init__(self, seed, state_size, action_size, fc1_units, fc2_units):
        """
        Initialize parameters and build the actor model.
        :param state_size: (int) State dimensions
        :param action_size: (int) Action dimensions
        :param seed: Random seed
        :param fc1_units: Number of units for the first hidden layer
        :param fc2_units: Number of units for the second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        # Batch normalization can help in reducing the variance between updates of the network weights
        self.batch_norm = nn.BatchNorm1d(fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        if state.dim() == 1:
            state = torch.unsqueeze(state, 0)

        layer_1 = self.batch_norm(F.relu(self.fc1(state)))
        layer_2 = F.relu(self.fc2(layer_1))
        return tanh(self.fc3(layer_2))
