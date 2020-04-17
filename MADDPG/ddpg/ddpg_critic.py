import torch.nn as nn
import torch.nn.functional as F

from MADDPG.config import *
from MADDPG.utils import hidden_init


class Critic(nn.Module):
    """Value function approximation model, a.k.a the 'Critic'"""

    def __init__(self, seed, state_size, action_size, num_agents, fc1_units, fc2_units):
        """
        Initialize parameters and build the critic model.
        :param seed: (int) Random seed to reproduce results
        :param state_size: (int) State dimensions
        :param action_size: (int) Action dimensions
        :param seed: Random seed
        :param fc1_units: Number of units for the first hidden layer
        :param fc2_units: Number of units for the second hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear((state_size + action_size) * num_agents, fc1_units)
        self.batch_norm = nn.BatchNorm1d(fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.reset_parameters()

    def reset_parameters(self):
        """Initialise network weights given a range that is inversely proportial to the layer's size (no. of units)"""
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs to Q-values."""
        if state.dim() == 1:
            state = torch.unsqueeze(state, 0)

        # Modified DDPG architecture
        # The actions and states should be concatenated directly at the input of the network
        concat_state_action = torch.cat((state, action.float()), dim=1)
        layer_1 = self.batch_norm(F.relu(self.fc1(concat_state_action)))
        layer_2 = F.relu(self.fc2(layer_1))
        return self.fc3(layer_2)
