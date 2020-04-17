import numpy as np

from MADDPG.config import *
from collections import namedtuple, deque


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed, device):
        """
        Initialize a ReplayBuffer object.
        :param action_size: (int) dimension of each action
        :param buffer_size: (int) maximum size of buffer
        :param batch_size: (int) size of each training batch
        :param seed: (int) random seed
        :param device: (torch.device) The device to register tasks to (CPU or CUDA-enabled GPU)
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done_signal"])
        self.seed = random.seed(seed)
        self.device = device
    
    def add(self, state, action, reward, next_state, done_signal):
        """
        Add a new experience to memory.
        :param state: (array_like) current state
        :param action: (array_like) current action
        :param reward: (array_like) current reward
        :param next_state: (array_like) state following the current state
        :param done_signal: (bool) whether the current state is terminal or not
        :return: None
        """
        experience = self.experience(state, action, reward, next_state, done_signal)
        self.memory.append(experience)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack(
            [e.next_state for e in experiences if e is not None])).float().to(self.device)
        done_signals = torch.from_numpy(np.vstack(
            [e.done_signal for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)

        return (states, actions, rewards, next_states, done_signals)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
