import torch
import random


class Config:
    def __init__(self, seed):
        """
        Convenience class that will host parameters concerning the network architecture
        :param seed: (int) Random seed
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.seed = seed
        random.seed(seed)
        torch.manual_seed(seed)

        self.buffer_size = None
        self.batch_size = None
        self.epsilon = None
        self.epsilon_decay = None

        self.env = None
        self.env_info = None

        self.num_agents = None
        self.brain_name = None
        self.brain = None
        self.state_size = None
        self.action_size = None
        self.actor_fn = None
        self.actor_opt_fn = None
        self.critic_fn = None
        self.critic_opt_fn = None
        self.replay_fn = None
        self.noise_fn = None
        self.discount = None  # gamma parameter
        self.target_mix = None  # tau parameter

        self.max_episodes = None
        self.add_noise = None

        self.learn_every = None
        self.learn_number = None

        self.actor_path = None
        self.critic_path = None
        self.scores_path = None
        self.goal_score = .5
