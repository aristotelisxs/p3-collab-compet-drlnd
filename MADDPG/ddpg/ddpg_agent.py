import numpy as np
import torch.nn.functional as F

from MADDPG.config import *
from MADDPG.utils import OrnsteinUhlenbeckNoise


class DDPGAgent:
    """The agent that will interact with and learn from the environment"""
    def __init__(self, cnfg, agent_id):
        # Collect what we need from pre-set configurations
        self.state_size = cnfg.state_size
        self.action_size = cnfg.action_size
        self.seed = random.seed(cnfg.seed)
        self.agent_id = agent_id
        self.discount = cnfg.discount
        self.device = cnfg.device
        self.target_mix = cnfg.target_mix
        self.num_agents = cnfg.num_agents

        self._init_actor(cnfg)
        self._init_critic(cnfg)

        # Make sure the Critic Target Network has the same weight values as the Local Network
        for target, local in zip(self.target_critic.parameters(), self.local_critic.parameters()):
            target.data.copy_(local.data)

        # Make sure that the target-local model pairs are initialized to the 
        # same weights
        self.hard_update(self.local_actor, self.target_actor)
        self.hard_update(self.local_critic, self.target_critic)

        self.noise = cnfg.noise_fn()

        self.epsilon = cnfg.epsilon
        self.epsilon_decay = cnfg.epsilon_decay

    def _init_actor(self, cnfg):
        self.local_actor = cnfg.actor_fn().to(self.device)
        self.target_actor = cnfg.actor_fn().to(self.device)
        self.actor_opt = cnfg.actor_opt_fn(self.local_actor.parameters())

    def _init_critic(self, cnfg):
        self.local_critic = cnfg.critic_fn().to(self.device)
        self.target_critic = cnfg.critic_fn().to(self.device)
        self.critic_opt = cnfg.critic_opt_fn(self.local_critic.parameters())

    def act(self, state, add_noise=False):
        """
        Returns actions for the given state as per current policy.
        :param state: The current state
        :param add_noise: (bool) Whether to add exploration noise
        :return: (int) The action to take, between values of [-1, 1] inclusive
        """
        state = torch.from_numpy(state).float().to(self.device)

        self.local_actor.eval()
        with torch.no_grad():
            action = self.local_actor(state).cpu().data.numpy()

        self.local_actor.train()

        if add_noise:
            action += self.noise.sample() * self.epsilon
            self._decay_epsilon()

        return np.clip(action, -1, 1)

    def reset(self):
        """Resets the OU Noise for this agent."""
        self.noise.reset()

    def _learn_critic(self, experiences, next_actions, agent_id_tensor):
        """
        Critic updates triggered by enough samples in the replay buffer
        :param experiences: (collection) The accumulated set of experiences and their properties
        :param next_actions: (array-like) The next actions that were taken by each agent
        :param agent_id_tensor: (torch.tensor) The id of the agent
        :return:
        """
        states, actions, rewards, next_states, done_signals = experiences

        # Update Q-functions
        self.critic_opt.zero_grad()
        Q_targets_next = self.target_critic(next_states, next_actions)

        # Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        Q_targets = rewards.index_select(1, agent_id_tensor) + (
                    self.discount * Q_targets_next * (1 - done_signals.index_select(1, agent_id_tensor)))
        Q_expected = self.local_critic(states, actions)

        # Optimization step
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        critic_loss.backward()
        self.critic_opt.step()

    def _learn_actor(self, states, predicted_actions):
        """
        Actor updates triggered by enough samples in the replay buffer
        :param states: Current state of each agent
        :param predicted_actions: Predictions of actions for each agent's current state
        :return: None
        """
        # Optimization steps
        self.actor_opt.zero_grad()

        actor_loss = -self.local_critic(states, predicted_actions).mean()
        actor_loss.backward()
        self.actor_opt.step()

    def learn(self, experiences, next_actions, predicted_actions):
        """
        Policy and value parameter updates using sampled experience tuples from the replay buffer
        :param experiences: (collection) The accumulated set of experiences and their properties
        :param next_actions: The next actions that were taken by each agent
        :param predicted_actions: Predictions of actions for each agent's current state
        :return: None
        """
        agent_id_tensor = torch.tensor([self.agent_id - 1]).to(self.device)

        self._learn_critic(experiences, next_actions, agent_id_tensor)
        self._learn_actor(experiences[0], predicted_actions)

        # Update target networks
        self.soft_update(self.local_critic, self.target_critic)
        self.soft_update(self.local_actor, self.target_actor)

    def hard_update(self, local_model, target_model):
        """Hard update model parameters.
        :param local_model: The model currently being trained
        :param target_model: The model to copy over parameters from the local model
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            # Ignore target mix and copy over the local (online) network's entirely, i.e. set θ_target = θ_local
            target_param.data.copy_(local_param.data)

    def soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
            where τ is the interpolation parameter (self.config.target_mix)
        :param local_model: (pytorch.model) The model that is being trained
        :param target_model: (pytorch.model) The network with frozen weights
        :return: None
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            # Mix in local's network weight's into the frozen target network's weight by a small percentage
            target_param.data.copy_(self.target_mix * local_param.data + (1.0 - self.target_mix) * target_param.data)

    def _decay_epsilon(self):
        """Reduce exploration noise amplification by a decay rate."""
        self.epsilon *= self.epsilon_decay
