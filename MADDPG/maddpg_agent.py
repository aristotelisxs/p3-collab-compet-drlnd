import os
import numpy as np

from MADDPG.ddpg.ddpg_agent import DDPGAgent
from MADDPG.config import *
from MADDPG.memory import ReplayBuffer


class MADDPGAgent:
    """Wrapper managing different agents in the environment."""

    def __init__(self, cnfg):
        """
        Wrapper for MADDPG algorithm functions and classes
        :param cnfg: (MADDPG.config.Config) Configuration class
        """
        self.num_agents = cnfg.num_agents
        self.state_size = cnfg.state_size
        self.action_size = cnfg.action_size
        self.device = cnfg.device
        self.learn_every = cnfg.learn_every
        self.learn_number = cnfg.learn_number
        self.seed = cnfg.seed
        self.buffer_size = cnfg.buffer_size
        self.batch_size = cnfg.batch_size

        self.agents = [DDPGAgent(cnfg, i+1) for i in range(self.num_agents)]
        
        # Replay buffer
        self.memory = ReplayBuffer(self.action_size, self.buffer_size, self.batch_size, self.seed, self.device)
        
        # Will help to decide when to update the model weights
        self.time_step = 0
        
        # Directory where to save the model
        self.model_dir = os.getcwd() + "/MADDPG/saved_models"
        os.makedirs(self.model_dir, exist_ok=True)
    
    def reset(self):
        """Resets OrnsteinUhlenbeck Noise for every agent."""
        for agent in self.agents:
            agent.reset()
            
    def act(self, observations, add_noise=False):
        """
        Picks an action for every agent depending on their individual observations and the current policy.
        :param observations: (array-like) A collection of observations for each agent managed by MADDPG
        :param add_noise: (bool) If exploration noise should be added or not
        :return:
        """
        actions = list()
        for agent, observation in zip(self.agents, observations):
            action = agent.act(observation, add_noise=add_noise)
            actions.append(action.reshape(-1))

        return np.array(actions)
    
    def step(self, states, actions, rewards, next_states, done_signals):
        """
        Save experience in replay buffer, and sample randomly from it to learn.
        :param states: (array-like) Set of current states for each agent
        :param actions: (array-like) Set of current actions taken by each agent
        :param rewards: (array-like) Set of rewards received by each action taken from agents
        :param next_states: (array-like) The set of states each agent has moved to after taking an action
        :param done_signals: (array-like) Which of the agents have reached a terminal state
        :return: None
        """
        """"""
        states = states.reshape(1, -1)
        actions = actions.reshape(1, -1)
        next_states = next_states.reshape(1, -1)
        
        self.memory.add(states, actions, rewards, next_states, done_signals)
        
        self.time_step = (self.time_step + 1) % self.learn_every  # Resets the timer to re-count next learning event

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.batch_size and self.time_step == 0:
            for a_i, agent in enumerate(self.agents):
                for _ in range(self.learn_number):
                    experiences = self.memory.sample()
                    self.learn(experiences, a_i)

    def learn(self, experiences, agent_number):
        """
        Learn from all observations recorded by each agent, i.e. each agent contributes to taking the next action
        within the environment
        :param experiences: (array-like) Set of experience from each agent managed by MADDPG
        :param agent_number: (int) Reference ID of the agent currently examined
        :return: None
        """
        next_actions = []
        actions_pred = []
        states, _, _, next_states, _ = experiences
        
        next_states = next_states.reshape(-1, self.num_agents, self.state_size)
        states = states.reshape(-1, self.num_agents, self.state_size)
        
        for agent_id, agent in enumerate(self.agents):
            agent_id_tensor = self._get_agent_number(agent_id)
            
            state = states.index_select(1, agent_id_tensor).squeeze(1)
            next_state = next_states.index_select(1, agent_id_tensor).squeeze(1)
            
            next_actions.append(agent.target_actor(next_state))
            actions_pred.append(agent.local_actor(state))
            
        next_actions = torch.cat(next_actions, dim=1).to(self.device)
        actions_pred = torch.cat(actions_pred, dim=1).to(self.device)
        
        agent = self.agents[agent_number]
        agent.learn(experiences, next_actions, actions_pred)
            
    def save_model(self, ts_started, scores_total, moving_averages):
        """Saves model weights to file."""
        for i in range(self.num_agents):
            torch.save(
                self.agents[i].local_actor.state_dict(),
                os.path.join(self.model_dir, 'actor_params_{}_{}.pth'.format(i, ts_started))
            )
            torch.save(
                self.agents[i].actor_opt.state_dict(),
                os.path.join(self.model_dir, 'actor_optim_params_{}_{}.pth'.format(i, ts_started))
            )
            torch.save(
                self.agents[i].local_critic.state_dict(),
                os.path.join(self.model_dir, 'critic_params_{}_{}.pth'.format(i, ts_started))
            )
            torch.save(
                self.agents[i].critic_opt.state_dict(),
                os.path.join(self.model_dir, 'critic_optim_params_{}_{}.pth'.format(i, ts_started))
            )

        with open('scores_chkpnt_{}.txt'.format(ts_started), 'w') as f:
            for x in scores_total:
                f.write(str(x) + '\n')

        f.close()

        with open('rolling_averages_chkpnt_{}.txt'.format(ts_started), 'w') as f:
            for x in moving_averages:
                f.write(str(x) + '\n')

        f.close()
    
    def load_model(self, ts_started):
        """Loads weights from saved model."""
        for i in range(self.num_agents):
            self.agents[i].local_actor.load_state_dict(
                torch.load(os.path.join(self.model_dir, 'actor_params_{}_{}.pth'.format(i, ts_started)))
            )
            self.agents[i].actor_opt.load_state_dict(
                torch.load(os.path.join(self.model_dir, 'actor_optim_params_{}_{}.pth'.format(i, ts_started)))
            )
            self.agents[i].local_critic.load_state_dict(
                torch.load(os.path.join(self.model_dir, 'critic_params_{}_{}.pth'.format(i, ts_started)))
            )
            self.agents[i].critic_opt.load_state_dict(
                torch.load(os.path.join(self.model_dir, 'critic_optim_params_{}_{}.pth'.format(i, ts_started)))
            )

    def _get_agent_number(self, agent_id):
        """
        Get the agent number as a Torch.tensor. Aids in collecting experience tuple properties.
        :param agent_id: (int) The agent's id
        :return: None
        """
        return torch.tensor([agent_id]).to(self.device)
