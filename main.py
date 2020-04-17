import argparse

from torch import optim
from datetime import datetime
from MADDPG.config import Config
from unityagents import UnityEnvironment
from MADDPG.maddpg_agent import MADDPGAgent
from MADDPG.ddpg.ddpg_actor import Actor
from MADDPG.ddpg.ddpg_critic import Critic
from MADDPG.memory import ReplayBuffer
from MADDPG.utils import OrnsteinUhlenbeckNoise, str2bool
from MADDPG.driver import plot_results, train


def collect_params():
    """
    Collect parameters from the command line
    :return: (argparse.ArgumentParser()) Collection with all parameters (default or modified based on user input).
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', dest='seed', help="Seed to reproduce results", type=int, required=True)
    parser.add_argument('--buffer_size', dest='buffer_size',
                        help="Maximum steps that an agent can take within an episode before reaching the terminal "
                             "state",
                        type=int, required=False, default=int(1e6))
    parser.add_argument('--discount', dest='discount',
                        help="Discount rewards (a.k.a. gamma parameter) by a factor between 0 and 1.",
                        type=float, required=False, default=.99)
    parser.add_argument('--target_mix', dest='target_mix',
                        help="How much of the local network's weights to 'mix in' with the target network's at each "
                             "time step (a.k.a. tau parameter)",
                        type=float, required=False, default=5e-2)
    parser.add_argument('--lr_actor', dest='lr_actor',
                        help="The learning rate used for the network weights' update step",
                        type=int, required=False, default=5e-4)
    parser.add_argument('--lr_critic', dest='lr_critic',
                        help="The learning rate used for the network weights' update step",
                        type=int, required=False, default=5e-4)
    parser.add_argument('--learn_every', dest='learn_every',
                        help="After how many times should the network be allowed to update its network weights and "
                             "learn",
                        type=int, required=False, default=2)
    parser.add_argument('--learn_number', dest='learn_number',
                        help="How many times should the network be updated after `learn_every` steps. "
                             "By default, this is unused (set to 1)",
                        type=int, required=False, default=1)
    parser.add_argument('--epsilon', dest='epsilon',
                        help="Exploration noise amplification to learn the optimal policy more quickly.",
                        type=int, required=False, default=1)
    parser.add_argument('--epsilon_decay', dest='epsilon_decay',
                        help="Exploration noise amplification decay, i.e. with what rate we will decrease noise "
                             "introduced during agent optimaly policy exploration.",
                        type=int, required=False, default=1)
    parser.add_argument('--ou_noise_mu', dest='ou_noise_mu',
                        help="Ornstein Uhlenbeck noise function mean parameter",
                        type=float, required=False, default=0.)
    parser.add_argument('--ou_noise_sigma', dest='ou_noise_sigma',
                        help="Ornstein Uhlenbeck noise function sigma parameter",
                        type=float, required=False, default=0.1)
    parser.add_argument('--ou_noise_theta', dest='ou_noise_theta',
                        help="Ornstein Uhlenbeck noise function sigma parameter (mean reversion rate)",
                        type=float, required=False, default=0.15)
    parser.add_argument('--batch_size', dest='batch_size',
                        help="Batch size to use for the neural network updates",
                        type=int, required=False, default=512)
    parser.add_argument('--reacher_fp', dest='reacher_fp',
                        help="The relative to the script's location or full file path of the folder containing the "
                             "Unity environment for the robotic arm control.",
                        type=str, required=False, default='Tennis_Windows_x86_64/Tennis.exe')
    parser.add_argument('--fc1_units', dest='fc1_units',
                        help="The number of units for the first hidden layer of the fully connected inference network",
                        type=int, required=False, default=256)
    parser.add_argument('--fc2_units', dest='fc2_units',
                        help="The number of units for the second hidden layer of the fully connected inference network",
                        type=int, required=False, default=256)
    parser.add_argument('--max_episodes', dest='max_episodes',
                        help="Maximum episodes to run the agent training procedure for.",
                        type=int, required=False, default=int(2500))
    parser.add_argument('--add_noise', dest='add_noise',
                        help="Whether we should add noise or not.",
                        type=str2bool, nargs='?', required=False, default=False)

    return parser


def init_config_obj(args):
    config = Config(seed=args.seed)

    # Environment related configurations
    config.env = UnityEnvironment(seed=config.seed, file_name=args.reacher_fp)
    config.brain_name = config.env.brain_names[0]
    config.brain = config.env.brains[config.brain_name]
    config.env_info = config.env.reset(train_mode=True)[config.brain_name]
    config.num_agents = len(config.env_info.agents)

    # Neural net related configuration
    config.batch_size = args.batch_size
    config.state_size = config.env_info.vector_observations.shape[1]
    config.action_size = config.env.brains[config.brain_name].vector_action_space_size
    config.states = config.env_info.vector_observations
    config.epsilon = args.epsilon
    config.epsilon_decay = args.epsilon_decay
    config.discount = args.discount
    config.target_mix = args.target_mix
    config.learn_every = args.learn_every
    config.learn_number = args.learn_number
    config.add_noise = args.add_noise

    # Neural nets
    config.actor_fn = lambda: Actor(config.seed, config.state_size, config.action_size,
                                    fc1_units=args.fc1_units, fc2_units=args.fc2_units)
    config.actor_opt_fn = lambda params: optim.Adam(params, lr=args.lr_actor)

    config.critic_fn = lambda: Critic(config.seed, config.state_size, config.action_size, config.num_agents,
                                      fc1_units=args.fc1_units, fc2_units=args.fc2_units)
    config.critic_opt_fn = lambda params: optim.Adam(params, lr=args.lr_critic)

    config.replay_fn = lambda: ReplayBuffer(config.action_size, buffer_size=args.buffer_size,
                                            batch_size=config.batch_size,
                                            seed=args.seed, device=config.device
                                            )
    config.noise_fn = lambda: OrnsteinUhlenbeckNoise(config.action_size,
                                                     mu=args.ou_noise_mu,
                                                     theta=args.ou_noise_theta,
                                                     sigma=args.ou_noise_sigma)

    # Other
    config.max_episodes = args.max_episodes

    return config


if __name__ == "__main__":
    args = collect_params().parse_args()
    config = init_config_obj(args=args)
    cur_ts = str(datetime.now().timestamp())

    maddpg = MADDPGAgent(cnfg=config)
    scores_total, rolling_score_averages = train(maddpg, config, cur_ts, total_episodes=config.max_episodes)
    plot_results(scores_total, rolling_score_averages, save_to_fn="p3_agent_training_scores_" + cur_ts + ".png")
