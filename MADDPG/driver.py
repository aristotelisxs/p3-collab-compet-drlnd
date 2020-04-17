import numpy as np
import progressbar as pb
import matplotlib.pyplot as plt

from collections import deque


def train(maddpg, cnfg, ts_started, total_episodes=1000, chkpoint_after=100):
    """
    Train agents using the MADDPG algorithm given an environment
    :param maddpg: Wrapper for MADDPG related classes
    :param cnfg: Configuration class that contains hyper-parameters relating to the network architecture and environment
     setup
     :param ts_started: The UNIX timestamp that the experiment has been initialized
    :param total_episodes: For how many episodes at maximum should the algorithm run
    :param chkpoint_after: After how many episodes should we create model checkpoints?
    :return: (list, list) List of scores and rolling average scores
    """
    widget = [
        "Episode #: ", pb.Counter(), '::', str(total_episodes), ' | ',
        pb.Percentage(), ' | ', pb.ETA(), ' | ', pb.Bar(marker=pb.RotatingMarker()), ' | ',
        'Rolling Average: ', pb.FormatLabel('')
    ]
    timer = pb.ProgressBar(widgets=widget, maxval=total_episodes).start()

    is_solved = False
    scores_total = []
    scores_deque = deque(maxlen=100)
    score_rolling_averages = []
    last_best_score = 0.

    # Environment information
    brain_name = cnfg.brain_name

    for i_episode in range(1, total_episodes + 1):
        current_average = 0. if i_episode == 1 else score_rolling_averages[-1]  # Check if first episode
        widget[12] = pb.FormatLabel(str(current_average)[:6])
        timer.update(i_episode)

        states = cnfg.env_info.vector_observations[:, -cnfg.state_size:]
        scores = np.zeros(cnfg.num_agents)
        maddpg.reset()

        while True:
            actions = maddpg.act(states, cnfg.add_noise)

            cnfg.env_info = cnfg.env.step(actions)[brain_name]
            next_states = cnfg.env_info.vector_observations[:, -cnfg.state_size:]
            rewards = cnfg.env_info.rewards
            done_signals = cnfg.env_info.local_done
            
            maddpg.step(states, actions, rewards, next_states, done_signals)

            scores += rewards
            states = next_states

            if np.any(done_signals):
                break

        cnfg.env_info = cnfg.env.reset(train_mode=True)[brain_name]
        max_episode_score = np.max(scores)

        scores_deque.append(max_episode_score)
        scores_total.append(max_episode_score)

        score_average = np.mean(scores_deque)
        score_rolling_averages.append(score_average)

        if score_average >= cnfg.goal_score and not is_solved:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(
                i_episode, score_average
            ))
            is_solved = True
            maddpg.save_model(ts_started, scores_total=scores_total, moving_averages=score_rolling_averages)
            last_best_score = score_average

        if i_episode % chkpoint_after == 0 and is_solved:
            # Only save these weights if they are better than the ones previously saved
            if score_average > last_best_score:
                last_best_score = score_average
                maddpg.save_model(ts_started, scores_total=scores_total, moving_averages=score_rolling_averages)

    return scores_total, score_rolling_averages


def plot_results(scores, rolling_score_averages, save_to_fn=None):
    """
    Produce a graph of training results after training agents with the MADDPG algorithm
    :param scores: List of highest scores given across agents for every training episode
    :param rolling_score_averages: List of highest scores given across agents from the last 100 training episodes
    :param save_to_fn: Filename to save figure to (optional) 
    :return: None
    """
    fig = plt.figure()
    _ = fig.add_subplot(111)

    plt.plot(np.arange(1, len(scores) + 1), scores, label="Highest Score")
    plt.plot(np.arange(1, len(rolling_score_averages) + 1), rolling_score_averages,
             label="Rolling Average (100 episodes)")
    # This line indicates the score at which the environment is considered solved
    plt.axhline(y=0.5, color="b", linestyle="-", label="Environment solved")

    plt.legend(bbox_to_anchor=(1., 1), loc=2, borderaxespad=0.)
    plt.ylabel("Score")
    plt.xlabel("Episode #")

    if save_to_fn is not None:
        plt.savefig(fname=save_to_fn)
    else:
        # For use in an interactive Jupyter notebook
        plt.show()
    

def play(maddpg, cnfg, num_games=11, load_model=False):
    """
    Have the agents play a match, preferrably after successfully solving the environment!
    :param maddpg: Wrapper for MADDPG related classes
    :param cnfg: Configuration class that contains hyper-parameters relating to the network architecture and environment
     setup
    :param num_games: How many games should be played
    :param load_model: Whether stored model weights should be loaded
    :return: None
    """
    if load_model:
        maddpg.load_model()

    print("Agent #0: Red racket")
    print("Agent #1: Blue racket")
    print("---------------------")

    game_scores = [0 for _ in range(cnfg.num_agents)]

    # Environment information
    brain_name = cnfg.env.brain_names[0]

    for i in range(1, num_games+1):
        env_info = cnfg.env.reset(train_mode=False)[brain_name]
        states = env_info.vector_observations
        scores = np.zeros(cnfg.num_agents)

        while True:
            actions = maddpg.act(states)

            env_info = cnfg.env.step(actions)[brain_name]
            next_states = cnfg.env_info.vector_observations
            rewards = env_info.rewards
            scores += rewards
            done_signals = cnfg.env_info.local_done

            if np.any(done_signals):
                winner = np.argmax(scores)
                game_scores[winner] += 1
                print("Partial game score: {}".format(game_scores))
                break

            states = next_states

    print("---------------------")
    print("Agent #{} wins!".format(np.argmax(game_scores)))
