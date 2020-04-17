import copy
import argparse
import numpy as np


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def hidden_init(layer):
    """
    Initialises hidden layer weights with an upper and lower limit that is inversely proportional to the square root of
    the layer's size.
    :param layer: (torch.nn) Torch layer
    :return: (tuple) Upper and lower limit
    """
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class OrnsteinUhlenbeckNoise:
    """Initialise the Ornstein Uhlenbeck noise function, drawing samples from a random normal distribution"""
    def __init__(self, size, mu=0.0, theta=0.15, sigma=0.2):
        self.state = None
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = copy.copy(self.mu)

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([np.random.normal(loc=0, scale=1)
                                                                 for _ in range(len(x))])
        self.state = x + dx
        return self.state
