# import copy
# import random
import numpy as np
import torch


# Ornstein-Uhlenbeck process
class OUNoise:
    def __init__(self, size, mu=0.0, sigma=0.1, theta=0.15, init_process=True):  # DDPG论文使用的参数mu=0.0,sigma=1,theta=0.5
        """Initialize parameters and noise process."""
        self.action_size = size
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.init_process = init_process
        self.state = np.ones(self.action_size) * self.mu
        self.reset()

    def reset(self):
        if self.init_process is not None:
            self.state = self.init_process
        else:
            self.state = np.zeros_like(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_size)
        self.state = x + dx
        return self.state


class ClipNoise:
    # noise = (torch.randn_like(actions) * 0.2 * max_action).clamp(-0.5 * max_action, 0.5 * max_action)
    def __init__(self, size, max_action, actions, policy_noise=0.2, clip_noise=0.5):
        """Initialize parameters and noise """
        self.action_size = size
        self.policy_noise = policy_noise
        self.clip_noise = clip_noise
        self.max_action = max_action
        self.actions = actions

    def sample(self):
        noise = (torch.randn_like(self.actions) * 0.2 * self.max_action).clamp(-0.5 * self.max_action, 0.5 * self.max_action)
        return noise


if __name__ == '__main__':
    ou = OUNoise(1)
    states = []
    for i in range(1000):
        states.append(ou.sample())
    import matplotlib.pyplot as plt

    plt.plot(states)
    plt.show()

# class OUNoise:
#     """Ornstein-Uhlenbeck process."""
#
#     def __init__(self, size, seed, mu=0.0, theta=0.1, sigma=.1, sigma_min=0.05, sigma_decay=.99):
#         """Initialize parameters and noise process."""
#         self.mu = mu * np.ones(size)
#         self.theta = theta
#         self.sigma = sigma
#         self.sigma_min = sigma_min
#         self.sigma_decay = sigma_decay
#         self.seed = random.seed(seed)
#         self.size = size
#         self.reset()
#
#     def reset(self):
#         """Reset the internal state (= noise) to mean (mu)."""
#         self.state = copy.copy(self.mu)
#         self.sigma = max(self.sigma_min, self.sigma * self.sigma_decay)
#
#     def sample(self):
#         """Update internal state and return it as a noise sample."""
#         x = self.state
#         dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
#         self.state = x + dx
#         return self.state
