import torch
import torch.nn as nn
from networks.base import Network
import torch.nn.functional

# A-C结构不变  ??trainable_std??


class Actor(Network):
    # actor网络
    def __init__(self, layer_num, input_dim, output_dim, hidden_dim, activation_function=torch.tanh,
                 last_activation=None, trainable_std=False):
        super(Actor, self).__init__(layer_num, input_dim, output_dim, hidden_dim, activation_function, last_activation)
        self.trainable_std = trainable_std
        if self.trainable_std is True:
            self.logstd = nn.Parameter(torch.zeros(1, output_dim))

    def forward(self, x):
        mu = self._forward(x)
        if self.trainable_std is True:
            std = torch.exp(self.logstd)
        else:
            logstd = torch.zeros_like(mu)
            std = torch.exp(logstd)
        return mu, std


class Critic(Network):
    def __init__(self, layer_num, input_dim, output_dim, hidden_dim, activation_function, last_activation=None):
        super(Critic, self).__init__(layer_num, input_dim, output_dim, hidden_dim, activation_function, last_activation)

    def forward(self, *x):  # 获取所有的x
        x = torch.cat(x, -1)
        # example：
        # >>> torch.cat((x, x), -1)
        # tensor([[-0.8168, -1.9389, 0.0781, -0.8168, -1.9389, 0.0781],
        #         [0.3570, -1.3199, -0.1600, 0.3570, -1.3199, -0.1600]])
        return self._forward(x)


class TD3Critic(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TD3Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(input_dim + output_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(input_dim + output_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = torch.nn.functional.relu(self.l1(sa))
        q1 = torch.nn.functional.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = torch.nn.functional.relu(self.l4(sa))
        q2 = torch.nn.functional.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

