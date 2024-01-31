from networks.AC import Actor, Critic
from utils.utils import ReplayBuffer, convert_to_tensor

import torch
import torch.nn as nn
import torch.nn.functional
import torch.optim as optim


class TD7(nn.Module):
    def __init__(self, writer, device, state_dim, action_dim, args, noise):
        super(TD7, self).__init__()

