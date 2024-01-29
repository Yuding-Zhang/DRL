# from abc import *
import torch
import torch.nn as nn


# 抽象化基本网络，后续使用需要继承


# class BaseNetwork(nn.Module, metaclass=ABCMeta):  # nn.Module is Base class for all neural network modules
#     @abstractmethod
#     def __init__(self):
#         super(BaseNetwork, self).__init__()
#
#     @abstractmethod
#     def forward(self, x):
#         return x


# 无论是什么方法都需要的网络结构，在这个文件中统一编写


class Network(nn.Module):
    def __init__(self, layer_num, input_dim, output_dim, hidden_dim, activation_function=torch.relu,
                 last_activation=None):
        super(Network, self).__init__()
        # 在 Pytorch 中继承 torch.nn.Module 后,
        # 要执行self(自己, self).__init__( ) 才能通过调用实例化的对象的方法调用 forward 函数
        self.activation = activation_function
        self.last_activation = last_activation
        layers_unit = [input_dim] + [hidden_dim] * (layer_num - 1)
        layers = ([nn.Linear(layers_unit[idx], layers_unit[idx + 1]) for idx in range(len(layers_unit) - 1)])
        # Examples::
        # >> > m = nn.Linear(20, 30)
        # >> > input = torch.randn(128, 20)
        # >> > output = m(input)
        # >> > print(output.size())
        # torch.Size([128, 30])
        self.layers = nn.ModuleList(layers)
        self.last_layer = nn.Linear(layers_unit[-1], output_dim)
        self.network_init()

    def forward(self, x):
        return self._forward(x)

    def _forward(self, x):
        for layer in self.layers:
            x = self.activation(layer(x))  # 在每一层之间添加relu激活层
        x = self.last_layer(x)
        if self.last_activation is not None:
            x = self.last_activation(x)
        return x

    def network_init(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):  # 对线性层初始化
                nn.init.orthogonal_(layer.weight)  # 正交初始化
                layer.bias.data.zero_()
