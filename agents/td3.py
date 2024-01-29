from networks.AC import Actor, TD3Critic
from utils.utils import ReplayBuffer, convert_to_tensor

import torch
import torch.nn as nn
import torch.nn.functional
import torch.optim as optim


class TD3(nn.Module):
    def __init__(self, writer, device, state_dim, action_dim, args, noise, max_action, policy_freq=2):
        super(TD3, self).__init__()

        self.noise = noise
        self.args = args
        self.actor = Actor(self.args.layer_num, state_dim, action_dim, self.args.hidden_dim,
                           self.args.activation_function, self.args.last_activation, self.args.trainable_std)

        self.target_actor = Actor(self.args.layer_num, state_dim, action_dim, self.args.hidden_dim,
                                  self.args.activation_function, self.args.last_activation, self.args.trainable_std)

        self.q = TD3Critic(state_dim, action_dim)
        # self.q_2 = Critic(self.args.layer_num, state_dim + action_dim, 1, self.args.hidden_dim,
        #                   self.args.activation_function, self.args.last_activation)

        self.target_q = TD3Critic(state_dim, action_dim)
        # self.target_q_2 = Critic(self.args.layer_num, state_dim + action_dim, 1, self.args.hidden_dim,
        #                          self.args.activation_function, self.args.last_activation)

        self.soft_update(self.q, self.target_q, 1.)
        # self.soft_update(self.q_2, self.target_q_2, 1.)

        self.data = ReplayBuffer(action_prob_exist=False, max_size=int(self.args.memory_size), state_dim=state_dim,
                                 num_action=action_dim)

        self.q_optimizer = optim.Adam(self.q.parameters(), lr=self.args.q_lr)
        # self.q_2_optimizer = optim.Adam(self.q_2.parameters(), lr=self.args.q_lr)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.args.actor_lr)

        self.device = device
        self.writer = writer
        self.policy_freq = policy_freq  # 延迟更新policy网络的轮数
        self.max_action = max_action

    def put_data(self, transition):
        self.data.put_data(transition)

    def get_action(self, x):
        next_action = (self.actor(x)[0] + torch.as_tensor(self.noise.sample()).to(self.device))  # .clamp(-self.max_action, self.max_action)
        return next_action, self.actor(x)[1]

    def soft_update(self, network, target_network, rate):
        for network_params, target_network_params in zip(network.parameters(), target_network.parameters()):
            target_network_params.data.copy_(target_network_params.data * (1.0 - rate) + network_params.data * rate)

    def train_net(self, batch_size, n_epi):

        data = self.data.sample(shuffle=True, batch_size=batch_size)
        states, actions, rewards, next_states, dones = convert_to_tensor(self.device, data['state'], data['action'],
                                                                         data['reward'], data['next_state'],
                                                                         data['done'])

        # q update
        q_1, q_2 = self.target_q(next_states, self.target_actor(next_states)[0])
        target_q = torch.min(q_1, q_2)
        target_q = rewards + self.args.gamma * (1 - dones) * target_q
        current_q_1, current_q_2 = self.q(states, actions)
        q_loss = torch.nn.functional.mse_loss(current_q_1, target_q)
        # q_2_loss = torch.nn.functional.mse_loss(current_q_2, target_q)

        self.q_optimizer.zero_grad()
        # self.q_2_optimizer.zero_grad()
        q_loss.backward()
        # q_2_loss.backward()
        self.q_optimizer.step()
        # self.q_2_optimizer.step()

        # Delayed policy updates
        if n_epi % self.policy_freq == 0:
            # Compute actor lose
            actor_loss = -self.q.forward(states, self.actor(states)[0])[0].mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            self.soft_update(self.q, self.target_q, self.args.soft_update_rate)
            self.soft_update(self.actor, self.target_actor, self.args.soft_update_rate)
            if self.writer is not None:
                self.writer.add_scalar("loss/q", q_loss, n_epi)
                # self.writer.add_scalar("loss/q", q_2_loss, n_epi)
                self.writer.add_scalar("loss/actor", actor_loss, n_epi)

        return actions

# next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)
