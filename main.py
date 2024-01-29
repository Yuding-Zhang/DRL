from configparser import ConfigParser
from argparse import ArgumentParser
from utils.noise import OUNoise, ClipNoise

import torch
import gym
import numpy as np
import os
import pickle

from agents.td7 import TD7
from agents.td3 import TD3
from agents.ddpg import DDPG
from agents.ppo import PPO
from agents.sac import SAC

from utils.utils import make_transition, Dict, RunningMeanStd

os.makedirs('./model_weights', exist_ok=True)

##############
parser = ArgumentParser('parameters')

parser.add_argument("--env_name", type=str, default='BipedalWalker-v3', help="'Ant-v2','HalfCheetah-v2','Hopper-v2','Humanoid-v2','HumanoidStandup-v2',\
          'InvertedDoublePendulum-v2', 'InvertedPendulum-v2' (default : Hopper-v2)")
parser.add_argument("--algo", type=str, default='ppo', help='algorithm to adjust (default : ddpg)')
parser.add_argument('--train', type=bool, default=True, help="(default: True)")
parser.add_argument('--render', type=bool, default=False, help="(default: False)")
parser.add_argument('--epochs', type=int, default=5000, help='number of epochs, (default: 1000)')
parser.add_argument('--tensorboard', type=bool, default=False, help='use_tensorboard, (default: False)')
parser.add_argument("--load", type=str, default='no', help='load network name in ./model_weights')
parser.add_argument("--save_interval", type=int, default=1000, help='save interval(default: 100)')
parser.add_argument("--print_interval", type=int, default=20, help='print interval(default : 20)')
parser.add_argument("--use_cuda", type=bool, default=False, help='cuda usage(default : True)')
parser.add_argument("--reward_scaling", type=float, default=0.1, help='reward scaling(default : 0.1)')
parser.add_argument("--draw_picture", type=float, default=True, help='draw reward data picture (default : True)')
args = parser.parse_args()
parser = ConfigParser()
parser.read('config.ini')
agent_args = Dict(parser, args.algo)
with open('./args.pkl', 'wb') as f:
    pickle.dump(vars(args), f)
##############

##############
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if not args.use_cuda:
    device = 'cpu'
##############

##############
if args.tensorboard:
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter()
else:
    writer = None
##############

env = gym.make(args.env_name)
# env.seed(0)
# torch.manual_seed(0)
action_dim = env.action_space.shape[0]
state_dim = env.observation_space.shape[0]
state_rms = RunningMeanStd(state_dim)
max_action = float(env.action_space.high[0])

score_lst = []
state_lst = []
score_lst_mean = []
score_lst_std = []
episode_lst = []

############################################################
if args.algo == 'td7':
    noise = OUNoise(action_dim, 0)
    agent = TD7(writer, device, state_dim, action_dim, agent_args, noise)
elif args.algo == 'td3':
    # noise = ClipNoise(action_dim, max_action, actions=torch.randn(1, action_dim))
    noise = OUNoise(action_dim, 0)
    agent = TD3(writer, device, state_dim, action_dim, agent_args, noise, max_action)
elif args.algo == 'ddpg':
    noise = OUNoise(action_dim, 0)
    agent = DDPG(writer, device, state_dim, action_dim, agent_args, noise)
elif args.algo == 'ppo':
    agent = PPO(writer, device, state_dim, action_dim, agent_args)
elif args.algo == 'sac':
    agent = SAC(writer, device, state_dim, action_dim, agent_args)
################################################################

if (torch.cuda.is_available()) and args.use_cuda:
    agent = agent.cuda()
if args.load != 'no':
    agent.load_state_dict(torch.load("./model_weights/{}/".format(args.algo) + args.load))


###############################################################
if agent_args.on_policy:
    score = 0.0
    state_ = (env.reset())
    state = np.clip((state_ - state_rms.mean) / (state_rms.var ** 0.5 + 1e-8), -5, 5)
    for n_epi in range(args.epochs+1):
        for t in range(agent_args.traj_length):
            if args.render:
                env.render()
            state_lst.append(state_)
            mu, sigma = agent.get_action(torch.from_numpy(state).float().to(device))
            dist = torch.distributions.Normal(mu, sigma[0])
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(-1, keepdim=True)
            next_state_, reward, done, info = env.step(action.cpu().numpy())
            next_state = np.clip((next_state_ - state_rms.mean) / (state_rms.var ** 0.5 + 1e-8), -5, 5)
            transition = make_transition(state,
                                         action.cpu().numpy(),
                                         np.array([reward * args.reward_scaling]),
                                         next_state,
                                         np.array([done]),
                                         log_prob.detach().cpu().numpy())
            agent.put_data(transition)
            score += reward
            if done:
                state_ = (env.reset())
                state = np.clip((state_ - state_rms.mean) / (state_rms.var ** 0.5 + 1e-8), -5, 5)
                score_lst.append(score)
                if args.tensorboard:
                    writer.add_scalar("score/score", score, n_epi)
                score = 0
            else:
                state = next_state
                state_ = next_state_

        agent.train_net(n_epi)
        state_rms.update(np.vstack(state_lst))
        if n_epi % args.print_interval == 0 and n_epi != 0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, sum(score_lst) / len(score_lst)))
            score_lst_mean.append(np.mean(score_lst))
            score_lst_std.append(np.std(score_lst))
            episode_lst.append(n_epi)
            score_lst = []
        if n_epi % args.save_interval == 0 and n_epi != 0:
            torch.save(agent.state_dict(), './model_weights/{}/{}_agent_'.format(args.algo, args.algo) + str(n_epi) + '.pt')

else:  # off policy
    for n_epi in range(args.epochs+1):
        score = 0.0
        state = env.reset()
        done = False
        while not done:
            if args.render:
                env.render()
            action, _ = agent.get_action(torch.from_numpy(state).float().to(device))
            action = action.cpu().detach().numpy()
            next_state, reward, done, info = env.step(action)
            transition = make_transition(state,
                                         action,
                                         np.array([reward * args.reward_scaling]),
                                         next_state,
                                         np.array([done]))
            agent.put_data(transition)

            state = next_state

            score += reward
            if agent.data.data_idx > agent_args.learn_start_size:
                actions = agent.train_net(agent_args.batch_size, n_epi)
                # if args.algo == 'td3':  # 平滑正则化
                #     noise = ClipNoise(action_dim, max_action, actions)
        score_lst.append(score)
        if args.tensorboard:
            writer.add_scalar("score/score", score, n_epi)
        if n_epi % args.print_interval == 0 and n_epi != 0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, sum(score_lst) / len(score_lst)))
            score_lst_mean.append(np.mean(score_lst))
            score_lst_std.append(np.std(score_lst))
            episode_lst.append(n_epi)
            score_lst = []
        if n_epi % args.save_interval == 0 and n_epi != 0:
            torch.save(agent.state_dict(), './model_weights/{}/{}_agent_'.format(args.algo, args.algo) + str(n_epi) + '.pt')

###############################################################

if args.draw_picture:
    d = {"mean": score_lst_mean, "std": score_lst_std, "episode": episode_lst}
    with open('./RewardResults/{}_results.pkl'.format(args.algo), "wb") as f:
        pickle.dump(d, f, pickle.HIGHEST_PROTOCOL)



