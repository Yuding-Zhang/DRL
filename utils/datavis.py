import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns  # 导入模块
import pickle
import torch
import shutil
import argparse
# https://blog.csdn.net/qq_47997583/article/details/124045574
# https://blog.csdn.net/qq_52661119/article/details/119223499
sns.set()  # 设置美化参数，一般默认就好

# file_list = [f for f in os.listdir() if f.endswith('.pt')]
# with open('all.pt', "wb") as f:
#     for file_name in file_list:
#         with open(file_name, 'rb') as file:
#             shutil.copyfileobj(file, f)
# with open('all.pt', "rb") as f:
#     data = torch.load(f)


with open('../args.pkl', 'rb') as f:
    args_dict = pickle.load(f)
args = argparse.Namespace(**args_dict)

file = '{}_results.pkl'.format(args.algo)
with open('../RewardResults/{}'.format(file), "rb") as f:
    data = pickle.load(f)

# with open('F:/DRL/RewardResults/TD3_HalfCheetah-v2_0.npy', "rb") as f:
#     data = np.load(f)
#     print(data)


def smooth(reward_data, sm=1):
    if sm > 1:
        smooth_data = []
        for d in reward_data['mean']:
            y = np.ones(sm) * 1.0 / sm
            d = np.convolve(y, d, "same")
            smooth_data.append(d[0])
    return smooth_data


data['mean'] = smooth(data, sm=2)

sns.set(style="darkgrid", font_scale=1.5)
sns.lineplot(x='episode', y='mean', data=data)
# sns.tsplot(time=time, data=x2, color="b", condition="dagger")

# file = "dagger_" + ENV_NAME + ".pkl"
# with open(os.path.join("test_data", file), "rb") as f:
#     data = pickle.load(f)
#
# x2 = data["mean"]
# x2 = smooth(x2, sm=2)


# sns.lineplot(x=range(len(rewards)), y=rewards)
# sns.relplot(x=range(len(rewards)),y=rewards,kind="line") # 与上面一行等价

if __name__ == '__main__':
    plt.ylabel("Reward")
    plt.xlabel("Episode Number")
    plt.title("{} learning in {}".format(args.algo, args.env_name))
    plt.savefig('../assets/{}/{} learning in {}.png'.format(args.algo, args.algo, args.env_name))
    plt.show()
