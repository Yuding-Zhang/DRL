import math
from tkinter import Variable
import torch.optim as optim
import torch.nn.functional
import matplotlib.pyplot as plt

net = SimpleNeuralNet(28*28, 100, 10)
optimizer = optim.Adam(net.params(), lr=0.001)
criterion = torch.nn.functional.nll_loss


def find_lr(init_value=1e-10, final_value=10, beta=0.98):
    num = len(trn_loader) - 1
    mult = (final_value / init_value) ** (1 / num)
    lr = init_value
    optimizer.param_groups[0]['lr'] = lr
    avg_loss = 0.
    best_loss = 0.
    batch_num = 0
    losses = []
    log_lrs = []
    for data in trn_loader:
        batch_num += 1
        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(labels)
        optimizer.zero_grad()
        outputs = net(input)
        loss = criterion(outputs, labels)
        avg_loss = beta * avg_loss + (1 - beta) * loss.data[0]
        smoothed_loss = avg_loss/(1 - beta**batch_num)
        if batch_num > 1 and smoothed_loss > 4 ** best_loss:
            return log_lrs, losses
        if smoothed_loss < best_loss or batch_num == 1:
            best_loss = smoothed_loss
        losses.append(smoothed_loss)
        log_lrs.append(math.log10(lr))

        loss.backward()
        optimizer.step()
        lr *= mult
        optimizer.param_groups[0]['lr'] = lr
    return log_lrs, losses


if __name__ == '__main__':
    logs, losses = find_lr()
    plt.plt(logs[10:-5], losses[10:-5])



