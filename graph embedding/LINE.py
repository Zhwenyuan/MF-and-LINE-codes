import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import datetime
from pre_process import getMatrixFromFile


class LINE(nn.Module):
    def __init__(self, n_fans, n_shopkeepers, n_factors=20):
        super(LINE, self).__init__()
        self.fan_factors = nn.Embedding(n_fans, n_factors)
        self.shopkeeper_factors = nn.Embedding(n_shopkeepers, n_factors)

    def forward(self, n_fans, n_shopkeepers):
        sigmoid_func = nn.Sigmoid()
        return sigmoid_func(self.fan_factors(torch.tensor(range(n_fans))).mm(self.shopkeeper_factors(torch.tensor(range(n_shopkeepers))).t()))


def train():

    curtime = datetime.datetime.now()
    print("%d: %d: %d\n" % (curtime.hour, curtime.minute, curtime.second))
    print("initializing relationmap...")

    relationmap, fans_dict, shopkeepers_dict = getMatrixFromFile()
    print("relationmap initializing completed.\n")

    print("initializing model and loss function...")
    # select nonzero rows and cols
    rows, cols = relationmap.nonzero()
    n_fans, n_shopkeepers = relationmap.shape

    # initialize model, weights, loss function, optimizer, targets
    model = LINE(n_fans, n_shopkeepers, n_factors=10)
    weights = torch.sparse.FloatTensor(torch.LongTensor([rows, cols]), torch.ones(rows.__len__()), (n_fans, n_shopkeepers)).to_dense()
    loss_func = nn.BCELoss(weight=weights, reduce=True, size_average=False)
    optimizer = torch.optim.ASGD(model.parameters(), lr=1e-3)
    targets = torch.sparse.FloatTensor(torch.LongTensor([rows, cols]), torch.ones(rows.__len__()), (n_fans, n_shopkeepers)).to_dense()
    print("initializing completed.\n")

    print("utilizing graph embedding...\nlearning rate = 1e-3, latent factors = 10\n")
    for i in range(5000):
        print("iteration is %d now." % i)
        optimizer.zero_grad()       # clear grad
        output = model(n_fans, n_shopkeepers)       # calculate forward
        loss = loss_func(output, targets)           # calculate loss function
        loss.backward()         # calculate backward
        optimizer.step()        # one step

    print("graph embedding completed.\n")

    print("calculating Error...")

    result = model(n_fans, n_shopkeepers)
    sum = 0
    for row, col in zip(rows, cols):
        sum += np.square(result[row, col].detach().numpy() - relationmap[row, col])
    print("now Error is {}".format(sum))

    curtime = datetime.datetime.now()
    print("%d: %d: %d\n" % (curtime.hour, curtime.minute, curtime.second))


if __name__ == '__main__':
    train()
