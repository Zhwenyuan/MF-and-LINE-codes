import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import datetime
import pre_process


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

    relationmap, arr1, arr2 = pre_process.getMatrixFromFile()
    print("relationmap initializing completed.\n")

    print("utilizing graph embedding...\nlearning rate = 1e-3, latent factors = 10\n")
    # select nonzero rows and cols
    rows, cols = relationmap.nonzero()
    p = np.random.permutation(len(rows))
    rows, cols = rows[p[:10000]], cols[p[:10000]]

    n_fans, n_shopkeepers = relationmap.shape
    model = LINE(n_fans, n_shopkeepers, n_factors=10)
    weights = torch.sparse.FloatTensor(torch.tensor(rows), torch.tensor(cols))
    loss_func = nn.BCELoss(weight=weights, reduce=True, size_average=False)
    optimizer = torch.optim.ASGD(model.parameters(), lr=1e-3)
    targets = torch.tensor(relationmap, dtype=torch.float)


    for i in range(500):
        print("iteration is %d now." % i)

        optimizer.zero_grad()
        output = model(rows, cols)
        loss = loss_func(output, targets)
        loss.backward()
        optimizer.step()

    print("graph embedding completed.\n")

    print("calculating Error...")

    result = model(rows, cols)
    sum = 0
    for row, col in zip(rows, cols):
        sum += np.square(result[row, col] - relationmap[row, col])
    print("now Error is {}".format(sum))

    curtime = datetime.datetime.now()
    print("%d: %d: %d\n" % (curtime.hour, curtime.minute, curtime.second))


if __name__ == '__main__':
    train()
