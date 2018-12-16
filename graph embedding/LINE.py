import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


class LINE(nn.Module):
    def __init__(self, n_fans, n_shopkeepers, n_factors=20):
        super(LINE, self).__init__()
        self.fan_factors = nn.Embedding(n_fans, n_factors)
        self.shopkeeper_factors = nn.Embedding(n_shopkeepers, n_factors)

    def forward(self, n_fans, n_shopkeepers):
        sigmoid_func = nn.Sigmoid()
        return sigmoid_func(self.fan_factors(torch.tensor(range(n_fans))).mm(self.shopkeeper_factors(torch.tensor(range(n_shopkeepers))).t()))


# class MyLoss(nn.Module):
#     def __init__(self):
#         super(MyLoss, self).__init__()
#
#     def forward(self, P, weight=1):
#         return -(weight * torch.log(P)).sum()


if __name__ == '__main__':
    arr = np.array([[0, 0, 0, 1, 0],
                    [1, 0, 0, 0, 1],
                    [0, 1, 0, 0, 0],
                    [1, 0, 0, 1, 0]])
    rows, cols = arr.shape

    model = LINE(4, 5, n_factors=10)
    weights = torch.tensor(arr, dtype=torch.float)
    loss_func = nn.BCELoss(weight=weights, reduce=True, size_average=False)
    optimizer = torch.optim.ASGD(model.parameters())
    targets = torch.tensor(arr, dtype=torch.float)

    for i in range(1000):

        optimizer.zero_grad()
        output = model(rows, cols)
        loss = loss_func(output, targets)
        loss.backward()
        optimizer.step()

    result = model(rows, cols)
    print(result)
