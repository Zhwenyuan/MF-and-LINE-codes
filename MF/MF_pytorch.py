import numpy as np
import torch
import re
import datetime


class MatrixFactorization(torch.nn.Module):
    def __init__(self, n_users, n_items, n_factors=20):
        super().__init__()
        self.user_factors = torch.nn.Embedding(n_users, n_factors, sparse=True)
        self.item_factors = torch.nn.Embedding(n_items, n_factors, sparse=True)

    def forward(self, user, item):
        return (self.user_factors(user) * self.item_factors(item)).sum()


if __name__ == '__main__':

    curtime = datetime.datetime.now()
    print("%d: %d: %d\n" % (curtime.hour, curtime.minute, curtime.second))
    print("initializing ratings...")
    n_users = 6040
    n_items = 3952
    ratings = np.zeros([6040, 3952])

    f = open(".\\ratings.dat", 'r')
    strlist = re.split(r"::|\n", f.read())
    for i in range(strlist.__len__() - 1):
        if i % 4 == 0:
            ratings[int(strlist[i])-1, int(strlist[i+1])-1] = int(strlist[i+2])
    print("ratings initializing completed.\n")

    print("utilizing matrix factorization...\nlearning rate = 1e-3, latent factors = 10\n")
    model = MatrixFactorization(n_users, n_items, n_factors=10)
    loss_func = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    rows, cols = ratings.nonzero()  # get positions of nonzero elements
    p = np.random.permutation(len(rows))    # randomly sort
    rows, cols = rows[p[:10000]], cols[p[:10000]]

    for i in range(1000):
        print("iteration is %d now." % i)
        for row, col in zip(rows, cols):
            # print("\trow is %d, col is %d now." % (row, col))
            rating = torch.tensor([ratings[row, col]], dtype=torch.float)
            row = torch.tensor([np.long(row)], dtype=torch.long)
            col = torch.tensor([np.long(col)], dtype=torch.long)

            optimizer.zero_grad()
            prediction = model(row, col)    # calculate forward
            loss = loss_func(prediction, rating)

            loss.backward()     # calculate backward

            optimizer.step()    # one step

    print("matrix factorization completed.\n")

    print("calculating Error...")
    user_tensor = model.user_factors(torch.tensor(range(n_users)))
    item_tensor = model.item_factors(torch.tensor(range(n_items)))
    m_ratings = user_tensor.matmul(item_tensor.t())
    m_ratings = m_ratings.detach().numpy()
    sum = 0
    for row, col in zip(rows, cols):
        sum += np.square(m_ratings[row, col] - ratings[row, col])
    print("now Error is {}".format(sum))

    curtime = datetime.datetime.now()
    print("%d: %d: %d\n" % (curtime.hour, curtime.minute, curtime.second))
