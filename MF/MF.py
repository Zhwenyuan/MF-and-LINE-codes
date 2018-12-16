import numpy as np
import re


def matrix_factorization(R, P, Q, k=10, steps=500, alpha=0.005, beta=0.02):

    rows, cols = R.nonzero()
    p = np.random.permutation(len(rows))  # randomly sort
    rows, cols = rows[p[:100]], cols[p[:100]]

    for step in range(steps):
        print("iteration is %d now" % step)
        for i in rows:
            for j in cols:
                e_ij = R[i][j] - np.dot(P[i, :], Q[:, j])
                for _k in range(k):
                    P[i][_k] += alpha * (2*e_ij*Q[_k][j] - beta*P[i][_k])
                    Q[_k][j] += alpha * (2*e_ij*P[i][_k] - beta*Q[_k][j])

        # Error_arr = ((R - np.dot(P, Q))**2).sum() + beta/2*((P**2).sum() + (Q**2))
        # if error < 0.001:
        #     break

    print("matrix factorization completed.\n")
    print("calculating Error...")
    m_ratings = np.dot(P, Q)
    sum = 0
    for row, col in zip(rows, cols):
        sum += np.square(m_ratings[row, col] - R[row, col])
    print("now Error is {}".format(sum))
    return P, Q


if __name__ == '__main__':
    # test coding:
    # np.set_printoptions(suppress=True)
    # arr = np.array([[5, 3, 0, 1],
    #                 [4, 0, 0, 1],
    #                 [1, 1, 0, 5],
    #                 [1, 0, 0, 4],
    #                 [0, 1, 5, 4]])
    # P = np.random.rand(5, 10)
    # Q = np.random.rand(10, 4)
    # P, Q = matrix_factorization(arr, P, Q, 10)
    # print(np.dot(P, Q))

    print("initializing ratings...")
    P = np.random.rand(6040, 10)
    Q = np.random.rand(10, 3952)
    ratings = np.zeros([6040, 3952])

    f = open(".\\ratings.dat", 'r')
    strlist = re.split(r"::|\n", f.read())
    for i in range(strlist.__len__() - 1):
        if i % 4 == 0:
            ratings[int(strlist[i])-1, int(strlist[i+1])-1] = int(strlist[i+2])
    print("ratings initializing completed.\n")

    print("utilizing matrix factorization...\nlearning rate = 1e-3, latent factors = 10\n")
    P, Q = matrix_factorization(ratings, P, Q, k=10, steps=300)

