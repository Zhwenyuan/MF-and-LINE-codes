import re
import numpy as np
import torch
from torch import nn, functional
from torch.autograd import Variable

f = open('beidian_shopkeepers_and_fans_filted_1_2_zyx.txt','r')
fdata = f.read()
re.sub('\s','\t',fdata)
spliteddata = fdata.split('\n')
spliteddata.pop() #最后一个是个空格
shopkeepers = []
users = []
for d in spliteddata:
    couple = d.split()
    if not couple[0] in shopkeepers:
        shopkeepers.append(couple[0])
    if not couple[1] in users:
        users.append(couple[1])
relationmap = np.zeros((len(shopkeepers),len(users)))
for i in range(len(shopkeepers)):
    for j in range(len(users)):
        couple = shopkeepers[i]+'\t'+users[j]
        if couple in spliteddata:
            relationmap[i][j] = 1