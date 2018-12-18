import numpy as np


def getMatrixFromFile():
    f = open('.\\beidian_shopkeepers_and_fans_filted_1_2_zyx.txt', 'r')
    fans = set()
    shopkeeperes = set()
    for line in f:
        strlist = line.split('\t')
        fans.add(strlist[1])
        shopkeeperes.add((strlist[0]))

    fans = list(fans)
    shopkeeperes = list(shopkeeperes)
    relationmap = np.zeros([fans.__len__(), shopkeeperes.__len__()], dtype=np.uint8)

    count = 0
    for line in f:
        count += 1
        if(count%1000==0):
            print("loop:{}".format(count))
        strlist = line.split('\t')
        if strlist[0] in shopkeeperes and strlist[1] in fans:
            relationmap[fans.index(strlist[1]), shopkeeperes.index(strlist[0])] = 1

    return relationmap, fans, shopkeeperes
