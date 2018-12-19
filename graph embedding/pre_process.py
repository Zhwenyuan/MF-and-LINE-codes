import numpy as np

def getMatrixFromFile():
    f = open('.\\beidian_shopkeepers_and_fans_filted_1_2_zyx.txt', 'r')
    textStr = f.read()
    f.close()
    lines = textStr.split('\n')

    # select 1000*1000 from dataset
    # lines = lines[:1000]

    fans = set()
    shopkeepers = set()
    for line in lines:
        strlist = line.split('\t')
        if strlist.__len__() != 2:
            break
        fans.add(strlist[1])
        shopkeepers.add((strlist[0]))

    fans = list(fans)
    shopkeepers = list(shopkeepers)

    # each key is uid or pid, each value is uid's index in matrix
    fans_dict = dict()
    shopkeepers_dict = dict()
    for i in range(fans.__len__()):
        fans_dict.update({fans[i]: i})
    for j in range(shopkeepers.__len__()):
        shopkeepers_dict.update({shopkeepers[j]: j})

    # write relationmap to file
    file = open("relationMap.txt", 'w')
    for l in lines:
        strlist = l.split('\t')
        if strlist.__len__() != 2:
            break
        file.write(str(shopkeepers_dict[strlist[0]]))
        file.write("::")
        file.write(str(fans_dict[strlist[1]]))
        file.write("::1\n")
    file.close()

    relationmap = np.zeros([fans.__len__(), shopkeepers.__len__()])
    count = 0
    for _line in lines:
        strlist = _line.split('\t')
        if strlist.__len__() != 2:
            break
        relationmap[fans_dict[strlist[1]], shopkeepers_dict[strlist[0]]] = 1



    return relationmap, fans_dict, shopkeepers_dict

if __name__ == '__main__':
    getMatrixFromFile()