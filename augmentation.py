import numpy as np


def rotate4(RGBarrays_list):
    # take a list, return a list, this design point is to be discussed, we could use numpy arrays instead, optimize memory usage
    all = []
    for RGBarrays in RGBarrays_list:
        ret = [RGBarrays.copy(), RGBarrays.copy(), RGBarrays.copy(), RGBarrays.copy()]
        ret[1] = np.rot90(ret[0])
        ret[2] = np.rot90(ret[1])
        ret[3] = np.rot90(ret[2])
        all.append(ret[0])
        all.append(ret[1])
        all.append(ret[2])
        all.append(ret[3])
    return all