import numpy as np

import utility
cartmesh = utility.cartmesh
splot = utility.splot

def function2(xyz, t):

    if isinstance(t, float) or isinstance(t, int):
        return_ = ((xyz[:, 0] * xyz[:, 1] * xyz[:, 2]) * 9) * np.cos(t * 2 * np.pi)

    else:
        part1 = (xyz[:, 0] * xyz[:, 1] * xyz[:, 2]) * 9
        part1 = part1.reshape(part1.shape[0], 1)
        part2 = np.cos(t * 2 * np.pi).reshape(1, t.shape[0])
        return_ = part1 * part2
    return return_

def function(xyz, t):

    if isinstance(t, float) or isinstance(t, int):
        return_ = ((xyz[:, 0] * xyz[:, 1] * xyz[:, 2]) * 9) * np.cos(t * 2 * np.pi) + \
                  ((xyz[:, 0] * xyz[:, 1] + xyz[:, 2] * 0.2) * (1 + 2 * t))


    else:
        part1 = (xyz[:, 0] * xyz[:, 1] * xyz[:, 2]) * 9
        part1 = part1.reshape(part1.shape[0], 1)
        part2 = np.cos(t * 2 * np.pi).reshape(1, t.shape[0])
        return_ = part1 * part2
        part1 = (xyz[:, 0] * xyz[:, 1] + xyz[:, 2] * 0.2).reshape(part1.shape[0], 1)
        part2 = (1 + 2 * t).reshape(1, t.shape[0])
        return_ = return_ + part1 * part2

    return return_

def plot_function(t):
    N = 100
    theta, phi = np.linspace(0, 2 * np.pi, N), np.linspace(0, np.pi, N)
    xyz = cartmesh(theta, phi)
    XYZ = np.vstack((xyz[0,].reshape(-1), xyz[1,].reshape(-1), xyz[2,].reshape(-1))).transpose()
    c = function(XYZ, t)
    splot(xyz, c.reshape(N, N))