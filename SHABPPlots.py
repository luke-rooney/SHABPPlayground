import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d


def PlotPressureMap(cp, mesh):
    xx = mesh.x
    yy = mesh.y
    zz = mesh.z
    face = mesh.vectors
    range = max(cp) - min(cp)
    cp = (cp - min(cp)) / range
    r = cp
    g = np.ones(len(cp)) * 0
    b = -cp + 1

    fcs = np.zeros((len(cp), 3))
    fcs[:, 0] = r
    fcs[:, 1] = g
    fcs[:, 2] = b

    fig = plt.figure()
    axes = plt.axes(projection='3d')

    axes.add_collection3d(mplot3d.art3d.Poly3DCollection(face, facecolor=fcs))

    scale = mesh.points.flatten()
    axes.auto_scale_xyz(scale, scale, scale)

    rx = (np.max(xx) - np.min(xx))
    ry = (np.max(yy) - np.min(yy))
    rz = (np.max(zz) - np.min(zz))

    cx = (np.max(xx) + np.min(xx)) / 2
    cy = (np.max(yy) + np.min(yy)) / 2
    cz = (np.max(zz) + np.min(zz)) / 2

    r = np.max([rx, ry, rz])

    axes.set_xbound(cx - r / 1.5, cx + r / 1.5)
    axes.set_ybound(cy - r / 1.5, cy + r / 1.5)
    axes.set_zbound(cz - r / 1.5, cz + r / 1.5)

    plt.show()

def PlotXY(x, y, xlabel, ylabel, legend_text, title):

    if len(x.shape) == 1:
        for i in range(y.shape[0]):
            plt.plot(x, y[i])
    else:
        for i in range(x.shape[0]):
            plt.plot(x[i], y[i])

    plt.legend(legend_text)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid()
    plt.show()