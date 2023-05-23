import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def PlotStreamlineLength(stl_mesh, Ls):
    xx = stl_mesh.x
    yy = stl_mesh.y
    zz = stl_mesh.z
    face = stl_mesh.vectors
    range = max(Ls) - min(Ls)
    cp = (Ls - min(Ls)) / range
    r = cp
    g = np.ones(len(cp)) * 0
    b = -cp + 1

    fcs = np.zeros((len(cp), 3))
    fcs[:, 0] = r
    fcs[:, 1] = g
    fcs[:, 2] = b

    fig  = plt.figure()
    axes = plt.axes(projection='3d')

    axes.add_collection3d(mplot3d.art3d.Poly3DCollection(face, facecolor=fcs))

    scale = stl_mesh.points.flatten()
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