import numpy
import numpy as np
import Vehicle
from mpl_toolkits import mplot3d
from matplotlib import pyplot
import SHABPy

def CLRangeTest(vehicle, low, high):
    low  = low/180*np.pi
    high = high/180*np.pi
    steps = 100
    aoarange = np.arange(0, steps)*(high-low)/steps + low
    methods  = [1, 2, 3, 5, 6]


    pyplot.figure(1)
    for i in methods:
        vehicle.UpdatePanelMethod(i)
        CL = np.zeros(steps)
        count = 0
        for j in aoarange:
            [cp, cx, cy, cz, cmx, cmy, cmz, cl, cd, cyPrime] = SHABPy.RunSHABPy(j, 0, vehicle)
            CL[count] = cl
            count += 1
        pyplot.plot(aoarange*180/np.pi, CL)

    pyplot.legend(['Newtonian', 'Newtonian Prandtl-Meyer', 'Modified Newtonian', 'Van Dyke', 'Busemann'])
    pyplot.xlabel('Angle of Attack (Degrees)')
    pyplot.ylabel('CL')
    pyplot.grid('on')
    pyplot.show()

def CmxRangeTest(vehicle, low, high):
    low  = low/180*np.pi
    high = high/180*np.pi
    steps = 100
    aoarange = np.arange(0, steps)*(high-low)/steps + low
    methods  = [1, 2, 3, 5, 6]


    pyplot.figure(2)
    for i in methods:
        vehicle.UpdatePanelMethod(i)
        Cmx = np.zeros(steps)
        count = 0
        for j in aoarange:
            [cp, cx, cy, cz, cmx, cmy, cmz, cl, cd, cyPrime] = SHABPy.RunSHABPy(j, 0, vehicle)
            Cmx[count] = cmx
            count += 1
        pyplot.plot(aoarange*180/np.pi, Cmx)

    pyplot.legend(['Newtonian', 'Newtonian Prandtl-Meyer', 'Modified Newtonian', 'Van Dyke', 'Busemann'])
    pyplot.xlabel('Angle of Attack (Degrees)')
    pyplot.ylabel('Cmx')
    pyplot.grid('on')
    pyplot.show()

def Plot3D(vehicle):
    X = vehicle.mesh.x
    Y = vehicle.mesh.y
    Z = vehicle.mesh.z

    fig, axs = pyplot.subplots(2,2)
    fig.suptitle('Vehicle VIEWS')
    axs[0, 0].plot(X,Z)
    axs[0, 1].plot(Y,Z)
    axs[1, 0].plot(X,Y)

    pyplot.show()

def PlotCxCyCz(vehicle, low, high):

    low  = low/180*np.pi
    high = high/180*np.pi
    steps = 100
    aoarange = np.arange(0, steps)*(high-low)/steps + low

    Cx = np.zeros(steps)
    Cy = np.zeros(steps)
    Cz = np.zeros(steps)
    count = 0
    for j in aoarange:
        [cp, cx, cy, cz, cmx, cmy, cmz, cl, cd, cyPrime] = SHABPy.RunSHABPy(j, 0, vehicle)
        Cx[count] = cx
        Cy[count] = cy
        Cz[count] = cz
        count += 1


    fig, axs = pyplot.subplots(2,2)
    fig.suptitle('Cx, Cy, Cz')
    axs[0, 0].plot(aoarange * 180 / np.pi, Cx)
    axs[0, 1].plot(aoarange * 180 / np.pi, Cy)
    axs[1, 0].plot(aoarange * 180 / np.pi, Cz)

    pyplot.show()

def PlotPressureMap(vehicle, cp):
    xx = vehicle.mesh.x
    yy = vehicle.mesh.y
    zz = vehicle.mesh.z
    face = vehicle.mesh.vectors
    range = max(cp) - min(cp)
    cp = (cp - min(cp))/range
    r  = cp
    g  = np.ones(len(cp))*0
    b  = -cp + 1

    fcs = np.zeros((len(cp), 3))
    fcs[:, 0] = r
    fcs[:, 1] = g
    fcs[:, 2] = b

    plt3d = pyplot.figure()
    ax    = mplot3d.Axes3D(plt3d)
    ax.add_collection3d(mplot3d.art3d.Poly3DCollection(face, facecolor=fcs))

    scale = vehicle.mesh.points.flatten()
    ax.auto_scale_xyz(scale, scale, scale)

    rx = (np.max(xx) - np.min(xx))
    ry = (np.max(yy) - np.min(yy))
    rz = (np.max(zz) - np.min(zz))

    cx = (np.max(xx) + np.min(xx))/2
    cy = (np.max(yy) + np.min(yy))/2
    cz = (np.max(zz) + np.min(zz))/2

    r  = np.max([rx, ry, rz])

    ax.set_xbound(cx - r/1.5, cx + r/1.5)
    ax.set_ybound(cy - r/1.5, cy + r/1.5)
    ax.set_zbound(cz - r/1.5, cz + r/1.5)

    pyplot.show()


