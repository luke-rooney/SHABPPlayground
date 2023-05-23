import math
import numpy as np
import multiprocessing
from itertools import repeat
from numba import jit
import time

def BodyToWindAxisConversion(alpha, beta, phi, cx, cy, cz):
    ca = math.cos(alpha)
    sa = math.sin(alpha)
    cb = math.cos(beta)
    sb = math.sin(beta)
    cr = math.cos(phi)
    sr = math.sin(phi)

    CD = cx * ca * cb - cy * sr * sa * cb - cy * cr * sb + cz * cr * sa * cb - cz * sr * sb
    CL = -cx * sa - cy * sr * ca + cz * cr * ca
    cyPrime = cx * ca * sb - cy * sr * sa * sb + cy * cr * cb + cz * cr * sa * sb + cz * sr * cb

    return [CL, CD, cyPrime]


def GetDelta(normal, uinf):
    cosdel = - np.dot(normal, uinf)
    return np.pi / 2 - np.arccos(cosdel)


def GetAirflowVector(alpha, beta):
    return np.array([math.cos(alpha)*math.cos(beta), -math.sin(beta), math.sin(alpha)*math.cos(beta)])


def ComputeNetworkForcesAndMoments(normal, centroid, area, cp, xref, yref, zref):
    #compute the force magnitude
    fmag = -1 * np.multiply(area, cp.reshape(-1, 1))

    #compute the forces on all the individuals elements
    dFx = np.multiply(fmag, normal[:, 0].reshape(-1, 1))
    dFy = np.multiply(fmag, normal[:, 1].reshape(-1, 1))
    dFz = np.multiply(fmag, normal[:, 2].reshape(-1, 1))

    #compute all the moment contributions in the X directions
    dMx = -1 * dFy * (centroid[:, 2].reshape(-1, 1) - zref)
    dMx = dMx + dFz*(centroid[:, 1].reshape(-1, 1) - yref)

    #compute all the moment contributions in the y directions
    dMy = dFx * (centroid[:, 2].reshape(-1, 1) - zref)
    dMy = dMy - dFz*(centroid[:, 0].reshape(-1, 1) - xref)
    
    #compute all the moment contributions in the z directions
    dMz = -1 * dFx * (centroid[:, 1].reshape(-1, 1) - yref)
    dMz = dMz + dFy * (centroid[:, 0].reshape(-1, 1) - xref)

    Fx = np.sum(dFx)
    Fy = np.sum(dFy)
    Fz = np.sum(dFz)
    Mx = np.sum(dMx)
    My = np.sum(dMy)
    Mz = np.sum(dMz)

    return [Fx, Fy, Fz, Mx, My, Mz]


def RunSHABPy(alpha, beta, vehicle):
    normals     = vehicle.mesh.get_unit_normals()
    centroids   = vehicle.mesh.centroids
    areas       = vehicle.mesh.areas
    cp          = np.zeros(len(areas))

    uinf = GetAirflowVector(alpha, beta)

    # compute the vector of delta's
    deltaVec = GetDelta(normals, uinf)

    #Compression (where the angle between the flow vector and the panel is > 0)
    cp[(deltaVec > 0)] = vehicle.compression.calculatecompression(deltaVec[(deltaVec > 0)])

    # expansion (where the angle between the flow vector and the panel is <= 0)
    cp[deltaVec <= 0] = vehicle.expansion.calculateexpansion(deltaVec[deltaVec <= 0])

    [Fx, Fy, Fz, Mx, My, Mz]  = ComputeNetworkForcesAndMoments(normals, centroids, areas, cp, vehicle.xref, vehicle.yref, vehicle.zref)

    [cl, cd, cyPrime]         = BodyToWindAxisConversion(alpha, beta, 0, Fx, Fy, Fz)

    cx       = Fx/vehicle.sref
    cy       = Fy/vehicle.sref
    cz       = Fz/vehicle.sref
    cmx      = Mx/vehicle.span
    cmy      = My/vehicle.cbar
    cmz      = Mz/vehicle.span
    cl       = cl/vehicle.sref
    cd       = cd/vehicle.sref
    cyPrime  = cyPrime/vehicle.sref

    return [cp, cx, cy, cz, cmx, cmy, cmz, cl, cd, cyPrime]
