#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 14:16:59 2022

@author: lukerooney
"""
import math

import MethodsModule
import Vehicle
import SHABPy
import FlightDynamics
import numpy as np
import time
from matplotlib import pyplot
from mpl_toolkits import mplot3d

#vehicle parameters including initial mach number
M       = 7
gamma   = 1.4
cbar    = 0.6
span    = 0.6
sref    = 0.0293
xref    = 0.15
yref    = 0
zref    = -0.04
m       = 2710*span*cbar*0.05
compression = 1
expansion   = 1
Ixx     = 640
Iyy     = 1000
Izz     = 1000
Ixz     = 100

# M, gamma, cbar, span, sref, xref, yref, zref, m, stlfile, compression, expansion, Ixx, Iyy, Izz, Ixz
oml = Vehicle.Vehicle(M, gamma, cbar, span, sref, xref, yref, zref, m, "Vehicles/oml1.stl", compression, expansion, Ixx, Iyy, Izz, Ixz)

oml.mesh.rotate([1, 0, 0], math.radians(90))
oml.mesh.rotate([0, 0, 1], math.radians(90))

oml.mesh.update_areas()
oml.mesh.centroids /= 1000
oml.mesh.points    /= 1000
oml.mesh.areas     /= 1000*1000

oml.mesh.translate([-np.min(oml.mesh.x), 0, -np.max(oml.mesh.z)])

##3D PLOT
#figure = pyplot.figure()
#axes   = mplot3d.Axes3D(figure)

#axes.add_collection3d(mplot3d.art3d.Poly3DCollection(oml.mesh.vectors))
# Auto scale to the mesh size
#scale = oml.mesh.points.flatten()
#axes.auto_scale_xyz(scale, scale, scale)
#axes.set_xlabel('X Axis')
#axes.set_ylabel('Y Axis')
#axes.set_zlabel('Z Axis')

# Show the plot to the screen
#pyplot.show()

alpha = 0
beta  = 0

## Newtonian Results
[cp, cx, cy, cz, cmx, cmy, cmz, cl, cd, cyPrime] = SHABPy.RunSHABPy(alpha, beta, oml)
print("Newtonian CL Result: ", cl)

## Newtonian Prandtl Meyer Results
oml.UpdatePanelMethod(2)
[cp, cx, cy, cz, cmx, cmy, cmz, cl, cd, cyPrime] = SHABPy.RunSHABPy(alpha, beta, oml)
print("Newtonian Prandtl Meyer CL Result: ", cl)

## Newtonian Prandtl Meyer Results
oml.UpdatePanelMethod(3)
[cp, cx, cy, cz, cmx, cmy, cmz, cl, cd, cyPrime] = SHABPy.RunSHABPy(alpha, beta, oml)
print("Modified Newtonian CL Result: ", cl)

## Hankey Results
oml.UpdatePanelMethod(4)
[cp, cx, cy, cz, cmx, cmy, cmz, cl, cd, cyPrime] = SHABPy.RunSHABPy(alpha, beta, oml)
print("Modified Newtonian CL Result: ", cl)

## Busemann Results
oml.UpdatePanelMethod(6)
[cp, cx, cy, cz, cmx, cmy, cmz, cl, cd, cyPrime] = SHABPy.RunSHABPy(alpha, beta, oml)
print("Busemann CL Result: ", cl)

## Vandyke Results
oml.UpdatePanelMethod(5)
[cp, cx, cy, cz, cmx, cmy, cmz, cl, cd, cyPrime] = SHABPy.RunSHABPy(alpha, beta, oml)
print("Vandyke CL Result: ", cl)

eul     = [0, 0, 0]
quat    =  FlightDynamics.e2q(eul)

dt    = 0.0001
X0    = [2000, 0, 0, 0, 0, 0, quat[0], quat[1], quat[2], quat[3],  10,  0,  20000]
t     = 0.5
steps = int(t/dt)
X     = np.zeros((steps, 13))

X[0] = X0

for i in range(1, steps):

    start = time.time()
    X[i] = FlightDynamics.step(X[i-1], oml, dt, 9.81)
    end   = time.time()


    # Note That State is [u, v, w, p, q, r, e0, e1, e2, e3,  x,  y,  z]
    #                    [0, 1, 2, 3, 4, 5,  6,  7,  8,  9, 10, 11, 12]

time_arr    = range(0,steps)
u           = X[:, 0]
v           = X[:, 1]
w           = X[:, 2]
p           = X[:, 3]
q           = X[:, 4]
r           = X[:, 5]
e0          = X[:, 6]
e1          = X[:, 7]
e2          = X[:, 8]
e3          = X[:, 9]
x           = X[:, 10]
y           = X[:, 11]
z           = X[:, 12]
V           = np.sqrt(np.power(u,2) + np.power(v,2) + np.power(w,2))

phi         = np.zeros(len(time_arr))
theta       = np.zeros(len(time_arr))
psi         = np.zeros(len(time_arr))

for i in range(len(time_arr)):
    [phi[i], theta[i], psi[i]] = FlightDynamics.q2e([e0[i], e1[i], e2[i], e3[i]])

pyplot.figure(1)
pyplot.plot(time_arr, V)
pyplot.plot(time_arr, u)
pyplot.plot(time_arr, v)
pyplot.plot(time_arr, w)
pyplot.legend(['V', 'u', 'v', 'w'])
pyplot.xlabel('Time (s)')
pyplot.ylabel('Speed (m/s)')

pyplot.figure(2)
pyplot.plot(time_arr, phi)
pyplot.plot(time_arr, theta)
pyplot.plot(time_arr, psi)
pyplot.legend(['phi', 'theta', 'psi'])
pyplot.xlabel('Time (s)')
pyplot.ylabel('Angle (rad)')

#pyplot.figure(3)

pyplot.show()