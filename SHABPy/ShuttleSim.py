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
import matmos
from matplotlib import pyplot
from mpl_toolkits import mplot3d
import LoadVehicle

[unsw, flap_l, flap_r]        = LoadVehicle.LoadUNSW5_Control()
unsw5 = LoadVehicle.LoadUNSW5()

eul     = [0, 0, 0]
quat    =  FlightDynamics.e2q(eul)

alt     = 40000
atmos   = matmos.ISA(alt/1000)
T       = atmos.t
pa      = atmos.p
rho     = atmos.d

R       = 286
a       = np.sqrt(R*T*unsw.gamma)

dt    = 0.001
X0    = [unsw.M*a, 0, 0, 0, 0, 0, quat[0], quat[1], quat[2], quat[3],  0,  0,  alt]
t     = 25
steps = int(t/dt)
X     = np.zeros((steps, 13))

tgt   = [880000, 880000, 20000]

X[0] = X0

for i in range(1, steps):

    start = time.time()
    X[i] = FlightDynamics.step(X[i-1], unsw, flap_l, flap_r, dt, 9.81, tgt)
    end   = time.time()

time_arr    = np.arange(0,steps)*dt
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
V           = np.sqrt(np.power(u, 2) + np.power(v, 2) + np.power(w, 2))

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

pyplot.figure(3)
ax  = pyplot.axes(projection = '3d')
ax.plot3D(x, y, z)
ax.set_xlim3d([-2500, 42500])
ax.set_ylim3d([-22500, 22500])
ax.set_zlim3d([10000, 55000])

pyplot.show()