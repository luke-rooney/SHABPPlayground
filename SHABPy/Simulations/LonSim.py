
import FlightDynamics
import numpy as np
import time
import matmos
from matplotlib import pyplot
import LoadVehicle

[unsw, flap_l, flap_r]        = LoadVehicle.LoadUNSW5_Control()

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
X0    = [unsw.M*a,  0, 0, quat[0], quat[1], quat[2], quat[3],  0, alt]
t     = 75
steps = int(t/dt)
X     = np.zeros((steps, 9))

tgt   = [200000, 30000]

X[0] = X0

for i in range(1, steps):
    start = time.time()
    X[i] = FlightDynamics.step3DOF(X[i-1], unsw, flap_l, flap_r, dt, 9.81, tgt)
    end   = time.time()

time_arr    = np.arange(0, steps)*dt
u           = X[:, 0]
w           = X[:, 1]
q           = X[:, 2]
e0          = X[:, 3]
e1          = X[:, 4]
e2          = X[:, 5]
e3          = X[:, 6]
x           = X[:, 7]
z           = X[:, 8]
V           = np.sqrt(np.power(u, 2) + np.power(w, 2))

phi         = np.zeros(len(time_arr))
theta       = np.zeros(len(time_arr))
psi         = np.zeros(len(time_arr))

for i in range(len(time_arr)):
    [phi[i], theta[i], psi[i]] = FlightDynamics.q2e([e0[i], e1[i], e2[i], e3[i]])

pyplot.figure(1)
pyplot.plot(time_arr, V)
pyplot.plot(time_arr, u)
pyplot.plot(time_arr, w)
pyplot.legend(['V', 'u', 'w'])
pyplot.xlabel('Time (s)')
pyplot.ylabel('Speed (m/s)')

pyplot.figure(2)
pyplot.plot(time_arr, theta)
pyplot.legend(['theta'])
pyplot.xlabel('Time (s)')
pyplot.ylabel('Orientation Angle (rad)')

pyplot.figure(3)
pyplot.plot(x, z)
pyplot.plot(tgt[0], tgt[1], 'x')
pyplot.xlabel('X Position (m)')
pyplot.ylabel('Z Position (m)')
pyplot.axis('equal')
pyplot.grid('on')

pyplot.figure(4)
pyplot.plot(time_arr, q)
pyplot.xlabel('time (s)')
pyplot.ylabel('pitch rate (rad/s)')


pyplot.show()