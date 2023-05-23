import numpy as np
import SHABPy
import matmos
import math


def step3DOF(X, vehicle, flaperon_l, flaperon_r, dt, g, tgt):
    # Note State is [u, w, q, e0, e1, e2, e3, x, z]
    #               [0, 1, 2, 3,   4,  5,  6, 7, 8]
    V       = np.sqrt(X[0]**2 + X[1]**2)
    eul     = q2e(X[3:7])
    alpha   = np.arctan2(-X[1], X[0])
    gamma   = eul[1] - alpha
    atmos   = matmos.ISA(X[8]/1000)
    T = atmos.t
    rho = atmos.d

    R = 286
    a = np.sqrt(R * T * vehicle.gamma)

    # update vehicle mach number
    vehicle.M = V / a
    flaperon_l.M = V / a
    flaperon_r.M = V / a

    [dr, dl] = ControlFlaperon3DOF(tgt, X)

    [cp, cx, cy, cz, cmx, cmy, cmz, cl, cd, cyPrime] = SHABPy.RunSHABPy(alpha, 0, vehicle)
    [cp_l, cx_l, cy_l, cz_l, cmx_l, cmy_l, cmz_l, cl_l, cd_l, cyPrime_l] = SHABPy.RunSHABPy(alpha + dl, 0, flaperon_l)
    [cp_r, cx_r, cy_r, cz_r, cmx_r, cmy_r, cmz_r, cl_r, cd_r, cyPrime_r] = SHABPy.RunSHABPy(alpha + dr, 0, flaperon_r)

    Fx = 0.5 * rho * V ** 2 * vehicle.sref * (cx + cx_l + cx_r)
    Fz = 0.5 * rho * V ** 2 * vehicle.sref * (cz + cz_l + cz_r)

    My = 0.5 * rho * V ** 2 * vehicle.cbar * (cmy + cmy_l + cmy_r)

    correction = 1 - (X[3] ** 2 + X[4] ** 2 + X[5] ** 2 + X[6] ** 2)

    dX = np.array([- X[2] * X[1] - g * np.sin(eul[1]) - Fx / vehicle.m,
                   X[2] * X[0]  - g * np.cos(eul[1]) + Fz / vehicle.m,
                   vehicle.C[5] * My,
                   -0.5 * (X[5] * X[2]) + 0.5 * correction * X[3],
                   -0.5 * (X[6] * X[2]) + 0.5 * correction * X[4],
                    0.5 * (X[3] * X[2]) + 0.5 * correction * X[5],
                    0.5 * (X[4] * X[2]) + 0.5 * correction * X[6],
                   V * np.cos(gamma),
                   V * np.sin(gamma)])

    return X + dX * dt


def step(X, vehicle, flaperon_l, flaperon_r, dt, g, tgt):

    # Note That State is [u, v, w, p, q, r, e0, e1, e2, e3,  x,  y,  z]
    #                    [0, 1, 2, 3, 4, 5,  6,  7,  8,  9, 10, 11, 12]
    V       = np.sqrt(X[0]**2 + X[1]**2 + X[2]**2)
    eul     = q2e(X[6:10])
    alpha   = np.arctan(X[2]/X[0])
    beta    = np.arctan(X[1]/X[0])
    gamma   = eul[1] - alpha
    atmos   = matmos.ISA(X[12]/1000)
    T       = atmos.t
    pa      = atmos.p
    rho     = atmos.d

    R       = 286
    a       = np.sqrt(R*T*vehicle.gamma)

    #update vehicle mach number
    vehicle.M = V/a
    flaperon_l.M = V/a
    flaperon_r.M = V/a

    [dr, dl] = ControlFlaperon(tgt, X)

    [cp, cx, cy, cz, cmx, cmy, cmz, cl, cd, cyPrime] = SHABPy.RunSHABPy(alpha, beta, vehicle)
    [cp_l, cx_l, cy_l, cz_l, cmx_l, cmy_l, cmz_l, cl_l, cd_l, cyPrime_l] = SHABPy.RunSHABPy(alpha + dl, beta, flaperon_l)
    [cp_r, cx_r, cy_r, cz_r, cmx_r, cmy_r, cmz_r, cl_r, cd_r, cyPrime_r] = SHABPy.RunSHABPy(alpha + dr, beta, flaperon_r)

    Fx = 0.5 * rho * V ** 2 * vehicle.sref * (cx + cx_l + cx_r)
    Fy = 0.5 * rho * V ** 2 * vehicle.sref * (cy + cy_l + cy_r)
    Fz = 0.5 * rho * V ** 2 * vehicle.sref * (cz + cz_l + cz_r)

    Mx = 0.5 * rho * V ** 2 * vehicle.span * (cmx + cmx_l + cmx_r)
    My = 0.5 * rho * V ** 2 * vehicle.cbar * (cmy + cmy_l + cmy_r)
    Mz = 0.5 * rho * V ** 2 * vehicle.span * (cmz + cmz_l + cmz_r)

    correction = 1 - (X[6]**2 + X[7]**2 + X[8]**2 + X[9]**2)

    dX = np.array([  X[5]*X[1] - X[4]*X[2] - g*np.sin(eul[1]) + Fx/vehicle.m,
           -X[5]*X[0] + X[3]*X[2] - g*np.sin(eul[0])*np.cos(eul[1])  + Fy/vehicle.m,
            X[4]*X[0] - X[3]*X[1] - g*np.cos(eul[0])*np.cos(eul[1])  + Fz/vehicle.m,
            vehicle.C[3]*X[3]*X[4] + vehicle.C[4]*X[4]*X[5] + vehicle.C[1]*Mx + vehicle.C[2]*Mz,
            vehicle.C[7]*X[3]*X[5] - vehicle.C[6]*(X[3]**2 - X[5]**2) + vehicle.C[5]*My,
            vehicle.C[9]*X[3]*X[4] - vehicle.C[3]*X[4]*X[5] + vehicle.C[2]*Mx + vehicle.C[8]*Mz,
            -0.5*(X[7]*X[3] + X[8]*X[4] + X[9]*X[5]) + 0.5*correction*X[6],
            0.5*(X[6]*X[3] - X[9]*X[4] + X[8]*X[5]) + 0.5*correction*X[7],
            0.5*(X[9]*X[3] + X[6]*X[4] - X[7]*X[5]) + 0.5*correction*X[8],
            0.5*(-X[8]*X[3] + X[7]*X[4] + X[6]*X[5]) + 0.5*correction*X[9],
            V*np.cos(gamma)*np.cos(eul[2] - beta),
            V*np.cos(gamma)*np.sin(eul[2] - beta),
            V*np.sin(gamma)])

    return X + dX*dt

#Inertia Coefficients to reduce complexity of the state change
def getInertiaCoeffs(Ixx, Iyy, Izz, Ixz):
    C       = np.zeros(10)
    C[0]    = Ixx*Izz - Ixz**2
    C[1]    = Izz/C[0]
    C[2]    = Ixz/C[0]
    C[3]    = C[2]*(Ixx - Iyy + Izz)
    C[4]    = C[1]*(Iyy - Izz) - C[2]*Ixz
    C[5]    = 1/Iyy
    C[6]    = C[5]*Ixz
    C[7]    = C[5]*(Izz - Ixx)
    C[8]    = Ixx/C[0]
    C[9]    = C[8]*(Ixx - Iyy) + C[2]*Ixz
    return C

#Quaternion angles to Euler Angles
def q2e(quat):
    eul     = np.zeros(3)
    eul[0]  = np.arctan2(2*(quat[0]*quat[1] + quat[2]*quat[3]), 1 - 2*(quat[1]**2 + quat[2]**2))
    eul[1]  = np.arcsin(2*(quat[0]*quat[2]-quat[3]*quat[1]))
    eul[2]  = np.arctan2(2*(quat[1]*quat[2] + quat[0]*quat[3]), 1 - 2*(quat[2]**2 + quat[3]**2))
    return eul

def e2q(eul):
    quat    = np.zeros(4)

    cph     = np.cos(eul[0]/2)
    sph     = np.sin(eul[0]/2)
    cth     = np.cos(eul[1]/2)
    sth     = np.sin(eul[1]/2)
    cps     = np.cos(eul[2]/2)
    sps     = np.sin(eul[2]/2)

    quat[0] = cph*cth*cps + sph*sth*sps
    quat[1] = sph*cth*cps - cph*sth*sps
    quat[2] = cph*sth*cps + sph*cth*sps
    quat[3] = cph*cth*sps - sph*sth*cps

    return quat

def ControlFlaperon(tgt, X):
    [roll, pitch, yaw]     = q2e(X[6:10])

    dx = tgt[0] - X[10]
    dy = tgt[1] - X[11]
    dz = tgt[2] - X[12]

    tgt_pitch = np.arctan(dz/np.sqrt(dx**2 + dy**2))
    tgt_yaw   = np.arctan(dy/dx)

    dpitch = tgt_pitch - pitch
    dyaw   = tgt_yaw - yaw
    maxroll = 30*np.pi/180
    if dyaw > 0:
        droll = np.min([maxroll, dyaw]) - roll
    else:
        droll = np.max([-maxroll, dyaw]) - roll

    inc = 0.1*np.pi/180

    k_pitch     = 1*inc
    k_pitchrate = 1*inc
    k_roll      = 1*inc
    k_rollrate  = 1*inc

    pitch_dr = -k_pitch*dpitch + k_pitchrate*X[4]
    pitch_dl = -k_pitch*dpitch + k_pitchrate*X[4]

    roll_dr  =  k_roll*droll - k_rollrate*X[3]
    roll_dl  = - k_roll*droll + k_rollrate*X[3]

    dr = pitch_dr + roll_dr
    dl = pitch_dl + roll_dl

    maxdef = 25*np.pi/180
    dr = np.min([np.max([dr, -maxdef]), maxdef])
    dl = np.min([np.max([dl, -maxdef]), maxdef])

    return [dr, dl]

def ControlFlaperon3DOF(tgt, X):
    # Note State is [u, w, q, e0, e1, e2, e3, x, z]
    #               [0, 1, 2, 3,   4,  5,  6, 7, 8]
    # tgt = [x, z]
    [roll, pitch, yaw] = q2e(X[3:7])

    dx = tgt[0] - X[7]
    dz = tgt[1] - X[8]

    tgt_pitch = math.atan2(dz, dx)
    dpitch = tgt_pitch - pitch

    print(pitch, dpitch)

    k_pitch = 100
    k_pitchrate = 100

    pitch_dr = -k_pitch * dpitch + k_pitchrate * X[2]

    dr = pitch_dr

    maxdef = 12 * np.pi / 180
    dr = np.min([np.max([dr, -maxdef]), maxdef])

    return [dr, dr]
