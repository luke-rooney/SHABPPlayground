# Viscous Corrections for the S/HABP program
# Luke Rooney - l.rooney@adfa.edu.au
# Last Edited - 26/04/2023

# Calculations done referencing
# "Viscous optimized hypersonic v/averiders designed from flows over cones and minimum drag bodies"
# Written by Dr Stephen Corda - 1988
# University of Maryland

import math
import numpy as np
from . import MeshProcessing
from itertools import repeat
from multiprocessing import Pool, cpu_count, freeze_support


def wallTemperature(M, T_inf, Ls):
    # Wall Temperature Equation on figure 4.1, Page 48
    # M         - Mach Number
    # T_inf     - Upstream Temperature (K)
    # Ls        - Streamline Length (m)
    return (1 + (Ls - 1)/2*M**2)*T_inf


def referenceTemperature(T_inf, M_e, T_w):
    # Reference Temperature (4-3), Page 47
    # T_inf     - Upstream Temperature (K)
    # M_e       - edge Mach number
    # T_w       - Wall Temperature (K)
    return T_inf*(1 + 0.0032*M_e**2 + 0.58*(T_w/T_inf - 1))


def referenceViscosity(T_ref, T_inf, mu):
    # Reference Air Viscosity Equation (4-4), Page 47
    # T_ref     - Reference Temperature (K)
    # T_inf     - Upstream Temperature (K)
    # mu        - air viscosity
    omega = 0.75
    return np.sign(T_ref/T_inf)*(np.abs(T_ref/T_inf))**omega * mu


def localReynolds(rho_e, V_e, Ls, mu_e):
    # rho_e = boundary layer edge air density (kg/m^3)
    # V_e   = boundary layer edge air velocity (m/s^2)
    # Ls    = length of streamline to panel (m)
    # mu_e  = air viscosity at boundary layer edge

    return rho_e*V_e*Ls/mu_e


def laminarFriction(Rex, Tw, T_inf):
    # Rex   - Local Reynolds Number
    # Tw    - Wall Temperature (K)
    # T_inf - Upstream Temperature
    # gamma - Ratio of Specific Heats
    # Pr    -
    # M     - Mach number
    #
    # note: omega is the exponent of an assumed exponential variation of air viscosity with temperature
    omega  = 0.75
    return 0.664*Rex**(-0.5) * np.sign(Tw/T_inf)*(np.abs(Tw/T_inf))**((omega - 1)/2)


def turbulentFriction(Rex):
    # Turbulent Friction Calculation (4 - 6), Page 49
    # Rex    - Reference Reynolds Number

    return 0.0592/(Rex**0.2)


def frictionCoeff(Cf, D, A, Sref):
    # Gives the Friction Coefficient in the Drag, y (Side), Lift Axis
    # Cf    - Panel Friction Coefficient
    # D     - Normalised Descent Vector
    # A     - Panel Areas
    # Sref  - Vehicle Reference Area
    cf  = Cf * (-D) * A.reshape((-1, 1)) / np.sum(A) / Sref
    cfx = np.sum(cf[:, 0])
    cfy = np.sum(cf[:, 1])
    cfz = np.sum(cf[:, 2])
    return cfx, cfy, cfz


def StreamlineTrace(mesh, faces, edges, V, vertices, desc):

    iter = range(len(faces))

    with Pool(cpu_count()) as pool:
        Ls = pool.starmap(MeshProcessing.runTristream, zip(iter, repeat(mesh), repeat(faces), repeat(edges), repeat(V), repeat(vertices), repeat(desc)))

    return np.array(Ls)


def GetDelta(normal, uinf):
    cosdel = - np.dot(normal, uinf)
    return np.pi / 2 - np.arccos(cosdel)


def RetrieveCoefficient(mesh, faces, edges, V, vertices, desc, T_inf, mu, M_inf, S_ref, speed, rho, normals, Rey):
    Ls      = StreamlineTrace(mesh, faces, edges, V, vertices, desc)
    Tw      = wallTemperature(M_inf, T_inf, Ls)
    T_ref   = referenceTemperature(T_inf, M_inf, Tw)
    mu_ref  = referenceViscosity(T_ref, T_inf, mu)
    Re_loc  = localReynolds(rho, speed, Ls, mu_ref)
    cf      = np.zeros(len(Ls))

    cf[(normals[:, 2] >= 0)] = laminarFriction(Re_loc[(normals[:, 2] >= 0)], Tw[(normals[:, 2] >= 0)], T_inf)
    cf[(normals[:, 2] < 0)]  = turbulentFriction(Rey)
    cf[cf == -np.inf]        = 0
    cf[np.isnan(cf)]         = 0

    cf      = cf.reshape((-1, 1))
    desc    = desc.reshape((-1, 3))
    cfx, cfy, cfz  = frictionCoeff(cf, desc, mesh.areas, S_ref)
    return cfx, cfy, cfz


def GetAirflowVector(alpha, beta):
    return np.array([math.cos(alpha)*math.cos(beta), -math.sin(beta), math.sin(alpha)*math.cos(beta)])
