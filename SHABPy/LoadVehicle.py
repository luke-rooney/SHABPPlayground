
import math
import MethodsModule
from . import Vehicle
import SHABPy
import FlightDynamics
import numpy as np
import time
import matmos
from matplotlib import pyplot
from mpl_toolkits import mplot3d

def LoadShuttle():
    M = 7
    gamma = 1.4
    cbar = 12.06
    span = 23.79
    sref = 249.91
    xref = 0.665 * 31.5
    yref = 0
    zref = 0
    m = 83388
    compression = 1
    expansion = 1
    Ixx = 1169236
    Iyy = 8729397
    Izz = 8991771
    Ixz = -218615

    # M, gamma, cbar, span, sref, xref, yref, zref, m, stlfile, compression, expansion, Ixx, Iyy, Izz, Ixz
    shuttle = Vehicle.Vehicle(M, gamma, cbar, span, sref, xref, yref, zref, m, "Vehicles/shuttle_resized.stl", compression, expansion, Ixx, Iyy, Izz, Ixz)

    return shuttle

def LoadOML():
    M = 7
    gamma = 1.4
    cbar = 12.06
    span = 23.79
    sref = 380.8
    xref = 0.6 * 32.14
    yref = 0
    zref = 0
    m = 83388
    compression = 1
    expansion = 1
    Ixx = 1169236
    Iyy = 8729397
    Izz = 8991771
    Ixz = -218615

    # M, gamma, cbar, span, sref, xref, yref, zref, m, stlfile, compression, expansion, Ixx, Iyy, Izz, Ixz
    oml = Vehicle.Vehicle(M, gamma, cbar, span, sref, xref, yref, zref, m, "Vehicles/oml_sized.stl", compression, expansion, Ixx, Iyy, Izz, Ixz)

    return oml

def LoadUNSW5():
    M = 7
    gamma = 1.4
    cbar = 0.6
    span = 0.4446
    sref = 0.218
    xref = 0.57685
    yref = 0
    zref = -0.047
    m = 35
    compression = 1
    expansion = 1
    Ixx = 0.141
    Iyy = 2.531
    Izz = 2.596
    Ixz = 0.205

    filepath = "/Users/lukerooney/Documents/UNSW_Masters/SHABPy/Vehicles/unsw_sized.stl"

    # M, gamma, cbar, span, sref, xref, yref, zref, m, stlfile, compression, expansion, Ixx, Iyy, Izz, Ixz
    unsw = Vehicle.Vehicle(M, gamma, cbar, span, sref, xref, yref, zref, m, filepath, compression, expansion, Ixx, Iyy, Izz, Ixz)

    return unsw

def LoadUNSW5_Control():
    M = 7
    gamma = 1.4
    cbar = 0.6
    span = 0.4446
    sref = 0.218
    xref = 0.57685
    yref = 0
    zref = -0.047
    m = 35
    compression = 1
    expansion = 1
    Ixx = 0.141
    Iyy = 2.531
    Izz = 2.596
    Ixz = 0.205

    filepath    = "/Users/lukerooney/Documents/UNSW_Masters/SHABPy/Vehicles/final_body.stl"
    filepath_fr = "/Users/lukerooney/Documents/UNSW_Masters/SHABPy/Vehicles/final_fr.stl"
    filepath_fl = "/Users/lukerooney/Documents/UNSW_Masters/SHABPy/Vehicles/final_fl.stl"

    # M, gamma, cbar, span, sref, xref, yref, zref, m, stlfile, compression, expansion, Ixx, Iyy, Izz, Ixz
    unsw    = Vehicle.Vehicle(M, gamma, cbar, span, sref, xref, yref, zref, m, filepath, compression, expansion, Ixx, Iyy, Izz, Ixz)
    flap_l  = Vehicle.Vehicle(M, gamma, cbar, span, sref, xref, yref, zref, m, filepath_fl, compression, expansion, Ixx, Iyy, Izz, Ixz)
    flap_r  = Vehicle.Vehicle(M, gamma, cbar, span, sref, xref, yref, zref, m, filepath_fr, compression, expansion, Ixx, Iyy, Izz, Ixz)

    return [unsw, flap_l, flap_r]

def LoadHemiCylinder():
    M = 1.9
    gamma = 1.4
    cbar = 39.15*0.0254
    span = 5.8*0.0254
    sref = math.pi*(2.9*0.0254)**2
    xref = 39.15*0.0254/2
    yref = 0
    zref = 0
    m = 5
    compression = 1
    expansion = 1
    Ixx = 0.141
    Iyy = 2.531
    Izz = 2.596
    Ixz = 0.205

    filepath = "/Users/lukerooney/Documents/UNSW_Masters/SHABPy/Vehicles/AD0261501_HemiCylinder.stl"

    # M, gamma, cbar, span, sref, xref, yref, zref, m, stlfile, compression, expansion, Ixx, Iyy, Izz, Ixz
    hemicyl = Vehicle.Vehicle(M, gamma, cbar, span, sref, xref, yref, zref, m, filepath, compression, expansion, Ixx, Iyy,
                           Izz, Ixz)

    return hemicyl

def LoadLukeGeom():
    M = 7
    gamma = 1.4
    cbar = 515
    span = 467.6417
    sref = 0.137771*1000*1000
    xref = 515
    yref = 0
    zref = 0
    m = 5
    compression = 1
    expansion = 1
    Ixx = 0.141
    Iyy = 2.531
    Izz = 2.596
    Ixz = 0.205

    filepath = "/Users/lukerooney/Documents/UNSW_Masters/TriStream/LMA.stl"

    # M, gamma, cbar, span, sref, xref, yref, zref, m, stlfile, compression, expansion, Ixx, Iyy, Izz, Ixz
    LMA = Vehicle.Vehicle(M, gamma, cbar, span, sref, xref, yref, zref, m, filepath, compression, expansion, Ixx,
                              Iyy,
                              Izz, Ixz)

    return LMA