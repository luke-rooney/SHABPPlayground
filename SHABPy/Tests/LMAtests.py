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
import VehicleTests
import pandas as pd

LMA = LoadVehicle.LoadLukeGeom()
LMA.UpdatePanelMethod(1)

alpha = np.array([-15, -12, -9, -6, -3, 0, 3, 6, 9, 12, 15])*np.pi/180

cl    = np.zeros(len(alpha))
cd    = np.zeros(len(alpha))

for i in range(len(alpha)):
    [cp, cx, cy, cz, cmx, cmy, cmz, cl_i, cd_i, cyPrime] = SHABPy.RunSHABPy(alpha[i], 0, LMA)
    cd[i] = cd_i
    cl[i] = cl_i




print(cl)
print(cd)

