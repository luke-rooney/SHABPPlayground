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

hemicyl = LoadVehicle.LoadHemiCylinder()
hemicyl.UpdatePanelMethod(5)

df      = pd.read_excel('/Users/lukerooney/Documents/UNSW_Masters/Data/HemiCylinder/AD0261501_Table4.xlsx')
df      = df.to_numpy()
machnum = df[:, 0]
cd_ref  = df[:, 3]
cd      = np.zeros(len(machnum))
print(cd_ref)

for i in range(len(machnum)):
    hemicyl.M = machnum[i]
    [cp, cx, cy, cz, cmx, cmy, cmz, cl, cd_t, cyPrime] = SHABPy.RunSHABPy(0, 0, hemicyl)
    cd[i] = cd_t

pyplot.figure()
pyplot.plot(machnum, cd_ref, 'x')
pyplot.plot(machnum, cd, 'o')
pyplot.show()

#VehicleTests.PlotPressureMap(hemicyl, cp)


