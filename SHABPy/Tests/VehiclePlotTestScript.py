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

[unsw, flap_l, flap_r] = LoadVehicle.LoadUNSW5_Control()

unsw5 = LoadVehicle.LoadUNSW5()

VehicleTests.Plot3D(unsw)

