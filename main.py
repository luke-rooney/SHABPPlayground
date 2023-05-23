# This is a S/HABP Tutorial Script
# This is the main script to run, all the packages are imported yeehaw.
import numpy

import SHABPy.Vehicle as Vehicle
import SHABPy.SHABPy  as Shabpy
import TriStream.ViscousCorrections as vc
import TriStream.MeshProcessing as meshproc
import pickle
import SHABPPlots
import numpy as np

if __name__ == '__main__':
    # Part 1 OHHHH YEAH. Non-Viscous Analysis!
    # find your stl file - yeah!
    filename = "Vehicles/LMA.stl"

    # Part 1.1 Create your vehicle!
    # These are the vehicle and flight conditions for S/HABP
    # If you are just looking for CL / CD the only coefficients used are:
    #   - M
    #   - gamma
    #   - sref
    #   - compression
    #   - expansion
    # Other parameters are used in the moment coefficient calculations.
    M = 7                                                           # Mach Number
    gamma = 1.4                                                     # Ratio of Specific Heats
    cbar = 1                                                        # average aerodynamic chord length (m)
    span = 1                                                        # span of vehicle (m)
    sref = 1                                                        # reference area (m^2)
    xref = 1                                                        # cg (x axis) (m)
    yref = 0                                                        # cg (y axis) (m)
    zref = 0                                                        # cg (z axis) (m)
    m = 10                                                          # mass (kg)
    compression = 1                                                 # compression method 1 - 6
    expansion = 1                                                   # expansion method 1 - 6
    Ixx = 1                                                         # mass moment of inertia xx axis
    Iyy = 1                                                         # mass moment of inertia yy axis
    Izz = 1                                                         # mass moment of inertia zz axis
    Ixz = 1                                                         # mass moment of inertia xz axis

    # Part 1.2 Retrieve formatted vehicle object
    vehicle = Vehicle.Vehicle(M, gamma, cbar, span, sref, xref, yref, zref, m, filename, compression, expansion, Ixx, Iyy, Izz, Ixz)

    # Part 2 - Using SHABP - here we go!!!!
    # Basic Run:
    # Here are your orientation angles for the vehicle. how good.
    alpha = 0
    beta  = 0

    [cp, cx, cy, cz, cmx, cmy, cmz, cl, cd, cyPrime] = Shabpy.RunSHABPy(alpha, beta, vehicle)
    # coefficients returned:
    # cp  - Panel pressures
    # cx  - x axis coefficient
    # cy  - y axis coefficient
    # cz  - z axis coefficient
    # cmx - moment coefficient about x axis (roll)
    # cmy - moment coefficient about y axis (pitch)
    # cmz - moment coefficient about z axis (yaw)
    # cl  - lift coefficient
    # cd  - drag coefficient
    # cy' - side force coefficient

    # Part 2.1 SANITY CHECKING (plot pressure map)
    SHABPPlots.PlotPressureMap(cp, vehicle.mesh)

    # Part 2.2 ooooh mama time to plot some coefficients
    # Panel Methods;
    # 1 - Newtownian
    # 2 - Newtonian Prandyl Meyer
    # 3 - Modified Newtonian
    # 4 - Hankey (I don't recommend this its garbage)
    # 5 - Van Dyke
    # 6 - Busemann Second Order Theory
    methods = [1, 2]                                                    # These are the panel methods to compare / use (1 - 6)
    aoa     = np.array([-15, -10, -5, 0, 5, 10, 15]) * np.pi / 180      # angle of attacks to check (radians of course, always radians)

    # set up empty arrays - gotta fill these things for results
    cx      = numpy.zeros((len(methods), len(aoa)))
    cy      = numpy.zeros((len(methods), len(aoa)))
    cz      = numpy.zeros((len(methods), len(aoa)))
    cmx     = numpy.zeros((len(methods), len(aoa)))
    cmy     = numpy.zeros((len(methods), len(aoa)))
    cmz     = numpy.zeros((len(methods), len(aoa)))
    cl      = numpy.zeros((len(methods), len(aoa)))
    cd      = numpy.zeros((len(methods), len(aoa)))
    cyPrime = numpy.zeros((len(methods), len(aoa)))

    # collecting the results
    for i in range(len(methods)):
        vehicle.UpdatePanelMethod(methods[i])
        for j in range(len(aoa)):
            [cp, cx[i, j], cy[i, j], cz[i, j], cmx[i, j], cmy[i, j], cmz[i, j], cl[i, j], cd[i, j], cyPrime[i, j]] = Shabpy.RunSHABPy(aoa[j], 0, vehicle)

    # plot information.
    xlabel = 'Angle of Attack (degrees)'
    ylabel = 'Coefficient of Lift'
    title  = 'CL v AoA'
    legend_text = ['Newtownian', 'Newtonian Prandyl Meyer']
    SHABPPlots.PlotXY(aoa*180/np.pi, cl, xlabel, ylabel, legend_text, title)

    # Part 3 Viscous Analysis if time permits. this may make your laptop sound like it is going to explode.
    # SWITCH TO LMA.stl for this part!!!!!
    # Some additional flight condition info for viscous analysis
    T_inf = 247.6                                                       # Temperature Upstream (K)
    rho = 0.004627                                                      # air density (kg/m^3)
    speed = 2207                                                        # speed (m/s)
    Rey = 570700                                                        # Reynolds number
    mu = 1.458 * 10 ** (-6) * T_inf ** (3 / 2) / (T_inf + 110.4)        # Air Viscosity

    # Generate viscous preprocessing pkl file - Lengthy Process - Only needs to be done once - comment when done once.
    # I have already made the file for LMA.stl to save time - does not need to be done.
    # meshproc.GetPointEdgeFace(vehicle.mesh)

    # load in your pickle file with all your mesh processing info.
    data = pickle.load(open('lma_faces.pkl', 'rb'))
    face_adj = data.faces                               # Object containing face adjacencys to streamline trace
    vertices = data.points                              # Verticies on the Mesh
    edges    = data.edges                               # Edges on the mesh - defined by vertex indices
    normal   = vehicle.mesh.get_unit_normals()          # normal vectors of each panel
    V        = np.array([1, 0, 0])                      # basic airstream vector (aoa = 0, beta = 0)
    desc = np.cross(np.cross(V, normal), normal)        # airstream descent vectors for each panel
    # SANITY CHECKING - Important - Lengthy (potentially).
    # our viscous analysis depends on the streamline length to the panel from a leading edge.
    Ls = vc.StreamlineTrace(vehicle.mesh, face_adj, edges, V, vertices, desc)
    # Plot will heatmap your streamline length
    SHABPPlots.PlotPressureMap(Ls, vehicle.mesh)

    # Gathering Viscous Correction Data
    # oooooooh yeeeeah getting serious now
    cfd = np.zeros((1, len(aoa)))
    cfy = np.zeros((1, len(aoa)))
    cfl = np.zeros((1, len(aoa)))

    for i in range(len(aoa)):
        V   = vc.GetAirflowVector(aoa[i], 0)
        cfd[i], cfy[i], cfl[i] = vc.RetrieveCoefficient(vehicle.mesh, face_adj, edges, V, vertices, desc, T_inf, mu, vehicle.M, vehicle.sref, speed, rho, normal, Rey)

        # plot information.
        xlabel = 'Angle of Attack (degrees)'
        ylabel = 'Viscous Coefficient of Lift'
        title = 'CLf v AoA'
        legend_text = ['CL_f']
        SHABPPlots.PlotXY(aoa * 180 / np.pi, cfl, xlabel, ylabel, legend_text, title)