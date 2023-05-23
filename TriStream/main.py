import stl as mesh
import numpy as np

import CustomPlots
import sys
import pickle
import warnings
warnings.filterwarnings('ignore')

import ViscousCorrections


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    sys.setrecursionlimit(300)
    mesh = mesh.Mesh.from_file('LMA.stl')
    #MeshProcessing.GetPointEdgeFace(mesh)
    data = pickle.load(open('face_adjacency.pkl', 'rb'))

    face_adj    = data.faces
    vertices    = data.points
    edges       = data.edges

    normal      = mesh.get_unit_normals()

    V       = np.array([1, 0, 0])
    desc    = np.cross(np.cross(V, normal), normal)

    T_inf   = 247.6
    rho     = 0.004627
    speed   = 2207
    S_ref   = 0.137771
    Rey     = 570700
    M_inf   = 7
    mu      = 1.458*10**(-6)*T_inf**(3/2)/(T_inf + 110.4)

    alpha = np.array([-15, -12, -9, -6, -3, 0, 3, 6, 9, 12, 15]) * np.pi / 180

    # for i in range(len(alpha)):
    #     V   = ViscousCorrections.GetAirflowVector(alpha[i], 0)
    #     cfd, cfy, cfl = ViscousCorrections.RetrieveCoefficient(mesh, face_adj, edges, V, vertices, desc, T_inf, mu, M_inf, S_ref, speed, rho, normal, Rey)
    #     print('angle:', alpha[i]*180/np.pi)
    #     print('cfd: ', cfd)
    #     print('cfy: ', cfy)
    #     print('cfl: ', cfl)
    #     print('-------------------------------')

    #Multiprocessing run
    Ls = ViscousCorrections.StreamlineTrace(mesh, face_adj, edges, V, vertices, desc)

    # Profiling Run

    # Ls = np.zeros(len(face_adj))
    # iter = range(len(face_adj))
    #
    # for i in iter:
    #     print(i)
    #     Ls[i] = MeshProcessing.runTristream(i, mesh, face_adj, edges, V, vertices, desc)

    CustomPlots.PlotStreamlineLength(mesh, Ls)
