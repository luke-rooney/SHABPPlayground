import numpy as np
import networkx as nx
import math
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import h5py
from numba import njit
import pickle


class Face:
    def __init__(self, index, faces):
        self.index      = index
        self.points     = faces[index]
        self.adjacency  = GetAdjacentFaces(faces, index)


class WriteData:
    def __init__(self, faces, edges, points):
        self.faces      = faces
        self.edges      = edges
        self.points     = points

    def writefile(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self, file)
        file.close()


def GetPointIndex(arr, point):
    return np.where(np.all(arr == point, axis=1))[0]


def GetNextFaceIndex(face_list, adj_faces, edge):

    faces = np.array([face_list[adj_faces[0]].points, face_list[adj_faces[1]].points, face_list[adj_faces[2]].points])
    all_index = np.where(np.logical_and(np.any(faces == edge[0], axis=1), np.any(faces == edge[1], axis=1)))[0]
    return int(adj_faces[all_index])


def GetAdjacentFaces(faces, index):
    # Collects the faces adjacent to the current face
    # Inputs:
    # faces - [n x 3] numpy array of point indices which make up a face in the mesh
    # index - current face index
    # Outputs:
    # - [1 x 3] array of face indices which locate faces adjacent to current face
    current  = faces[index]
    allindex = np.where(np.logical_or(np.logical_or(np.logical_and(np.any(faces == current[0], axis=1), np.any(faces == current[1], axis=1)),
                        np.logical_and(np.any(faces == current[0], axis=1), np.any(faces == current[2], axis=1))),
                        np.logical_and(np.any(faces == current[1], axis=1), np.any(faces == current[2], axis=1))))[0]
    return np.delete(allindex, np.where(allindex == index), axis=None)


def GetPointEdgeFace(mesh):
    # A Preprocessing algorithm to create the file necessary for a viscous calculation
    # input the mesh object from numpy stl.

    x = mesh.x
    y = mesh.y
    z = mesh.z

    point = np.stack((np.array([x[:, 0], y[:, 0], z[:, 0]]).T, np.array([x[:, 1], y[:, 1], z[:, 1]]).T, np.array([x[:, 2], y[:, 2], z[:, 2]]).T))
    point = np.reshape(point, (-1, 3))
    point = np.unique(point, axis=0)

    edge  = np.array([])
    faces = np.array([])
    print('creating edges')
    for i in range(len(x)):
        f   = GetPointIndex(point, np.array([x[i, 0], y[i, 0], z[i, 0]]))[0]
        s   = GetPointIndex(point, np.array([x[i, 1], y[i, 1], z[i, 1]]))[0]
        t   = GetPointIndex(point, np.array([x[i, 2], y[i, 2], z[i, 2]]))[0]

        if edge.size:
            edge = np.vstack((edge, np.sort(np.array([f, s])), np.sort(np.array([f,t])), np.sort(np.array([s,t]))))
        else:
            edge = np.vstack((np.sort(np.array([f, s])), np.sort(np.array([f,t])), np.sort(np.array([s,t]))))

        if faces.size:
            faces = np.vstack((faces, np.sort(np.array([f, s, t]))))
        else:
            faces = np.sort(np.array([f, s, t]))

    edge = np.unique(edge, axis=0)

    face_adj = []

    print('creating face object list')
    for j in range(len(faces)):
        face_adj.append(Face(j, faces))

    print('writingfile')
    filename = 'face_adjacency.pkl'
    filewrite = WriteData(face_adj, edge, point)
    filewrite.writefile(filename)


def TraversePanel(start, entry_e, face, faces, points, edges, D, all_descent, Ls, face_index, normals, V):
    # Given a single panel and the start of the streamline, this function finds the exit point and length of the stream-
    # line across the panel
    # Inputs:
    # start   = coordinate point values of the starting point of the streamline [x, y, z]
    # entry_e = edge the start point lies on
    # face    = indexes of the three points that create the face panel [A, B, C]
    # faces   = all the faces n x 3 array
    # points  = list of coordinates for the vertices of the mesh, n x 3 input
    # edges   = list of all edges m x 2
    # D       = descent vector of the given face [i, j, k]
    # all_descent = all descent vectors n x 3
    # Ls      = current length of streamline
    # face_index = current index of face.
    # normals = list of normal vectors for each panel n x 3
    # V       = normalised incoming velocity vector 1 x 3 array
    # Outputs:
    # exit_p  = exit point [x, y, z]
    # Ls      = length of streamline across panel
    # exit_e  = exit edge indicies [A, B]

    if magnitude(D) < 1e-14:
        return Ls

    # determine expansion or compression:
    cosdel = - np.dot(normals[face_index], V)
    dv     = np.pi/2 - np.arccos(cosdel)
    # if expansion face, there is no streamline
    if dv <= 0:
        return Ls

    if Ls == 0:
        # vectors from centroid to each vertex of the face
        V_f = points[face.points[0]] - start
        V_s = points[face.points[1]] - start
        V_t = points[face.points[2]] - start

        # angles between descent and vectors to vertices
        a_f = AngleBetweenVectors(D, V_f)
        a_s = AngleBetweenVectors(D, V_s)
        a_t = AngleBetweenVectors(D, V_t)

        # find point order (minimum angle between descent and vertex indicates direction)
        if np.min([a_f, a_s, a_t]) == a_f:
            A = points[face.points[1]]
            B = points[face.points[2]]
            C = points[face.points[0]]
        elif np.min([a_f, a_s, a_t]) == a_s:
            A = points[face.points[0]]
            B = points[face.points[2]]
            C = points[face.points[1]]
        else:
            A = points[face.points[0]]
            B = points[face.points[1]]
            C = points[face.points[2]]

    else:
        A = points[entry_e[0]]
        B = points[entry_e[1]]
        indx = int(np.setdiff1d(np.array(face.points), entry_e))
        C = points[indx]

    # vectors from start to the vertexes
    Vv = C - start
    Sv = B - start

    # as per quadstream documentation, found in the documentation folder -
    theta_one  = AngleBetweenVectors(Sv, D)
    theta_four = AngleBetweenVectors(Sv, Vv)

    # if theta one is less than theta four then the streamline crosses edge B C, otherwise crosses edge A C
    if theta_one < theta_four:
        outer_edge = C - B
        f = face.points[GetPointIndex(points[face.points], C)[0]]
        s = face.points[GetPointIndex(points[face.points], B)[0]]
        crossing_edge = np.sort([f, s])
        sect_b = B
    else:
        outer_edge = C - A
        f = face.points[GetPointIndex(points[face.points], C)[0]]
        s = face.points[GetPointIndex(points[face.points], A)[0]]
        crossing_edge = np.sort([f, s])
        sect_b = A

    if checkEnd(D, outer_edge, start, sect_b):
        return Ls

    # move the starting point along the streamline to the edge of the next panel.
    new_start = VectorIntersection(D, outer_edge, start, sect_b)

    # get length across panel
    panel_Ls = CoodinateDistance(start, new_start)

    # intersection point has drifted off the panel - Occurs when adjacent descent vectors are pointed in contradicting
    # directions, indicating the start of the streamline
    if abs(CoodinateDistance(C, sect_b) - (CoodinateDistance(C, new_start) + CoodinateDistance(sect_b, new_start))) > 0.000001:
        return Ls

    # update the total streamline distance
    Ls += panel_Ls

    # find new face and recursion occurs
    new_face_index = GetNextFaceIndex(faces, faces[face_index].adjacency, crossing_edge)
    return TraversePanel(new_start, crossing_edge, faces[new_face_index], faces, points, edges,
                         all_descent[new_face_index], all_descent, Ls, new_face_index, normals, V)


def VectorIntersection(A, B, start_a, start_b):
    # returns the intersection of lines A and B which start at start_a and start_b respectively
        g = start_b - start_a
        h = magnitude(crossproduct(B, g))
        k = magnitude(crossproduct(B, A))
        if k == 0 or h == 0:
            breakpoint()
        l = h/k*A
        if np.sign(crossproduct(B, g)[0]) == np.sign(crossproduct(B, A)[0]):
            return l + start_a
        else:
            return start_a - l


def checkEnd(A, B, start_a, start_b):
    # if lines do not intersect return
    g = start_b - start_a
    h = magnitude(crossproduct(B, g))
    k = magnitude(crossproduct(B, A))
    if k == 0 or h == 0:
        return True
    return False


def AngleBetweenVectors(A, B):
    # returns an angle between two vectors A and B
    return np.arccos(np.dot(A, B)/(magnitude(A) * magnitude(B)))


def normalize(v):
    # normalisation of a vector v
    return v/np.sqrt(np.sum(v**2))


def magnitude(vector):
    # returns the magnitude of a vector
    return math.sqrt(sum(pow(element, 2) for element in vector))


def CoodinateDistance(A, B):
    # returns the distance between two coordinates
    return math.sqrt((A[0] - B[0])**2 + (A[1] - B[1])**2 + (A[2] - B[2])**2)


def runTristream(i, mesh, faces, edges, V, vertices, desc):
    # Function Returns the streamline length from the panel centroid
    # Inputs:
    # i         = index of starting face
    # mesh      = mesh object
    # faces     = faces list of points indices
    # edges     = edge list of points indices
    # V         = Freestream Vector
    # vertices  = list of all vertex locations
    # desc      = list of descent vectors for each panel
    #
    # Output
    # Ls        = Length of the streamline from panel to leading edge

    try:
        face_index = i
        face    = faces[face_index]
        start   = mesh.centroids[face_index]
        entry_e = edges[0, 0]
        Ls_temp = 0
        D       = desc[face_index]
        pathing = start
        Ls      = TraversePanel(start, entry_e, face, faces, vertices, edges, D, desc, Ls_temp, face_index, mesh.normals, V)
        return Ls
    except:
        print('recursion exceeded at ', i)
        return 0


def crossproduct(A, B):
    # returns cross product of two vectors
    z = [0, 0, 0]
    z[0] = A[1] * B[2] - A[2] * B[1]
    z[1] = A[2] * B[0] - A[0] * B[2]
    z[2] = A[0] * B[1] - A[1] * B[0]
    return z
