import numpy as np
import airfoils
from sailing_vlm.solver.interpolator import Interpolator

def discrete_segment(p1, p2, n):
    segment = []
    step = (p2 - p1) / n

    for i in range(n):
        point = p1 + i * step
        segment.append(point)

    segment.append(p2)
    return np.array(segment)


# def interpolate_segemnt(p1, p2, n, interpolation_type, girths, chords):
#     interpolator = Interpolator(interpolation_type)
#     interpolator.interpolate_girths(girths, chords, n + 1),
    
def make_point_mesh(segment1, segment2, n):
    mesh = []
    # TODO: dodac camber w stylu NACA airfoil --> wzorek z wikipedii
    # TODO: upewnic sie ze normalna jest w dobra strone dla pochylonej lodki
    for p1, p2 in zip(segment1, segment2):
        s = discrete_segment(p1, p2, n)
        mesh.append(np.array(s))

    return np.array(mesh)

def make_airfoil_mesh(segment1, segment2, n, distance, camber):
    # segment1 , segemnt2 
    # distance and camber interpolated from girths
    # p = [0.4 0.45 0.56]
    mesh = []
    counter = 0
    for p1, p2 in zip(segment1, segment2):
        upper, lower = airfoils.gen_NACA4_airfoil(p=distance[counter]*100, m=camber[counter]*100, xx=0, n_points=n)
        # x_prims = upper[0]
        # y_prims = upper[1]
        x_prims = lower[0]
        y_prims = lower[1]
        
        xs = (p2[0] - p1[0]) * x_prims + p1[0] 
        ys = y_prims + p1[1]
        zs = np.linspace(p2[2], p1[2], num=n, endpoint=True)
        counter += 1
        mesh.append(list(zip(xs,ys,zs)))

    return np.array(mesh)
    # import matplotlib.pyplot as plt  
    # ax = plt.axes(projection='3d')
    # ax.plot3D(xs, ys, zs)
    # ax.plot3D(x_primes, y_primes, zs)

    