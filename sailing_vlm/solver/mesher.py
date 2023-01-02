import numpy as np
import airfoils

import  matplotlib.pyplot as plt
from scipy.interpolate import interp1d
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
        p = int(distance[counter]*100)
        m = int(camber[counter]*100)
        upper, lower = airfoils.gen_NACA4_airfoil(p=p, m=m, xx=0, n_points=n)
        
        x_upper = upper[0]
        y_upper = upper[1]
        
        x_lower = lower[0]
        y_lower = lower[0]
        
        # blad w paczce!
        foil = airfoils.Airfoil.NACA4('9502')
        # foil = airfoils.Airfoil.NACA4(f'{p}{m}00')
        foil.plot()
        #print(f"foil.y_lower = {foil.y_lower(x=[0.2, 0.6, 0.85])} \n\n")



        #step = 1. / n
        #chord_x = [step * i for i in range(0, int(step))]
        # tymczasowo
        n = 100
        chord_x = np.linspace(0, 1, num=n)
        camber = np.array(foil.camber_line(x=chord_x))
        y_upper = foil.y_upper(chord_x)

        
        plt.plot(chord_x, y_upper)
        # print(f"chord_x \t\t camber")
        # for x, c in zip(chord_x, camber):
        #     print(f"{x:.4f} \t\t\t {c:.4f}")

        # print(f"\nchord_x \t\t foil.y_upper")
        # for x, c in zip(chord_x, y_upper):
        #     print(f"{x:.4f} \t\t\t {c:.4f}")
            
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

    