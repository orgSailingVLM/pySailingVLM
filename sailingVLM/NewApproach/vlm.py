from enum import unique
import numpy as np
from typing import Tuple
from numpy.linalg import norm


class Panels:

    def __init__(self, M: int, N: int, mesh, gamma_orientation : float = 1) -> None:

        #np.set_printoptions(precision=3, suppress=True)
        self.M = M
        # num of rows
        self.N = N
        self.gamma_orientation = gamma_orientation
        ### FLIGHT CONDITIONS ###
        V = 1 * np.array([10.0, 0.0, 0.0])
        V_app_infw = np.array([V for i in range(self.M * self.N)])

        # panels contain 12 panels, each has 4 vortices: p1, p2, p3, p4
        
        self.panels = mesh
        #self.panels = np.flip(self.panels, 0)
        #test000 = self.panels - self.panels0s
        self.areas = self.get_panels_area(self.panels, self.N, self.M) 
        self.normals, self.collocation_points, self.center_of_pressure, self.rings, self.span_vectors = self.calculate_normals_collocations_cps_rings_spans(self.panels)
        #self.coefs, self.RHS, self.wind_coefs = self.get_influence_coefficients(self.collocation_points, self.rings, self.normals, self.M, self.N, V_app_infw)

        self.coefs, self.RHS, self.wind_coefs, self.trailing_rings = self.get_influence_coefficients_spanwise(self.collocation_points, self.rings, self.normals, self.M, self.N, V_app_infw)

        # gamma wyszła nieco inna, ale to wplyw innego chyba spsosbu liczenia koncowych paneli
        self.big_gamma = self.solve_eq(self.coefs, self.RHS)

    def get_leading_edge_mid_point(self, p2: np.ndarray, p3: np.ndarray) -> np.ndarray:
        return (p2 + p3) / 2.

    def get_trailing_edge_mid_points(self, p1: np.ndarray, p4: np.ndarray) -> np.ndarray:
        return (p4 + p1) / 2.

    def calculate_normals_collocations_cps_rings_spans(self, panels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        K = panels.shape[0]
        ns = np.zeros((K, 3))
        span_vectors = np.zeros((K, 3))
        collocation_points = np.zeros((K, 3))
        center_of_pressure = np.zeros((K, 3))
        rings = np.zeros(shape=panels.shape)
        for idx, panel in enumerate(panels):
            p1 = panel[0]
            p2 = panel[1]
            p3 = panel[2]
            p4 = panel[3]

            vect_A = p4 - p2
            vect_B = p3 - p1

            leading_mid_point = self.get_leading_edge_mid_point(p2, p3)
            trailing_edge_mid_point = self.get_trailing_edge_mid_points(p1, p4)
            dist = trailing_edge_mid_point - leading_mid_point

            collocation_points[idx] = leading_mid_point + 0.75 * dist
            center_of_pressure[idx] = leading_mid_point + 0.25 * dist

            p2_p1 = p1 - p2
            p3_p4 = p4 - p3

            A = p1 + p2_p1 / 4.
            B = p2 + p2_p1 / 4.
            C = p3 + p3_p4 / 4.
            D = p4 + p3_p4 / 4.

            rings[idx] = np.array([A, B, C, D])
            
            # span vectors
            bc = C - B
            bc *= self.gamma_orientation
            span_vectors[idx] = bc
            #####
                
            n = np.cross(vect_A, vect_B)
            n = n / np.linalg.norm(n)

            ns[idx] = n

        return ns, collocation_points, center_of_pressure, rings, span_vectors
    """
    def check_singular_condition(self, r1: np.array, r2: np.array) -> bool:
        # strona 254 punkt 3.
        n1 = np.linalg.norm(r1)
        n2 = np.linalg.norm(r2)
        n3 = np.square(np.linalg.norm((np.cross(r1, r2))))
        if (n1 or n2 or n3) < 1e-9:
            return True
        return False
    """    
    def is_in_vortex_core(self, vector_list):
        for vec in vector_list:
            if norm(vec) < 1e-9:
                return True

    def vortex_line(self, p: np.array, p1: np.array, p2: np.array, gamma: float = 1.0) -> np.array:
        # strona 254
        r0 = np.array(p2 - p1)
        r1 = np.array(p - p1)
        r2 = np.array(p - p2)

        r1_cross_r2 = np.cross(r1, r2)
        
        q_ind = np.array([0, 0, 0])
        if self.is_in_vortex_core([r1, r2, r1_cross_r2]):
            return [0.0, 0.0, 0.0]
        else:
            q_ind = r1_cross_r2 / np.square(np.linalg.norm(r1_cross_r2))
            q_ind *= np.dot(r0, (r1 / np.linalg.norm(r1) - r2 / np.linalg.norm(r2)))
            q_ind *= gamma / (4 * np.pi)

        return q_ind

    def vortex_infinite_line(self, P: np.ndarray, A: np.array, r0: np.ndarray, gamma : float = 1.0):

        u_inf = r0 / norm(r0)
        ap = P - A
        norm_ap = norm(ap)

        v_ind = np.cross(u_inf, ap) / (
                    norm_ap * (norm_ap - np.dot(u_inf, ap)))  # todo: consider checking is_in_vortex_core
        v_ind *= gamma / (4. * np.pi)
        return v_ind

    def vortex_horseshoe(self, p: np.array, B: np.array, C: np.array, V_app_infw: np.ndarray,
                         gamma: float = 1.0) -> np.array:
        """
        B ------------------ +oo
        |
        |
        C ------------------ +oo
        """
        sub1 = self.vortex_infinite_line(p, C, V_app_infw, gamma)
        sub2 = self.vortex_line(p, B, C, gamma)
        sub3 = self.vortex_infinite_line(p, B, V_app_infw, -1.0 * gamma)
        q_ind = sub1 + sub2 + sub3
        return q_ind

    def vortex_ring(self, p: np.array, A: np.array, B: np.array, C: np.array, D: np.array,
                    gamma: float = 1.0) -> np.array:

        sub1 = self.vortex_line(p, A, B, gamma)
        sub2 = self.vortex_line(p, B, C, gamma)
        sub3 = self.vortex_line(p, C, D, gamma)
        sub4 = self.vortex_line(p, D, A, gamma)

        q_ind = sub1 + sub2 + sub3 + sub4
        return q_ind
    # do kontynuacji myslenia w tym kierunku
    def get_vortex_wake_induced_downwash(self, p: np.array, B: np.array, C: np.array, V_app_infw: np.ndarray,
                        gamma: float = 1.0) -> np.array:
        # fig. 12.4
        # eq. 12.5
        sub1 = self.vortex_infinite_line(p, C, V_app_infw, gamma)
        sub3 = self.vortex_infinite_line(p, B, V_app_infw, -1.0 * gamma)
        q_ind = sub1 + sub3
        return q_ind
    
    
    # to jest dla chordwise czyli dla ostatnich w pionie paneli
    def get_influence_coefficients(self, collocation_points: np.ndarray, rings: np.ndarray, normals: np.ndarray, M: int, N: int, V_app_infw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        m = collocation_points.shape[0]

        RHS = [-np.dot(V_app_infw[i], normals[i]) for i in range(normals.shape[0])]
        coefs = np.zeros((m, m))
        wind_coefs = np.zeros((m, m, 3))
        for i, point in enumerate(collocation_points):

            # loop over other vortices
            for j, ring in enumerate(rings):
                A = ring[0]
                B = ring[1]
                C = ring[2]
                D = ring[3]
                a = self.vortex_ring(point, A, B, C, D)

                # poprawka na trailing edge
                # todo: zrobic to w drugim, oddzielnym ifie
                if j >= len(collocation_points) - N:
                    a = self.vortex_horseshoe(point, ring[0], ring[3], V_app_infw[j])
                    #a = self.vortex_horseshoe(point, ring[1], ring[2], V_app_infw[j])
                b = np.dot(a, normals[i].reshape(3, 1))
                wind_coefs[i, j] = a
                coefs[i, j] = b
        RHS = np.asarray(RHS)
        
        return coefs, RHS, wind_coefs
    def get_influence_coefficients_spanwise(self, collocation_points: np.ndarray, rings: np.ndarray, normals: np.ndarray, M: int, N: int, V_app_infw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        m = collocation_points.shape[0]

        RHS = [-np.dot(V_app_infw[i], normals[i]) for i in range(normals.shape[0])]
        coefs = np.zeros((m, m))
        wind_coefs = np.zeros((m, m, 3))
        trailing_rings = []
        for i, point in enumerate(collocation_points):

            # loop over other vortices
            for j, ring in enumerate(rings):
                A = ring[0]
                B = ring[1]
                C = ring[2]
                D = ring[3]
                a = self.vortex_ring(point, A, B, C, D)

                
                # poprawka na trailing edge
                # todo: zrobic to w drugim, oddzielnym ifie
                if j >= len(collocation_points) - M:
                    #a = self.vortex_horseshoe(point, ring[0], ring[3], V_app_infw[j])
                    a = self.vortex_horseshoe(point, ring[1], ring[2], V_app_infw[j])
                b = np.dot(a, normals[i].reshape(3, 1))
                wind_coefs[i, j] = a
                coefs[i, j] = b
        RHS = np.asarray(RHS)
        
        for j, ring in enumerate(rings):
            if j >= len(collocation_points) - M:
                A = ring[0]
                B = ring[1]
                C = ring[2]
                D = ring[3]
                trailing_rings.append([A, B, C, D])
                    
        return coefs, RHS, wind_coefs, trailing_rings
    
    def metoda_testowa(self, points: np.ndarray, rings: np.ndarray, normals: np.ndarray, M: int, N: int, V_app_infw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        m = points.shape[0]

        #RHS = [-np.dot(V_app_infw[i], normals[i]) for i in range(normals.shape[0])]
        #coefs = np.zeros((m, m))
        wind_coefs = np.zeros((m, m, 3))
        #trailing_rings = []
        for i, point in enumerate(points):

            # loop over other vortices
            for j, ring in enumerate(rings):
                A = ring[0]
                B = ring[1]
                C = ring[2]
                D = ring[3]
                a = self.vortex_ring(point, A, B, C, D)

                # poprawka na trailing edge
                # todo: zrobic to w drugim, oddzielnym ifie
                if j >= len(points) - M:
                    #a = self.vortex_horseshoe(point, ring[0], ring[3], V_app_infw[j])
                    a = self.vortex_horseshoe(point, ring[1], ring[2], V_app_infw[j])
                b = np.dot(a, normals[i].reshape(3, 1))
                wind_coefs[i, j] = a
                #coefs[i, j] = b
        #RHS = np.asarray(RHS)
        
        
        return  wind_coefs
    
    
    def solve_eq(self, coefs: np.ndarray, RHS: np.ndarray):
        big_gamma = np.linalg.solve(coefs, RHS)
        return big_gamma
    
    def calc_induced_velocity(self, v_ind_coeff, gamma_magnitude):
        N = gamma_magnitude.shape[0]
        
        t2 = len(gamma_magnitude)
        V_induced = np.zeros((N, 3))
        for i in range(N):
            for j in range(N):
                V_induced[i] += v_ind_coeff[i,j] * gamma_magnitude[j]

        return V_induced

    @staticmethod
    def get_panels_area(panels: np.ndarray, N: int, M: int)-> np.ndarray:
        
        m = N * M
        areas = np.zeros(m, dtype=float)
        sh = panels.shape[0]
        for i in range(0, sh):
           
            p = [panels[i, 0], panels[i, 1], panels[i, 2], panels[i, 3]]
            path = []
            for j in range(len(p) - 1):
                step = p[j + 1] - p[j]
                path.append(step)

            area = 0
            for k in range(len(path) - 1):
                s = np.cross(path[k], path[k + 1])
                s = np.linalg.norm(s)
                area += 0.5 * s
            areas[i] = area   

        return areas
    
    def is_no_flux_BC_satisfied(self, V_app_fw, panels, areas, normals):

        N = panels.shape[0]
        flux_through_panel = np.zeros(shape=N)
        #panels_area = np.zeros(shape=N)

        # dla kazdego panelu
        for i in range(0, N):
            #panel_surf_normal = panels[i].get_normal_to_panel()
            #panels_area[i] = panels[i].get_panel_area()
            flux_through_panel[i] = -np.dot(V_app_fw[i], normals[i])

        for area in areas:
            if np.isnan(area) or area < 1E-14:
                raise ValueError("Solution error, panel_area is suspicious")

        for flux in flux_through_panel:
            if abs(flux) > 1E-12 or np.isnan(flux):
                raise ValueError("Solution error, there shall be no flow through panel!")

        return True

    # czesc kodu sie powtarza, zrobic osobna funkcje
    def calc_V_at_cp_new(self, V_app_infw, gamma_magnitude, panels, center_of_pressure, rings, M, N, normals):
            
           # :param V_app_infw: apparent wind
           # :param gamma_magnitude: vector with circulations
           # :param panels:
           # :return: Wind at cp = apparent wind + wind_induced_at_cp
            
            #panels_1d = panels.flatten()
            #N = panels.shape[0]
            #v_ind_coeff = np.full((N, N, 3), 0., dtype=float)
            #######
            m = M * N
            coefs = np.zeros((m, m))
            wind_coefs = np.zeros((m, m, 3))
            for i, point in enumerate(center_of_pressure):

                # loop over other vortices
                for j, ring in enumerate(rings):
                    A = ring[0]
                    B = ring[1]
                    C = ring[2]
                    D = ring[3]
                    a = self.vortex_ring(point, A, B, C, D)

                    # poprawka na trailing edge
                    # todo: zrobic to w drugim, oddzielnym ifie
                    if j >= len(center_of_pressure) - M:
                        
                        #a = self.vortex_horseshoe(point, ring[0], ring[3], V_app_infw[j])
                        a = self.vortex_horseshoe(point, ring[1], ring[2], V_app_infw[j])
                    b = np.dot(a, normals[i].reshape(3, 1))
                    # v_ind_coeff to jest u mnie wind_coefs
                    wind_coefs[i, j] = a
                    coefs[i, j] = b
                
            #######
           
            V_induced = self.calc_induced_velocity(wind_coefs, gamma_magnitude)
            # V_induced_re = V_induced.reshape(N, 3)
            V_at_cp = V_app_infw + V_induced
            return V_at_cp, V_induced

    #poprawic
    def calc_force_wrapper_new(self, V_app_infw, gamma_magnitude, panels, rho,  center_of_pressure, rings, M, N, normals, span_vectors):
        # Katz and Plotkin, p. 346 Chapter 12 / Three-Dimensional Numerical Solution
        # f. Secondary Computations: Pressures, Loads, Velocities, Etc
        #Eq (12.25)

        V_at_cp, V_induced = self.calc_V_at_cp_new(V_app_infw, gamma_magnitude, panels, center_of_pressure, rings, M, N, normals)

        print("a")
        K = M * N
        force_xyz = np.zeros((K, 3))

        for i in range(K):
            # for spanwise only!
            # if panel is leading edge
            gamma = 0.0
            if i < M:
                gamma = span_vectors[i] * gamma_magnitude[i]
            else:
                gamma = span_vectors[i] * (gamma_magnitude[i] - gamma_magnitude[i-M])
            force_xyz[i] = rho * np.cross(V_at_cp[i], gamma)


        return force_xyz



    def calc_pressure(self, forces, normals, areas, N , M):
        K = N*M
        p = np.zeros(shape=K)

        for i in range(K):
            p[i] = np.dot(forces[i], normals[i]) / areas[i]

        return p
