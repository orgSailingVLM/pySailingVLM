import sys
import numpy as np
import matplotlib.pyplot as plt
import math 
class VlmAirfoil:
    # http://airfoiltools.com/airfoil/naca4digit
    # 1128 -> 0.01, 0.1, 0.28
    def __init__(self, max_camber : float, max_camber_position: float, thickness : float, n : int = 100):
        self._m = max_camber
        self._p = max_camber_position
        self._t = thickness
        self.n = n
        
        self.__a0 = 0.2969
        self.__a1 = -0.126
        self.__a2 = -0.3516
        self.__a3 = 0.2843
        self.__a4 = -0.1015
        self.__a4_te = -0.1036 # for x=1 (trailing edge)
        
        self.__check_naca_params()
        self._xc = np.linspace(0, 1, self.n)
        self._yc = self.__calculate_y_camber()
        
        self._yt = np.array([self.__get_camber_thickness(x) for x in self._xc])
        self.__dyc_dx = np.array([self.__get_dy_dx_camber(x) for x in self._xc])
        self.__theta = np.array([math.atan(x) for x in self.__dyc_dx])
        
        # u - upper, l - lower
        self._xu = self._xc - self._yt * np.sin(self.__theta)
        self._xl = self._xc + self._yt * np.sin(self.__theta)
        
        self._yu = self._yc + self._yt * np.cos(self.__theta)
        self._yl = self._yc - self._yt * np.cos(self.__theta)
    
    @property
    def xc(self) -> list:
        return self._xc
        
    @property
    def yc(self) -> list:
        return self._yc
    
    @property
    def yt(self) -> list:
        return self._yt
    
    @property
    def yu(self) -> list:
        return self._yu
    
    @property
    def yl(self) -> list:
        return self._yl
    
    @property
    def xu(self) -> list:
        return self._xu
    
    @property
    def xl(self) -> list:
        return self._xl
    
    def __check_naca_params(self):
        try:
            if not(self._m <= 0.3 and self._m >= 0):
                raise ValueError('Max camber must be between 0 and 30%!')
            
            if not(self._p <= 0.9 and self._p >= 0):
                raise ValueError('Max camber position must be between 0 and 90%!')
            
            if not(self._t <= 0.4 and self._t >= 0):
                raise ValueError('Thickness must be between 0 and 40%!')
        except ValueError as err:
            print(err.args[0])
            sys.exit(1)
    
    def __get_y_camber(self, x):
        # 0 <= x < p
        if x < self._p and x >= 0:
            return self._m * (2 * self._p * x - x**2) / (self._p**2)
        # p <= x <= 1
        elif x >= self._p and x <= 1:
            return self._m * (1 - 2*self._p + 2*self._p*x -x**2) / (1 - self._p)**2
        else:
            raise ValueError('x is not betwwen 0 and 1')
        
    def __get_dy_dx_camber(self, x):
        # 0 <= x < p
        if x < self._p and x >= 0:
            return 2 * self._m * (self._p  - x) / (self._p**2)
        # p <= x <= 1
        elif x >= self._p and x <= 1:
            return 2 * self._m * (self._p - x) / (1 - self._p)**2
        else:
            raise ValueError('x is not betwwen 0 and 1')
        
    def __get_camber_thickness(self, x):
        epsilon = 0.0001
        if 1.0 - x < epsilon:
            return 5 * self._t * (self.__a0 * math.sqrt(x) + self.__a1 *x + self.__a2 * x *x + self.__a3 * x * x * x + self.__a4_te * x * x * x * x) 
        else:
            return 5 * self._t * (self.__a0 * math.sqrt(x) + self.__a1 *x + self.__a2 * x *x + self.__a3 * x * x * x + self.__a4 * x * x * x * x) 
    
    def __calculate_y_camber(self):
        return np.array([self.__get_y_camber(x) for x in self._xc])
    
    def plot(self, show : bool = False):
        
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlim([0, 1])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.axis('equal')
        ax.grid()

        ax.plot(self._xc, self._yc, '-', color='blue')

        plt.subplots_adjust(left=0.10, bottom=0.10, right=0.98, top=0.98, wspace=None, hspace=None)
        if show:
            plt.show()

   
      
### tests
# 9500
#foil = VlmAirfoil(-0.09, 0.5, 0.0, 100)
#foil.plot(True)
