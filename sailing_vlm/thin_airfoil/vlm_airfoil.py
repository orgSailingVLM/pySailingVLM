import sys
import numpy as np
import matplotlib.pyplot as plt

class VlmAirfoil:
    # http://airfoiltools.com/airfoil/naca4digit
    
    def __init__(self, max_camber : float, max_camber_position: float, thickness : float, n : int = 100):
        self._m = max_camber
        self._p = max_camber_position
        self._t = thickness
        self.n = n
        
        self.__check_naca_params()
        self._xc = np.linspace(0, 1, self.n)
        self._yc = self.__calculate_y_camber()
    
    @property
    def xc(self) -> list:
        return self._xc
        
    @property
    def yc(self) -> list:
        return self._yc
    
   
    def __check_naca_params(self):
        try:
            if not(self._m <= 0.095 and self._m >= 0):
                raise ValueError('Max camber must be between 0 and 9.5%!')
            
            if not(self._p <= 0.9 and self._p >= 0):
                raise ValueError('Max camber position must be between 0 and 90%!')
            
            if not(self._t <= 0.4 and self._t >= 0):
                raise ValueError('Thickness must be between 0 and 40%!')
        except ValueError as err:
            print(err.args[0])
            sys.exit()
    
    def __get_y_camber(self, x):
        # 0 <= x <= p
        if x <= self._p and x >= 0:
            return self._m * (2 * self._p * x - x**2) / (self._p**2)
        # p < x <= 1
        elif x > self._p and x <= 1:
            return self._m * (1 - 2*self._p + 2*self._p*x -x**2) / (1 - self._p)**2
        else:
            raise ValueError('x is not betwwen 0 and 1')
        
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
