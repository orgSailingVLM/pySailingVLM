import sys
import numpy as np
import matplotlib.pyplot as plt

class VlmAirfoil:
    
    
    def __init__(self, naca : str, n : int = 100):
        self.naca = naca
        self.n = n
        
        self._m, self._p, self._xx = self.__unpack_naca_attr()
        self._xs = np.linspace(0, 1, self.n)
        self._yc = self.__calculate_y_camber()
    
    @property
    def xs(self) -> list:
        return self._xs
        
    @property
    def yc(self) -> list:
        return self._yc
    
    def __unpack_naca_attr(self):
        try:
            if len(self.naca) != 4:
                raise ValueError('NACA does not have 4 digits!')
            
            m = int(self.naca[0]) / 100
            p = int(self.naca[1]) / 10
            xx = int(self.naca[2:]) / 100
            
            return (m, p, xx)
        except ValueError as err:
            print(err.args)
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
        return np.array([self.__get_y_camber(x) for x in self._xs])
    
    def plot(self, show : bool = False):
        
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlim([0, 1])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.axis('equal')
        ax.grid()

        ax.plot(self._xs, self._yc, '-', color='blue')

        plt.subplots_adjust(left=0.10, bottom=0.10, right=0.98, top=0.98, wspace=None, hspace=None)
        if show:
            plt.show()

   
      
### tests
#foil = VlmAirfoil('9501', 100)
#foil.plot(True)
