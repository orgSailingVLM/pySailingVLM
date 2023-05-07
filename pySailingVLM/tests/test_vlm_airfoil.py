import numpy as np
import pandas as pd
import os
from unittest import TestCase
from pySailingVLM.thin_airfoil.vlm_airfoil import VlmAirfoil


class TestVlmAirfoil(TestCase):
    
    def test_airfoil(self):
        # 9501
        foil1 = VlmAirfoil(0.09, 0.5, 0.01)
        # 2415
        foil2 = VlmAirfoil(0.02, 0.4, 0.15)
        # 1800
        foil3 = VlmAirfoil(0.01, 0.8, 0.0)
        # 1028
        foil4 = VlmAirfoil(0.01, 0.0, 0.28) 
        
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, 'input_files/airfoils_coordinates.xlsx')

        e1 = pd.read_excel(open(filename, 'rb'), sheet_name='9501', usecols="A,B,C,D,E,F", header=None) 
        e2 = pd.read_excel(open(filename, 'rb'), sheet_name='2415', usecols="A,B,C,D,E,F", header=None) 
        e3 = pd.read_excel(open(filename, 'rb'), sheet_name='1800', usecols="A,B,C,D,E,F", header=None)
        e4 = pd.read_excel(open(filename, 'rb'), sheet_name='1028', usecols="A,B,C,D,E,F", header=None) 
        
        foils = [foil1, foil2, foil3, foil4]
        es = [e1, e2, e3, e4]
        
        for foil, e in zip(foils, es):
            np.testing.assert_array_almost_equal(foil.xu, np.array(e[0]))
            np.testing.assert_array_almost_equal(foil.yu, np.array(e[1]))
            np.testing.assert_array_almost_equal(foil.xl, np.array(e[2]))
            np.testing.assert_array_almost_equal(foil.yl, np.array(e[3]))
            np.testing.assert_array_almost_equal(foil.xc, np.array(e[4]))
            np.testing.assert_array_almost_equal(foil.yc, np.array(e[5]))
        

        # test raise exception for bad input
        # 0 <= m <= 30%
        # 0 <= p <= 90%
        # 0 <= xx <= 40%
        bad_args = [[-0.01, 0.8, 0.0], 
                    [0.1, -0.8, 0.0],
                    [0.1, 0.95, 0.0],
                    [0.1, 0.8, 0.5],
                    [0.1, 0.2, 0.5]]
        foil_errors = ['Max camber must be between 0 and 30%!',
                        'Max camber must be between 0 and 30%!'
                        'Max camber position must be between 0 and 90%!',
                        'Thickness must be between 0 and 40%!',
                        'Thickness must be between 0 and 40%!']
          
        for args, err in zip(bad_args, foil_errors):      
            with self.assertRaises(SystemExit) as context:
                VlmAirfoil(*args)
                print()
            print()
            self.assertEqual(context.exception.code, 1)
