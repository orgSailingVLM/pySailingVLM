import numpy as np
import pandas as pd
import os
from unittest import TestCase
from sailing_vlm.thin_airfoil.vlm_airfoil import VlmAirfoil


class TestVlmAirfoil(TestCase):
    
    def test_airfoil(self):
        # 9501
        foil1 = VlmAirfoil(0.09, 0.5, 0.01)
        # 2415
        foil2 = VlmAirfoil(0.02, 0.4, 0.15)
        # 1800
        foil3 = VlmAirfoil(0.01, 0.8, 0.0)
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, 'input_files/airfoils_coordinates.xlsx')

        e1 = pd.read_excel(open(filename, 'rb'), sheet_name='9501', usecols="A,B", header=None) 
        e2 = pd.read_excel(open(filename, 'rb'), sheet_name='2415', usecols="A,B", header=None) 
        e3 = pd.read_excel(open(filename, 'rb'), sheet_name='1800', usecols="A,B", header=None) 
        
        np.testing.assert_array_almost_equal(foil1.xc, np.array(e1[0]))
        np.testing.assert_array_almost_equal(foil1.yc, np.array(e1[1]))
        
        np.testing.assert_array_almost_equal(foil2.xc, np.array(e2[0]))
        np.testing.assert_array_almost_equal(foil2.yc, np.array(e2[1]))
        
        np.testing.assert_array_almost_equal(foil3.xc, np.array(e3[0]))
        np.testing.assert_array_almost_equal(foil3.yc, np.array(e3[1]))
        
        
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
