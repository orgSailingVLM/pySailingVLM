import numpy as np
import pandas as pd
import os
from unittest import TestCase
from sailing_vlm.thin_airfoil.vlm_airfoil import VlmAirfoil


class TestVlmAirfoil(TestCase):
    
    def test_airfoil(self):
        foil1 = VlmAirfoil('9501')
        foil2 = VlmAirfoil('2415')
        foil3 = VlmAirfoil('1800')
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
        