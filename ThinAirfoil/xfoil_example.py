
import numpy as np
import pandas as pd
from ThinAirfoil.xfoil_module import find_coefficients

# From wikipedia
# The NACA four-digit wing sections define the profile by:[1]
#
# First digit describing maximum camber as percentage of the chord.
# Second digit describing the distance of maximum camber from the airfoil leading edge in tenths of the chord.
# Last two digits describing maximum thickness of the airfoil as percent of the chord.[2]
# For example, the NACA 2412 airfoil has a maximum camber of 2% located 40% (0.4 chords)
# from the leading edge with a maximum thickness of 12% of the chord.
#
# The NACA 0015 airfoil is symmetrical, the 00 indicating that it has no camber.
# The 15 indicates that the airfoil has a 15% thickness to chord length ratio: it is 15% as thick as it is long.

step = 0.25
alfa_start = -5
alfa_end = 10
AoA_range = [step * i for i in range(int(alfa_start / step), int(alfa_end / step))]

max_camber = int(2)
max_camber_location = int(4)
thickness = int(10)
airfoil_name = f"naca{max_camber}{max_camber_location}{thickness}"

# Reynolds = 0 - inviscid flow --> makes xfoil more stable for thin airfoils
coeff = find_coefficients(airfoil=airfoil_name, alpha=AoA_range, Reynolds=0)

log_name = f"Polar_{airfoil_name}_{AoA_range[0]}_{AoA_range[-1]}"
df = pd.read_csv(log_name,  skiprows=[1, 2, 3, 4, 5, 6, 7, 8, 9, 11],  sep='\s+')
print(df)
