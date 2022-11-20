
from airfoils import Airfoil
import numpy as np
from numpy import diff

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

foil = Airfoil.NACA4('1028')
foil.plot()
print(f"foil.y_lower = {foil.y_lower(x=[0.2, 0.6, 0.85])} \n\n")



step = 0.1
chord_x = [step * i for i in range(0, int(1. / step))]
camber = np.array(foil.camber_line(x=chord_x))
y_upper = foil.y_upper(chord_x)

print(f"chord_x \t\t camber")
for x, c in zip(chord_x, camber):
    print(f"{x:.4f} \t\t\t {c:.4f}")

print(f"\nchord_x \t\t foil.y_upper")
for x, c in zip(chord_x, y_upper):
    print(f"{x:.4f} \t\t\t {c:.4f}")
