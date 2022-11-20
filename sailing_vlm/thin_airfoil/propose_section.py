import numpy as np
import pandas as pd
from airfoils import Airfoil
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

AoA_step = 0.2
alpha_start = -10
alpha_end = 10
AoA_range = [AoA_step * i for i in range(int(alpha_start / AoA_step), int(alpha_end / AoA_step))]

max_camber = int(9)  # range [0-9]
max_camber_location = int(2)  # range [0-9]
thickness = '20'
airfoil_name = f"naca{max_camber}{max_camber_location}{thickness}"

# Reynolds = 0 - inviscid flow --> makes xfoil more stable for thin airfoils
coeff = find_coefficients(airfoil=airfoil_name, alpha=AoA_range, Reynolds=1e6)

log_name = f"Polar_{airfoil_name}_{AoA_range[0]}_{AoA_range[-1]}"
df = pd.read_csv(log_name,  skiprows=[1, 2, 3, 4, 5, 6, 7, 8, 9, 11],  sep='\s+')

df['CLCD'] = df['CL']/df['CD']
# for some reason the {max_camber_location}-{max_camber} order is reverted in the Airfoil plotter
# foil = Airfoil.NACA4(f"{max_camber_location}{max_camber}{thickness}")
foil = Airfoil.NACA4(f"{max_camber_location}{max_camber}00")  # plot only camber line only
foil.plot(show=False, save=True)

camber_step = 0.01
camber_x = [camber_step * i for i in range(0, int(1. / camber_step))]
camber_z = np.array(foil.camber_line(x=camber_x))

print("camber_x \t - \t camber_z")
for x, c in zip(camber_x, camber_z):
    print(f"{x:.4f} \t\t\t {c:.4f}")

print("Airfoil Polar")
print(df)

xp = df['alpha']
fp = df['CL']
alpha_zero_lift_deg = np.interp(0, df['CL'], df['alpha'])
CL_desired = 1.0913
alpha_for_CL_desired = np.interp(CL_desired, df['CL'], df['alpha'])

print(f"alpha_zero_lift from xfoil: {alpha_zero_lift_deg} [deg]")
print(f"camber: {max_camber} | camber estimate: {(-alpha_zero_lift_deg * np.pi / 360) * 100:.3f}")

from sailingVLM.coeff_formulas import estimate_camber_from_AoA_0lift, calc_lift_slope_coeff
camber_estimate = estimate_camber_from_AoA_0lift(alpha_zero_lift_deg)
for i in range(4):
    a = calc_lift_slope_coeff(camber_estimate)
    print(f"a = {2*np.pi:.3f} \t a_aug = {a:.3f} ")
    camber_estimate = estimate_camber_from_AoA_0lift(alpha_zero_lift_deg)

print(f"V2 camber: {max_camber} | camber estimate: {100*camber_estimate:.3f}")

print(f"CL_desired {CL_desired:.3f}  --> alpha_for_CL_desired: {alpha_for_CL_desired:.3f} [deg]")
print('bye')