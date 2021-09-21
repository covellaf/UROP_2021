# Core imports
import numpy as np
from numpy import linalg as la
import pandas as pd
import math
from numpy.lib.function_base import append
# Other imports
from UROP_const import *
from UROP_aux_func import *


def Dromo2orbel(Lc, sigma, tau, zeta1, zeta2, zeta3, eta1, eta2, eta3, eta4):
    """
    conversion from DROMO elements to classical orbital elements
    taken from page 15-16 of Urrutxua et al. paper 2015
    """
    a = - Lc / (zeta3**2 * (zeta1**2 + zeta2**2 -1))
    e_norm = math.sqrt(zeta1**2 + zeta2**2)
    beta = np.arctan2(zeta2, zeta1)
    i = 2*np.arccos(math.sqrt(eta3**2 + eta4**2))
    RAAN = - np.arctan2(eta3, eta4) + np.arctan2(eta2, eta1)
    omega_tilda = - np.arctan2(eta3, eta4) - np.arctan2(eta2, eta1)
    omega = omega_tilda + beta
    theta = sigma - beta
    return a, e_norm, i, RAAN, omega, theta

r0d = np.array([0.0, -5888.9727, -3400.0]) #km   (class 'numpy.ndarray')
v0d = np.array([10.691338, 0.0, 0.0])      #km/s
[a0d, e0d, i0d, RAAN0d, omega0d, theta0d] = state2orb(r0d, v0d, GMe)
classic = [a0d, e0d, i0d, RAAN0d, omega0d, theta0d]
print("classical elements initial state: ", classic)

# Step 2: non-dimensionalise the ICs
r0 = r0d / a0d #[km/km] = [-] since the orbit is highly elliptical normalise with the initial semimajor axis
               # otherwise use r0 = r0d/la.norm(r0d)
v0 = v0d * math.sqrt(a0d/GMe)    #[-]
t0d = 0 #s
t0  = t0d / math.sqrt((a0d**3)/GMe) #[-]

# Step 3: tranform the non-dimensional ICs (r0, v0) in DROMO elements (sigma; tau, zeta1, zeta2, zeta3, eta1, eta2, eta3, eta4)
h0 = np.cross(r0, v0)                    # 3-components vector
e0 = - r0/la.norm(r0) - np.cross(h0, v0) # 3-components vector
sigma_0 = 0
tau_0   = t0
zeta1_0 = la.norm(e0)
zeta2_0 = 0
zeta3_0 = 1/la.norm(h0)
eta1_0  = math.sin(i0d/2)*math.cos((RAAN0d-omega0d)/2)
eta2_0  = math.sin(i0d/2)*math.sin((RAAN0d-omega0d)/2)
eta3_0  = math.cos(i0d/2)*math.sin((RAAN0d-omega0d)/2)
eta4_0  = math.cos(i0d/2)*math.cos((RAAN0d-omega0d)/2)
S0 = [tau_0, zeta1_0, zeta2_0, zeta3_0, eta1_0, eta2_0, eta3_0, eta4_0]
print("dromo initial state: ", S0)

test = Dromo2orbel(a0d, sigma_0, *S0)
print("the conversion yields: ", test)

for item, count in zip(test, range(np.shape(classic)[0])):
    if item == classic[count]:
        print("it is the same")
    else:
        print("not quite", item, classic[count])
