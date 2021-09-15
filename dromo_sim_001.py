# =============================================================================
# Created By  : Francesca Covella
# Created Date: Thursday 02 September 2021
# =============================================================================
import math
from numpy.lib.function_base import append
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
from numpy import linalg as la 

import dromo_const as const
from dromo_func import state2orb

"""
DROMO FORMULATION
m is the mass of particle M moving in a fixed intertial frame Ox1y1z1
O is the center of a celestial body
M is acted upon by the gravitational force of the celestial body (Keplerian motion) and 
the remaining forces which are included in the perturbing force
* means dimensional (d)
"""

# Step 1: Define Initial Conditions (dimensional)
# @t=t0*, (x0*, v0*) and (a0*, e0*, i0*, RAAN0*, omega0*, theta0*)
r0d = np.array([0.0, -5888.9727, -3400.0]) #km   (class 'numpy.ndarray')
v0d = np.array([10.691338, 0.0, 0.0])      #km/s
# perigee height should be 6800 km (correct!, as the minimum altitude comes out to be 428 km)
[a0d, e0d, i0d, RAAN0d, omega0d, theta0d] = state2orb(r0d, v0d, const.GMe)
# print([a0d, e0d, i0d, RAAN0d, omega0d, theta0d])
# from the conversion comes out that a0* = 136,000 km
# and that the orbital and equatorial plane coincide, as i~0 and RAAN is not defined

# Step 2: non-dimensionalise the ICs
r0 = r0d / a0d #[km/km] = [-] since the orbit is highly elliptical normalise with the initial semimajor axis
               # otherwise use r0 = r0d/la.norm(r0d)
v0 = v0d * math.sqrt(a0d/const.GMe)    #[-]
t0d = 0 #s
t0  = t0d / math.sqrt((a0d**3)/const.GMe) #[-]

# Step 3: tranform the non-dimensional ICs (r0, v0) in DROMO elements
# (sigma; tau, zeta1, zeta2, zeta3, eta1, eta2, eta3, eta4)
h0 = np.cross(r0, v0) # 3-components vector
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

# tf = 288.12768941*24*3600            # s
tf = 2*np.pi * 50                    # roughly 50 orbits
# NOTE:
# 314*math.sqrt(136000**3/398601) = 24944156.272633158
# 24944156.272633158/(24*3600) = 288.7055124147356
delta_t = 1
n_steps = math.floor((tf-t0)/delta_t - 1)
# duration of integration in seconds
tspan = np.linspace(0, tf, n_steps)
# print("tspan: ", tspan) ---> 0 to 314


# @t0 the departure perifocal ref frame
# k  = h0/la.norm(h0)
# u1 = e0/la.norm(e0)
# u2 = np.cross(k, u1) 
# # the unit vectors of the orbital frame
# i = r0/la.norm(r0)
# j = np.cross(k, i) 
# # cos(sigma_0) = i(0) . u1(0)
# sigma_0 = math.acos(np.dot(i, u1))
# print("state vector: \n", sigma_0, tau_0, zeta1_0, zeta2_0, zeta3_0, eta1_0, eta2_0, eta3_0, eta4_0)

# Additional relationships
# s = 1 + zeta1 * math.cos(sigma) + zeta2 * math.sin(sigma)

# 1/r
# dr/dtau
# epsilon1 = eta1 * math.cos(sigma/2) + eta2 * math.sin(sigma/2)
# epsilon2 = eta2 * math.cos(sigma/2) - eta1 * math.sin(sigma/2)
# epsilon3 = eta3 * math.cos(sigma/2) + eta4 * math.sin(sigma/2)
# epsilon4 = eta4 * math.cos(sigma/2) - eta3 * math.sin(sigma/2)
# 1 = eta1**2 + eta2**2 +eta3**2 + eta4**2

# Non-dimensional component of the perturbing acceleration
# Keplerian motion
apx = 0
apy = 0
apz = 0
# Perturbed motion
# apxd = # x-component of dimensional perturbing acceleration
# apx = apxd * (a0d**2/const.GMe)
# ...
# a_px = apx/(zeta3**4 * s**3)
# a_py = apy/(zeta3**4 * s**3)
# a_pz = apz/(zeta3**4 * s**3)

# pass parameters: s, a_px, a_py, a_pz
# ? sigma appears in the right hand side of the equations...
def dromo_basic(State, sigma):
    """
    
    """
    tau, zeta1, zeta2, zeta3, eta1, eta2, eta3, eta4 = State
    
    s = 1 + zeta1 * math.cos(sigma) + zeta2 * math.sin(sigma)
    a_px = apx/(zeta3**4 * s**3)
    a_py = apy/(zeta3**4 * s**3)
    a_pz = apz/(zeta3**4 * s**3)

    dStatedsigma = [1/(zeta3**3 * s**2), 
                    s * math.sin(sigma) * a_px   + (zeta1 + (1+s)*math.cos(sigma)) * a_py, 
                    - s * math.cos(sigma) * a_px + (zeta2 + (1+s)*math.sin(sigma)) * a_py,
                    zeta3 * a_pz,
                    1/2 * a_pz * (eta4 * math.cos(sigma) - eta3 * math.sin(sigma)),
                    1/2 * a_pz * (eta3 * math.cos(sigma) + eta4 * math.sin(sigma)),
                    1/2 * a_pz * (-eta2* math.cos(sigma) + eta1 * math.sin(sigma)),
                    1/2 * a_pz * (-eta1* math.cos(sigma) - eta2 * math.sin(sigma))
                    ]
    return dStatedsigma


def main():
    """
    Functions:
    scipy.integrate.odeint(func, y0, t, args=(), ...)
    Integrate a system of ordinary differential equations.
    Outputs: 
    """
    St = odeint(dromo_basic, S0, tspan)
    np.savetxt("/Users/utente73/Desktop/res.txt", St)


if __name__ == "__main__":
    main()