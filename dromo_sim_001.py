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
k0  = h0/la.norm(h0)
u1 = e0/la.norm(e0)
u2 = np.cross(k0, u1) 
# the unit vectors of the orbital frame
# i = r0/la.norm(r0)
# j = np.cross(k0, i) 
# # cos(sigma_0) = i(0) . u1(0)
# sigma_0 = math.acos(np.dot(i, u1))

# Initialise P matrix, page 4, formula (4):
P_0 = np.array([
        [1-2*(eta2_0**2 + eta3_0**2), 2*eta1_0*eta2_0 - 2*eta4_0*eta3_0, 2*eta1_0*eta3_0 - 2*eta4_0*eta2_0], 
        [2*eta1_0*eta2_0 + 2*eta4_0*eta3_0, 1-2*(eta1_0**2 + eta3_0**2),  2*eta2_0*eta3_0 - 2*eta4_0*eta1_0],
        [2*eta1_0*eta3_0 - 2*eta4_0*eta2_0, 2*eta3_0*eta2_0 + 2*eta4_0*eta1_0, 1-2*(eta1_0**2 + eta2_0**2)]
        ])

# page 5, formula (4):
P_01 = np.array([
        [math.cos(RAAN0d)*math.cos(omega0d) - math.cos(i0d)*math.sin(RAAN0d)*math.sin(omega0d), 
        -math.cos(RAAN0d)*math.sin(omega0d) - math.cos(i0d)*math.sin(RAAN0d)*math.cos(omega0d),
        math.sin(i0d)*math.sin(RAAN0d)], 
        [math.sin(RAAN0d)*math.cos(omega0d) + math.cos(i0d)*math.cos(RAAN0d)*math.sin(omega0d), 
        -math.sin(RAAN0d)*math.sin(omega0d) + math.cos(i0d)*math.cos(RAAN0d)*math.cos(omega0d),
        -math.sin(i0d)*math.cos(RAAN0d)], 
        [math.sin(i0d)*math.sin(omega0d), 
        math.sin(i0d)*math.cos(omega0d), 
        math.cos(i0d)]
        ])
# print(P_0, "\n", P_01) # NOTE these are different !!! they shouldnt be

unit_perifocal = np.array([u1, u2, k0])
unit_orb = np.empty((3,3))
unit_orb = np.matmul(unit_perifocal, la.inv(P_01))
i1 = unit_orb[0]
j1 = unit_orb[1]
k1 = unit_orb[2]

# print(unit_perifocal, unit_orb)

# Additional relationships
# s = 1 + zeta1 * math.cos(sigma) + zeta2 * math.sin(sigma)
# 1/r
# dr/dtau
# epsilon1 = eta1 * math.cos(sigma/2) + eta2 * math.sin(sigma/2)
# epsilon2 = eta2 * math.cos(sigma/2) - eta1 * math.sin(sigma/2)
# epsilon3 = eta3 * math.cos(sigma/2) + eta4 * math.sin(sigma/2)
# epsilon4 = eta4 * math.cos(sigma/2) - eta3 * math.sin(sigma/2)
# 1 = eta1**2 + eta2**2 +eta3**2 + eta4**2

def time_from_dromo(tau):
    # time in s
    # page 14. from paper
    time = t0 + tau * math.sqrt(a0d**3/const.GMe)
    return time


def pos_from_dromo(sigma, zeta3, s):
    # in the inertial frame, pos in km
    # page 15. from paper
    dummy = np.matmul(unit_orb, P_01)
    pos = a0d/(zeta3**2 * s) * np.matmul(dummy, np.array([math.cos(sigma), math.sin(sigma), 0]))
    return pos

print("pc: ", pos_from_dromo(sigma_0, zeta3_0, 1 + zeta1_0 * math.cos(sigma_0) + zeta2_0 * math.sin(sigma_0)))
# pc:  [   0.         6799.99996039    0.        ] when using P_0
# pc:  [-5.19161088e-14  6.79999996e+03 -3.19802611e-13] when using P_01

def vel_from_dromo(sigma, zeta1, zeta2, zeta3):
    # in the inertial frame, vel in km/s
    # page 15. from paper
    V1 = -zeta3*(math.sin(sigma)+zeta2)
    V2 =  zeta3*(math.cos(sigma)+zeta1)
    dummy = np.matmul(unit_orb, P_0)
    vel = (math.sqrt(const.GMe/a0d)) * np.matmul(dummy, np.array([V1, V2, 0]))
    return vel 


print("vc: ", vel_from_dromo(sigma_0, zeta1_0, zeta2_0, zeta3_0))
# vc:  [ 5.70376087 -2.67283453  8.63873693] when using P_0
# vc:  [-9.25897029e+00  3.41798928e-32 -5.34566903e+00] when using P_01


######### BOTH WRONG ? THESE SHOULD COINCIDE WITH THE INITIAL CONDITIONS...


# def dromo2cart(tau, zeta1, zeta2, zeta3, eta1, eta2, eta3, eta4):
#     return time, pos, vel



def dromo_basic(State, sigma, perturbation=None):
    """
    
    """
    tau, zeta1, zeta2, zeta3, eta1, eta2, eta3, eta4 = State
    # Auxiliary eq.
    s = 1 + zeta1 * math.cos(sigma) + zeta2 * math.sin(sigma)
    # Non-dimensional component of the perturbing acceleration
    # Keplerian motion
    apx = 0
    apy = 0
    apz = 0
    a_px = apx/(zeta3**4 * s**3)
    a_py = apy/(zeta3**4 * s**3)
    a_pz = apz/(zeta3**4 * s**3)
    
    if perturbation == 'J2':
        # transform the current dromo element in cartesian coord.
        # calculate the dimensional acceleration due to J2
        # adimensionalise it and go back to dromo eq.
        # is there a direct way? 
        # I looked at page 122 of Linear and Regular Celestial Mechanics, as in the paper citation, but there was not a hint on this
        pos = pos_from_dromo(sigma, zeta3, s)
        x, y, z = pos[0], pos[1], pos[2]
        r_norm = math.sqrt(x**2 + y**2 + z**2)
        apxd = ( (3/2)*const.J2*const.GMe*const.Re**2/r_norm**4 ) * (x/r_norm)*(5*(z**2/r_norm**2) -1) # x-component of dimensional perturbing acceleration
        apyd = ( (3/2)*const.J2*const.GMe*const.Re**2/r_norm**4 ) * (y/r_norm)*(5*(z**2/r_norm**2) -1) # y-component of dimensional perturbing acceleration
        apzd = ( (3/2)*const.J2*const.GMe*const.Re**2/r_norm**4 ) * (z/r_norm)*(5*(z**2/r_norm**2) -3) # z-component of dimensional perturbing acceleration
        apx = apxd * (a0d**2/const.GMe)
        apy = apyd * (a0d**2/const.GMe)
        apz = apzd * (a0d**2/const.GMe)
        a_px = apx/(zeta3**4 * s**3)
        a_py = apy/(zeta3**4 * s**3)
        a_pz = apz/(zeta3**4 * s**3)

    elif perturbation == 'L':
        # see if the above def. of i1,j1, k1 is correct
        # here t is the dimensional time ? 
        # r is the satellite position, page 23.
        # transform at each time step from dromo to cartesian to obtain the position, or is there a better way?
        t = time_from_dromo(tau)
        r3 = const.rl * (math.sin(const.omega_l*t)* i1 - math.cos(const.omega_l*t)/2 * (math.sqrt(3)*j1 +k1))
        r = pos_from_dromo(sigma, zeta3, s)
        acc_lunar = const.GMl * ( (r3 - r)/np.la.norm(r3 - r)**3 - r3/np.la.norm(r3)**3)
        apxd = acc_lunar[0] # x-component of dimensional perturbing acceleration
        apyd = acc_lunar[1] # y-component of dimensional perturbing acceleration
        apzd = acc_lunar[2] # z-component of dimensional perturbing acceleration
        apx = apxd * (a0d**2/const.GMe)
        apy = apyd * (a0d**2/const.GMe)
        apz = apzd * (a0d**2/const.GMe)
        a_px = apx/(zeta3**4 * s**3)
        a_py = apy/(zeta3**4 * s**3)
        a_pz = apz/(zeta3**4 * s**3)

    elif perturbation == 'J2_and_L':
        # superposition of the two perturbation, as above
        t = time_from_dromo(tau)
        r = pos_from_dromo(sigma, zeta3, s)
        x, y, z = r[0], r[1], r[2]
        r_norm = math.sqrt(x**2 + y**2 + z**2)
        r3 = const.rl * (math.sin(const.omega_l*t)* i1 - math.cos(const.omega_l*t)/2 * (math.sqrt(3)*j1 +k1))
        acc_lunar = const.GMl * ( (r3 - r)/np.la.norm(r3 - r)**3 - r3/np.la.norm(r3)**3)
        apxd = ( (3/2)*const.J2*const.GMe*const.Re**2/r_norm**4 ) * (x/r_norm)*(5*(z**2/r_norm**2) -1) + acc_lunar[0]
        apyd = ( (3/2)*const.J2*const.GMe*const.Re**2/r_norm**4 ) * (y/r_norm)*(5*(z**2/r_norm**2) -1) + acc_lunar[1]
        apzd = ( (3/2)*const.J2*const.GMe*const.Re**2/r_norm**4 ) * (z/r_norm)*(5*(z**2/r_norm**2) -3) + acc_lunar[2]
        apx = apxd * (a0d**2/const.GMe)
        apy = apyd * (a0d**2/const.GMe)
        apz = apzd * (a0d**2/const.GMe)
        a_px = apx/(zeta3**4 * s**3)
        a_py = apy/(zeta3**4 * s**3)
        a_pz = apz/(zeta3**4 * s**3)

    # EOMs
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
    St = odeint( dromo_basic, S0, tspan)
    # St = odeint( dromo_basic, S0, tspan, args=('J2', ) )
    np.savetxt("/Users/utente73/Desktop/res2.txt", St)


if __name__ == "__main__":
    main()