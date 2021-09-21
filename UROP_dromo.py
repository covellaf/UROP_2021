# =============================================================================
# Created By  : Francesca Covella
# Created Date: Friday 17 September 2021
# =============================================================================

# Core imports
import numpy as np
from numpy import linalg as la
import pandas as pd
import math
from numpy.lib.function_base import append
from scipy.integrate import odeint
# Sys imports
import time
# Plot imports
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as mtick
# Other imports
from UROP_const import *
from UROP_aux_func import *

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
[a0d, e0d, i0d, RAAN0d, omega0d, theta0d] = state2orb(r0d, v0d, GMe)
# print([a0d, e0d, i0d, RAAN0d, omega0d, theta0d])
# from the conversion comes out that a0* = 136,000 km
# and that the orbital and equatorial plane coincide, as i~0 and RAAN is not defined

# Step 2: non-dimensionalise the ICs
r0 = r0d / a0d #[km/km] = [-] since the orbit is highly elliptical normalise with the initial semimajor axis
               # otherwise use r0 = r0d/la.norm(r0d)
v0 = v0d * math.sqrt(a0d/GMe)    #[-]
t0d = 0 #s
t0  = t0d / math.sqrt((a0d**3)/GMe) #[-]

# Non-dimensionalise the parameters
GMe_hat = GMe * a0d / (a0d * GMe) # = 1
Re_hat  = Re / a0d
rl_hat  = rl / (6*Re)
omega_l_hat = omega_l * math.sqrt((a0d**3)/GMe)
GMl_hat = GMl * a0d / (a0d * GMe)
# check if the formula for GMl nd is correct!  

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

# Final time
tfd = 288.12768941*24*3600                   # s
tf  = tfd / math.sqrt((a0d**3)/GMe)          #[-]
# tf = 2*np.pi * 50                      # roughly 50 orbits

# delta_t = (tf-t0)/(N-1)
delta_td = 500                               # s (max time beween consecutive integration points)
delta_t  = delta_td / math.sqrt((a0d**3)/GMe)          #[-]
N = math.floor((tf-t0)/delta_t - 1)
print(N)
TSPAN = np.linspace(0, tf, N)               # duration of integration in seconds

# NOTE:
# 314*math.sqrt(136000**3/398601) = 24944156.272633158
# 24944156.272633158/(24*3600) = 288.7055124147356
# print("tspan: ", tspan) ---> 0 to 314

dromo2orb_res = []


def dromo_keplerian(State, sigma, perturbation=None):
    """
    Equations to be integrated
    Propagate DROMO EOMs using odeint
    """
    tau, zeta1, zeta2, zeta3, eta1, eta2, eta3, eta4 = State
    dromo2orb_res.append(Dromo2orbel(a0d, sigma, tau, zeta1, zeta2, zeta3, eta1, eta2, eta3, eta4))
    # Auxiliary eq.
    s = 1 + zeta1 * math.cos(sigma) + zeta2 * math.sin(sigma)
    # Keplerian motion
    apx = 0
    apy = 0
    apz = 0
    # Divide by a conventional factor the non-dimensional acceleration components
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


def dromo_perturbed(State, sigma, perturbation=None):
    """
    Equations to be integrated
    Propagate DROMO EOMs using odeint
    """
    tau, zeta1, zeta2, zeta3, eta1, eta2, eta3, eta4 = State
    dromo2orb_res.append(Dromo2orbel(a0d, sigma, tau, zeta1, zeta2, zeta3, eta1, eta2, eta3, eta4))
    # Auxiliary eq.
    s = 1 + zeta1 * math.cos(sigma) + zeta2 * math.sin(sigma)
    # Dromo element to cartesian: 1. Dromo2orbel, 2. orb2state   
    # Dimensional position and velocity    
    pos, vel = orb2state(*Dromo2orbel(a0d, sigma, tau, zeta1, zeta2, zeta3, eta1, eta2, eta3, eta4))
    # Non-dimensionalise the norm of the position vector of the satellite 
    x, y, z = pos[0]/a0d, pos[1]/a0d, pos[2]/a0d
    r = math.sqrt(x**2 + y**2 + z**2)
    # The lunar position is not defined by an ephemeris model but by trigonometric functions based on the time of the
    # integration. r (pos) is the satellite position, page 23.
    iI = np.array([1, 0, 0])
    jI = np.array([0, 1, 0])
    kI = np.array([0, 0, 1])
    # the position of the Moon depends explicitly on time
    r3 = rl_hat * (math.sin(omega_l_hat*tau)*iI - math.cos(omega_l_hat*tau)/2 * (math.sqrt(3)*jI + kI))
    acc_lunar = GMl_hat * ( (r3 - np.array([x, y, z]))/la.norm(r3 - np.array([x, y, z]))**3 - r3/la.norm(r3)**3)
    # x-component of dimensional perturbing acceleration
    apxd = ( (3/2)*J2*GMe_hat*Re_hat**2/r**4 ) * (x/r)*(5*(z**2/r**2) -1)  + acc_lunar[0] 
    # y-component of dimensional perturbing acceleration
    apyd = ( (3/2)*J2*GMe_hat*Re_hat**2/r**4 ) * (y/r)*(5*(z**2/r**2) -1)  + acc_lunar[1] 
    # z-component of dimensional perturbing acceleration
    apzd = ( (3/2)*J2*GMe_hat*Re_hat**2/r**4 ) * (z/r)*(5*(z**2/r**2) -3)  + acc_lunar[2] 

    # Non-dimensionalise the acceleration components 
    # Ma qui forse non serve questo passaggio giacché le sto anormalizzando prima...
    apx = apxd * (a0d**2/GMe)
    apy = apyd * (a0d**2/GMe)
    apz = apzd * (a0d**2/GMe)
    # Divide by a conventional factor the non-dimensional acceleration components
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
    St = odeint(dromo_keplerian, S0, TSPAN)

    dim1, dim2 = np.shape(St)

    print(np.shape(St)) # (49787, 8)
    # Convert non dimensional time to days 
    t_days = np.empty((dim1, 1))
    for t, i in zip(TSPAN, range(dim1)):
        t_days[i, 0] = ( t * math.sqrt((a0d**3)/GMe) )/(24*3600)
    print(np.shape(t_days))

    # # Dromo elements
    # df = pd.DataFrame(St, columns=["tau", "z1", "z2", "z3", "h1", "h2", "h3", "h4"])
    # fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8) = plt.subplots(8, sharex=True)
    # fig.suptitle('Evolution of Dromo parameters (Keplerian case)')
    # ax1.plot(t_days, df["tau"], 'tab:blue')
    # # ax1.set_title('a')
    # ax1.set(ylabel=r"$\tau$")
    # ax2.plot(t_days, df["z1"], 'tab:blue')
    # # ax2.set_title('e')
    # ax2.set(ylabel=r"$\zeta_1$")
    # ax3.plot(t_days, df["z2"], 'tab:blue')
    # ax3.set(ylabel=r"$\zeta_2$")
    # ax4.plot(t_days, df["z3"], 'tab:blue')
    # ax4.set(ylabel=r"$\zeta_3$")
    # ax5.plot(t_days, df["h1"], 'tab:blue') 
    # ax5.set(ylabel=r"$\eta_1$")
    # ax6.plot(t_days, df["h2"], 'tab:blue') 
    # ax6.set(ylabel=r"$\eta_2$")
    # ax7.plot(t_days, df["h3"], 'tab:blue') 
    # ax7.set(ylabel=r"$\eta_3$")
    # ax8.plot(t_days, df["h4"], 'tab:blue')
    # ax8.set(xlabel=r"$time (days)$", ylabel=r"$\eta_4$")
    # # xlabel="time (days) to last subplot only
    # # Add a grid to subplots
    # ax1.grid()
    # ax2.grid()
    # ax3.grid()
    # ax4.grid()
    # ax5.grid()
    # ax6.grid()
    # ax7.grid()
    # ax8.grid()
    # # Reformat the y axis notation
    # ax1.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.5e'))
    # ax2.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.5e'))
    # ax3.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.5e'))
    # ax4.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.5e'))
    # ax5.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.5e'))
    # ax6.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.5e'))
    # ax7.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.5e'))
    # ax8.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.5e'))
    # plt.show()

    # Orbital elements
    df_orb = pd.DataFrame(dromo2orb_res, columns=["a", "e", "i", "RAAN", "omega", "theta"])
    print(np.shape(dromo2orb_res)) # (10837, 6)
    # fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, sharex=True)
    # fig.suptitle('Evolution of orbital parameters using Dromo formulation (Keplerian)')
    # ax1.plot(t_days, df_orb["a"], 'tab:blue')
    # # ax1.set_title('a')
    # ax1.set(ylabel=r"$a$ (km)")
    # ax2.plot(t_days, df_orb["e"], 'tab:blue')
    # # ax2.set_title('e')
    # ax2.set(ylabel=r"$e$ (-)")
    # ax3.plot(t_days, df_orb["i"], 'tab:blue')
    # ax3.set(ylabel=r"$i$ (rads)")
    # ax4.plot(t_days, df_orb["RAAN"], 'tab:blue')
    # ax4.set(ylabel=r"$\Omega$ (rads)")
    # ax5.plot(t_days, df_orb["omega"], 'tab:blue') 
    # ax5.set(ylabel=r"$\omega$ (rads)")
    # ax6.plot(t_days, df_orb["theta"], 'tab:blue')
    # ax6.set(xlabel=r"$time$ (days)", ylabel=r"$\theta$ (rads)")
    # # xlabel="time (days) to last subplot only
    # # Add a grid to subplots
    # ax1.grid()
    # ax2.grid()
    # ax3.grid()
    # ax4.grid()
    # ax5.grid()
    # ax6.grid()
    # # Reformat the y axis notation
    # ax1.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.5e'))
    # ax2.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.5e'))
    # ax3.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.5e'))
    # ax4.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.5e'))
    # ax5.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.5e'))
    # ax6.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.5e'))
    # plt.show()
    # # elementwise = list(zip(*dromo2orb_res))


if __name__ == "__main__":
    main()