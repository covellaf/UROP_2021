# =============================================================================
# Created By  : Francesca Covella
# Created Date: Tuesday 21 September 2021
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
about odeint
NOTE
The input parameters rtol and atol determine the error control performed by the solver. 
The solver will control the vector, e, of estimated local errors in y, according to an 
inequality of the form max-norm of (e / ewt) <= 1, where ewt is a vector of positive error 
weights computed as ewt = rtol * abs(y) + atol. rtol and atol can be either vectors 
the same length as y or scalars. Defaults to 1.49012e-8.
"""

# Step 1: Define Initial Conditions (dimensional)
r0d = np.array([0.0, -5888.9727, -3400.0]) #km   (class 'numpy.ndarray')
v0d = np.array([10.691338, 0.0, 0.0])      #km/s
[a0d, e0d, i0d, RAAN0d, omega0d, theta0d] = state2orb(r0d, v0d, GMe)
print([a0d, e0d, i0d, RAAN0d, omega0d, theta0d])
# Step 2: Non-dimensionalise (it is a good practice in order to compare order of magnitude of perturbations
# more meaninfully and also for numerical errors)
r0 = r0d / a0d #[km/km] = [-] since the orbit is highly elliptical normalise with the initial semimajor axis
               # otherwise use r0 = r0d/la.norm(r0d)
v0 = v0d * math.sqrt(a0d/GMe)      #[-]
t0d = 0 #s
t0  = t0d / math.sqrt((a0d**3)/GMe) #[-]
# Non-dimensional ICs
S0 = [*r0, *v0]
# print(S0)

# Non-dimensionalise the parameters
GMe_hat = GMe * a0d / (a0d * GMe) # = 1
Re_hat  = Re / a0d
rl_hat  = rl / (6*Re)
omega_l_hat = omega_l * math.sqrt((a0d**3)/GMe)
GMl_hat = GMl * a0d / (a0d * GMe)
# check if the formula for GMl nd is correct!  

print(Re_hat, rl_hat, omega_l_hat, GMl_hat)
# Final time
tfd = 288.12768941*24*3600                   # s
tf  = tfd / math.sqrt((a0d**3)/GMe)          #[-]
# delta_t = (tf-t0)/(N-1)
delta_td = 500                               # s (max time beween consecutive integration points)
delta_t  = delta_td / math.sqrt((a0d**3)/GMe)          #[-]
N = math.floor((tf-t0)/delta_t - 1)
print(N)
TSPAN = np.linspace(0, tf, N)               # duration of integration in seconds
# Nd = math.floor((tfd-t0d)/delta_td - 1)
# print(Nd)


def relacc_keplerian(S, t):
    """
    relative acceleration (kepleriam motion)
    define the function to be integrated
    """
    x, y, z, x_v, y_v, z_v = S
    r = math.sqrt(x**2 + y**2 + z**2)
    dSdt = [x_v,
            y_v,
            z_v,
            (-GMe_hat/r**3) * x,
            (-GMe_hat/r**3) * y,
            (-GMe_hat/r**3) * z
            ]
    return dSdt


def relacc_with_J2(S, t):
    """
    relative acceleration (perturbed motion) considering the J2 effect
    simplified oblatness model: the radius of the earth changes (decreases) by increasing the
    latitude angle, while a change in longitude angle does not affect the radius, meaning that
    for a given latitude the earth radius is a constant, regardless the longitude.
    define the function to be integrated
    """
    x, y, z, x_v, y_v, z_v = S
    r = math.sqrt(x**2 + y**2 + z**2)
    dSdt_with_J2 = [x_v,
                    y_v,
                    z_v,
                    (-GMe_hat/(r**3)) * x + ( (3/2)*J2*GMe_hat*Re_hat**2/r**4 ) * (x/r)*(5*(z**2/r**2) -1),
                    (-GMe_hat/(r**3)) * y + ( (3/2)*J2*GMe_hat*Re_hat**2/r**4 ) * (y/r)*(5*(z**2/r**2) -1),
                    (-GMe_hat/(r**3)) * z + ( (3/2)*J2*GMe_hat*Re_hat**2/r**4 ) * (z/r)*(5*(z**2/r**2) -3)
                    ]
    return dSdt_with_J2


def relacc_perturbed(S, t):
    """
    relative acceleration (perturbed motion) considering J2 and the third-body effect (Moon)
    model taken from Urrutxua paper 2015
    define the function to be integrated
    """
    x, y, z, x_v, y_v, z_v = S
    r = math.sqrt(x**2 + y**2 + z**2)
    # The lunar position is not defined by an ephemeris model but by trigonometric functions based on the time of the
    # integration. r ([x, y, z]) is the satellite position, page 23.
    iI = np.array([1, 0, 0])
    jI = np.array([0, 1, 0])
    kI = np.array([0, 0, 1])
    # the position of the Moon depends explicitly on time
    r3 = rl_hat * (math.sin(omega_l_hat*t)*iI - math.cos(omega_l_hat*t)/2 * (math.sqrt(3)*jI + kI))
    acc_lunar = GMl_hat * ( (r3 - np.array([x, y, z]))/la.norm(r3 - np.array([x, y, z]))**3 - r3/la.norm(r3)**3)

    dSdt_pert = [x_v,
                y_v,
                z_v,
                (-GMe_hat/(r**3)) * x + ( (3/2)*J2*GMe_hat*Re_hat**2/r**4 ) * (x/r)*(5*(z**2/r**2) -1) + acc_lunar[0],
                (-GMe_hat/(r**3)) * y + ( (3/2)*J2*GMe_hat*Re_hat**2/r**4 ) * (y/r)*(5*(z**2/r**2) -1) + acc_lunar[1] ,
                (-GMe_hat/(r**3)) * z + ( (3/2)*J2*GMe_hat*Re_hat**2/r**4 ) * (z/r)*(5*(z**2/r**2) -3) + acc_lunar[2] 
                ]
    return dSdt_pert


def main():
    """
    Functions:
    scipy.integrate.odeint(func, y0, t, args=(), ...)
    Integrate a system of ordinary differential equations.
    Outputs:
    some interesting facts about the orbit plotted
    """
    St = odeint(relacc_perturbed, S0, TSPAN)
    # np.savetxt("/Users/utente73/Desktop/cowell_keplerian_0.txt", St)
    pos = St[:, :3]
    vel = St[:, -3:]
    dim1, dim2 = np.shape(St)

    # Convert to classical elements the dimensionalised state
    state2orb_res = []
    for item in range(dim1):
        state2orb_res.append(state2orb(pos[item]*a0d, vel[item]/math.sqrt(a0d/GMe), GMe))
    # print(np.shape(state2orb_res))

    t_days = np.empty((dim1, 1))
    for t, i in zip(TSPAN, range(dim1)):
        t_days[i, 0] = ( t * math.sqrt((a0d**3)/GMe) )/(24*3600)

    df = pd.DataFrame(state2orb_res, columns=["a", "e", "i", "RAAN", "omega", "theta"])
    fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, sharex=True)
    fig.suptitle('Evolution of orbital parameters using Cowell formulation (J2+Moon)')
    ax1.plot(t_days, df["a"], 'tab:green')
    # ax1.set_title('a')
    ax1.set(ylabel=r"$a$ (km)")
    ax2.plot(t_days, df["e"], 'tab:green')
    # ax2.set_title('e')
    ax2.set(ylabel=r"$e$ (-)")
    ax3.plot(t_days, df["i"], 'tab:green')
    ax3.set(ylabel=r"$i$ (rads)")
    ax4.plot(t_days, df["RAAN"], 'tab:green')
    ax4.set(ylabel=r"$\Omega$ (rads)")
    ax5.plot(t_days, df["omega"], 'tab:green') 
    ax5.set(ylabel=r"$\omega$ (rads)")
    ax6.plot(t_days, df["theta"], 'tab:green')
    ax6.set(xlabel=r"$time$ (days)", ylabel=r"$\theta$ (rads)")
    # xlabel="time (days) to last subplot only
    # Add a grid to subplots
    ax1.grid()
    ax2.grid()
    ax3.grid()
    ax4.grid()
    ax5.grid()
    ax6.grid()
    # Reformat the y axis notation
    ax1.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.5e'))
    ax2.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.5e'))
    ax3.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.5e'))
    ax4.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.5e'))
    ax5.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.5e'))
    ax6.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.5e'))
    plt.show()


if __name__ == "__main__":
    main()
