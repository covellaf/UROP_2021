# =============================================================================
# Created By  : Francesca Covella
# Created Date: Tuesday 28 September 2021
# =============================================================================

# Core imports
import numpy as np
from numpy import linalg as la
import pandas as pd
import math
from numpy.lib.function_base import append
from scipy.integrate import solve_ivp
# Sys imports
import time
# Plot imports
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as mtick
# Other imports
from UROP_const import *
from UROP_aux_func import *

# Step 1: Define Initial Conditions (dimensional)
r0 = np.array([0.0, -5888.9727, -3400.0]) #km   (class 'numpy.ndarray')
v0 = np.array([10.691338, 0.0, 0.0])      #km/s
# perigee height should be 6800 km (correct!, as the minimum altitude comes out to be 428 km)
[a0, e0_norm, i0, RAAN0, omega0, theta0] = rv2elorb(r0, v0, GMe)
# from the conversion comes out that a0* = 136,000 km and that the orbital and equatorial plane coincide, as i~0 and RAAN is not defined

# Step 2: non-dimensionalise the ICs
r0nd = r0 / a0 #[km/km] = [-] since the orbit is highly elliptical normalise with the initial semimajor axis
               # otherwise use r0nd = r0/la.norm(r0)
v0nd = v0 * math.sqrt(a0/GMe)    #[-]
t0   = 0                         #s
t0nd = t0 / math.sqrt((a0**3)/GMe) #[-]

# Non-dimensional ICs
S0 = [*r0nd, *v0nd]

# Final time
tf = 288.12768941*24*3600                     # s
tfnd  = tf / math.sqrt((a0**3)/GMe)           #[-]  (2*np.pi * 50:roughly 50 orbits)
delta_t = 500                                 # s (max time beween consecutive integration points)
delta_tnd  = delta_t / math.sqrt((a0**3)/GMe)          #[-]
N = math.floor((tfnd-t0nd)/delta_tnd - 1)
# print(N)
t_eval = np.linspace(0, tfnd, N)               # duration of integration in seconds

# Initial and final time
t_span = np.array([t0nd, tfnd])

# Non-dimensionalise the parameters
GMend = 1 # GMe * a0d / (a0d * GMe) 
Rend  = Re / a0
rlnd  = rl / (6*Re)
omega_lnd = omega_l * math.sqrt((a0**3)/GMe)
GMlnd = GMl / GMe


def relacc_keplerian(t, S):
    """
    relative acceleration (kepleriam motion)
    define the function to be integrated
    """
    x, y, z, x_v, y_v, z_v = S
    r_norm = math.sqrt(x**2 + y**2 + z**2)
    # Gravitational acceleration
    xmu = (-1/(r_norm**3)) * x
    ymu = (-1/(r_norm**3)) * y 
    zmu = (-1/(r_norm**3)) * z

    dSdt = [x_v,
            y_v,
            z_v,
            xmu,
            ymu,
            zmu
            ]
    return dSdt


def relacc_perturbed(t, S):
    """
    relative acceleration (perturbed motion) considering J2 and the third-body effect (Moon)
    
    simplified oblatness model: the radius of the earth changes (decreases) by increasing the
    latitude angle, while a change in longitude angle does not affect the radius, meaning that
    for a given latitude the earth radius is a constant, regardless the longitude.
    define the function to be integrated

    model for third body perturbation taken from Urrutxua paper 2015
    """
    x, y, z, x_v, y_v, z_v = S
    r = np.array([x, y, z])
    r_norm = math.sqrt(x**2 + y**2 + z**2)

    # Gravitational acceleration
    xmu = (-1/(r_norm**3)) * x
    ymu = (-1/(r_norm**3)) * y 
    zmu = (-1/(r_norm**3)) * z

    # J2 acceleration
    xj2 = ( (3/2)*J2*1*Rend**2/r_norm**4 ) * (x/r_norm)*(5*(z**2/r_norm**2) -1) 
    yj2 = ( (3/2)*J2*1*Rend**2/r_norm**4 ) * (y/r_norm)*(5*(z**2/r_norm**2) -1)
    zj2 = ( (3/2)*J2*1*Rend**2/r_norm**4 ) * (z/r_norm)*(5*(z**2/r_norm**2) -3)

    # The lunar position is not defined by an ephemeris model but by trigonometric functions based on the time of the
    # integration. r = [x, y, z] is the satellite position, page 23.
    # The position of the Moon depends explicitly on time (t)
    r3 = np.array([ rlnd * math.sin(omega_lnd*t),
                    rlnd *  (- (math.sqrt(3)*math.cos(omega_lnd*t))/2) ,
                    rlnd *  (- math.cos(omega_lnd*t)/2)])
    xl = GMlnd * ( (r3[0] - x)/(la.norm(r3 - r)**3) - r3[0]/(la.norm(r3)**3) )
    yl = GMlnd * ( (r3[1] - y)/(la.norm(r3 - r)**3) - r3[1]/(la.norm(r3)**3) )
    zl = GMlnd * ( (r3[2] - z)/(la.norm(r3 - r)**3) - r3[2]/(la.norm(r3)**3) )

    dSdt = [x_v,
            y_v,
            z_v,
            xmu + xj2 + xl,
            ymu + yj2 + yl,
            zmu + zj2 + zl
            ]
    return dSdt



def main():
    """
    scipy.integrate.solve_ivp(fun, t_span, y0, 
    method='RK45', t_eval=None, dense_output=False, 
    events=None, vectorized=False, args=None, **options)
    Default values are 1e-3 for rtol and 1e-6 for atol.
    LSODA (super bad), RK45 (really bad), 
    DOP853 (still bad), RK23 (bad bad),
    Radau (kind of), BDF (what? no!!)
    """
    St = solve_ivp(relacc_keplerian, t_span, S0, method="BDF", t_eval=t_eval, events=None, rtol=1e-3, atol=1e-8)

    yout = St.y
    # print(yout[:, 0])
    rnd = yout[:3, :]
    vnd = yout[-3:, :]
    # print(rnd[:, 0])

    dim1, dim2 = np.shape(yout)
    print(dim1, dim2)

    # Dimensionalise the output state
    r = np.empty((3, dim2))
    v = np.empty((3, dim2))
    for col in range(dim2):
        r[:, col] = rnd[:, col] * a0
        v[:, col] = vnd[:, col] / math.sqrt(a0/GMe)
    # print(r[:, 0], r[:, -1], v[:, -1])

    t_days = np.empty((1, dim2))
    for t, i in zip(t_eval, range(dim2)):
        t_days[0, i] = ( t*math.sqrt((a0**3)/GMe) )/(24*3600)
    
    print(t_days[:, -1])

    df_pos = pd.DataFrame(np.transpose(r), columns=["x", "y", "z"])
    print(df_pos.head(10))

    
    ax = plt.axes(projection='3d')
    ax.plot3D(df_pos["x"], df_pos["y"], df_pos["y"], color = 'b', label = 'orbit')
    # ax.set_aspect('equal', 'box')
    plt.show()

    # days = np.transpose(t_days)
    # fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
    # fig.suptitle(r'Evolution of components of position vector \textbf{r} (km)')
    # ax1.plot(days, df_pos["x"], 'tab:green')
    # # ax1.set_title('a')
    # ax1.set(ylabel=r"$x$ (km)")
    # ax2.plot(days, df_pos["y"], 'tab:green')
    # # ax2.set_title('e')
    # ax2.set(ylabel=r"$y$ (km)")
    # ax3.plot(days, df_pos["z"], 'tab:green')
    # ax3.set(xlabel=r"time (days)", ylabel=r"$z$ (km)")
    # # Add a grid to subplots
    # ax1.grid()
    # ax2.grid()
    # ax3.grid()
    # # Reformat the y axis notation
    # ax1.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.5e'))
    # ax2.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.5e'))
    # ax3.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.5e'))
    # plt.show()


if __name__ == "__main__":
    main()
