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
print([a0, e0_norm, i0, RAAN0, omega0, theta0])
# from the conversion comes out that a0* = 136,000 km and that the orbital and equatorial plane coincide, as i~0 and RAAN is not defined

# Step 2: non-dimensionalise the ICs
r0nd = r0 / a0 #[km/km] = [-] since the orbit is highly elliptical normalise with the initial semimajor axis
               # otherwise use r0nd = r0/la.norm(r0)
v0nd = v0 * math.sqrt(a0/GMe)    #[-]
t0   = 0                         #s
t0nd = t0 / math.sqrt((a0**3)/GMe) #[-]

# Step 3: tranform the non-dimensional ICs (r0nd, v0nd) in DROMO elements 
h0 = np.cross(r0nd, v0nd)                      # 3-components vector
e0 = - r0nd/la.norm(r0nd) - np.cross(h0, v0nd) # 3-components vector
# sigma0 from page 7. and from page 14. (initial conditions)
sigma0 = theta0  

# Initial state (sigma; tau, zeta1, zeta2, zeta3, eta1, eta2, eta3, eta4)
tau_0   = t0nd
zeta1_0 = la.norm(e0)
zeta2_0 = 0
zeta3_0 = 1/la.norm(h0)
eta1_0  = math.sin(i0/2)*math.cos((RAAN0-omega0)/2)
eta2_0  = math.sin(i0/2)*math.sin((RAAN0-omega0)/2)
eta3_0  = math.cos(i0/2)*math.sin((RAAN0+omega0)/2)
eta4_0  = math.cos(i0/2)*math.cos((RAAN0+omega0)/2)
S0 = [tau_0, zeta1_0, zeta2_0, zeta3_0, eta1_0, eta2_0, eta3_0, eta4_0]

# Final time
fin_day = 288.12768941
tf = fin_day*24*3600                     # s
tfnd  = tf / math.sqrt((a0**3)/GMe)           #[-]  (2*np.pi * 50:roughly 50 orbits)
delta_t = 300                                 # s (max time beween consecutive integration points)
delta_tnd = delta_t / math.sqrt((a0**3)/GMe)          #[-]
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


def event(sigma, State):
    tau, zeta1, zeta2, zeta3, eta1, eta2, eta3, eta4 = State
    zero = ((tau * math.sqrt((a0**3)/GMe))/(24*3600)) - fin_day 
    print(zero)

    
    return zero #((tau * math.sqrt((a0**3)/GMe))/(24*3600)) - fin_day  #tau - tfnd


def dromo_keplerian(sigma, State):
    """
    Equations to be integrated
    Propagate DROMO EOMs using odeint
    """
    tau, zeta1, zeta2, zeta3, eta1, eta2, eta3, eta4 = State
    # Auxiliary eq.
    s = 1 + zeta1 * math.cos(sigma) + zeta2 * math.sin(sigma)
    # Keplerian motion
    a_px = 0 
    a_py = 0
    a_pz = 0
    # EOMs
    dSdsigma = [1/((zeta3**3) * (s**2)), 
                s * math.sin(sigma)   * a_px   + (zeta1 + (1+s)*math.cos(sigma)) * a_py, 
                - s * math.cos(sigma) * a_px   + (zeta2 + (1+s)*math.sin(sigma)) * a_py,
                - zeta3 * a_pz,
                1/2 * a_pz * (eta4 * math.cos(sigma) - eta3 * math.sin(sigma)),
                1/2 * a_pz * (eta3 * math.cos(sigma) + eta4 * math.sin(sigma)),
                1/2 * a_pz * (-eta2* math.cos(sigma) + eta1 * math.sin(sigma)),
                1/2 * a_pz * (-eta1* math.cos(sigma) - eta2 * math.sin(sigma))
                ]
    return dSdsigma


def main():
    """
    scipy.integrate.solve_ivp(fun, t_span, y0, 
    method='RK45', t_eval=None, dense_output=False, 
    events=None, vectorized=False, args=None, **options)
    Default values are 1e-3 for rtol and 1e-6 for atol.
    LSODA (160), RK45 (205), 
    DOP853 (265), RK23 (80),
    Radau (160), BDF (42)
    outputs:
    t_events: list of ndarray or None
    Contains for each event type a list of arrays at which an event of 
    that type event was detected. None if events was None.
    y_events: list of ndarray or None
    For each value of t_events, the corresponding value of the solution. 
    None if events was None.
    """
    # solve ivp con tolleranza
    St = solve_ivp(dromo_keplerian, t_span, S0, method="DOP853", t_eval=t_eval, 
                    events=[event], rtol=1e-3, atol=1e-8)
    event.terminal = True
    yout = St.y
    dim1, dim2 = np.shape(yout)
    print(dim1, dim2)
    df_dromo = pd.DataFrame(np.transpose(yout), columns=["tau", "z1", "z2", "z3", "h1", "h2", "h3", "h4"])
    # print(df_dromo.head(10))

    np.savetxt("/Users/utente73/Desktop/dromo_tau_nd2.txt", df_dromo["tau"])
    
    print(t_eval[0], yout[:, 0])
    
    r = np.empty((3, dim2))
    v = np.empty((3, dim2))
    for col in range(dim2):
        r[:, col], v[:, col] = dromo2rv(a0, t_eval[col], *yout[:, col])[:3], dromo2rv(a0, t_eval[col], *yout[:, col])[-3:]
    print("#######################")
    print(r[:, 0], v[:, 0], r[:, -1], v[:, -1])

    tau = np.empty((1, dim2))
    for t, i in zip(df_dromo["tau"], range(dim2)):
        tau[0, i] = ( t*math.sqrt((a0**3)/GMe) )/(24*3600)
    np.savetxt("/Users/utente73/Desktop/dromo_tau_days2.txt", tau)
    print(tau[:, -1])


if __name__ == "__main__":
    main()


# Very dependent on rtol, atol, integration method, hard to compare results
# also odeint seems very dependent on rtol, atol
# 