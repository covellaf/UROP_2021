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
# print([a0, e0_norm, i0, RAAN0, omega0, theta0])
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
print(S0)

# Test
# T = 2*np.pi * np.sqrt(a0**3/GMe) # orbital period
# fin_day = (10.5*T)/(24*3600) # days
# print(f"fin day = {fin_day} in days = {10.5*T} s, adimensional = {(10.5*T)/np.sqrt(a0**3/GMe)} ")
# # Valore finale di sigma
# sigma_fin = 50*2*np.pi
# # numero di passi per orbita
# m = 500 # 1000
# N = math.floor((sigma_fin - sigma0)/(2*np.pi)) * m
# # print("N: ", N)
# sigma_span = np.array([sigma0, sigma_fin])
# t_eval = np.linspace(sigma_span[0], sigma_span[-1], N)

# Valore finale di sigma
sigma_fin = 60*2*np.pi
# Final time
fin_day =  288.1276894125  #24894232.36524/(24*3600) one orbital period 5.777065623908596 # days
# Initial and final values of sigma
sigma_span = np.array([sigma0, sigma_fin])
delta_sigma = 1                                 # rad
N = math.floor((sigma_fin-sigma0)/delta_sigma - 1)
print(N)
sigma_eval = np.linspace(sigma_span[0], sigma_span[-1], N)               # duration of integration in seconds

# Non-dimensionalise the parameters
GMend = 1 # GMe * a0d / (a0d * GMe) 
Rend  = Re / a0
rlnd  = rl / a0
omega_lnd = omega_l * math.sqrt((a0**3)/GMe)
GMlnd = GMl / GMe


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


def event(sigma, State):
    tau, zeta1, zeta2, zeta3, eta1, eta2, eta3, eta4 = State
    # event.direction = 0
    return ((tau * math.sqrt((a0**3)/GMe))/(24*3600)) - fin_day 
event.terminal = True
event.direction = 0

def dromo_perturbed(sigma, State):
    """
    Equations to be integrated
    Propagate DROMO EOMs using odeint
    """
    tau, zeta1, zeta2, zeta3, eta1, eta2, eta3, eta4 = State
    # Auxiliary eq.
    s = 1 + zeta1 * math.cos(sigma) + zeta2 * math.sin(sigma)
    # Dromo elements to (Dimensional) cartesian representation of state vector (page 14)  
    x,  y,  z, xv, yv, zv = dromo2rv(a0, sigma, tau, zeta1, zeta2, zeta3, eta1, eta2, eta3, eta4)
    # Create vectors
    r = np.array([x,  y,  z])
    # print(r.shape) (3, 1)
    v = np.array([xv, yv, zv])
    r_norm = la.norm(r)        
    # Define the unit vectors of the (local) orbital frame
    io = r / r_norm
    ko = np.cross(r, v, axis=0) / la.norm(np.cross(r, v, axis=0))
    jo = np.cross(ko, io, axis=0)
    xx, xy, xz = io
    yx, yy, yz = jo
    zx, zy, zz = ko

    # J2 acceleration
    xj2 = ( (3/2)*J2*1*Rend**2/r_norm**4 ) * (x/r_norm)*(5*(z**2/r_norm**2) -1) 
    yj2 = ( (3/2)*J2*1*Rend**2/r_norm**4 ) * (y/r_norm)*(5*(z**2/r_norm**2) -1)
    zj2 = ( (3/2)*J2*1*Rend**2/r_norm**4 ) * (z/r_norm)*(5*(z**2/r_norm**2) -3)

    # Perturbation due to Third body (Moon)
    r3 = np.array([ rlnd * math.sin(omega_lnd*tau),
                    rlnd *  (- (math.sqrt(3)*math.cos(omega_lnd*tau))/2) ,
                    rlnd *  (- math.cos(omega_lnd*tau)/2)])

    xl = GMlnd * ( (r3[0] - x)/(la.norm(r3 - r)**3) - r3[0]/(la.norm(r3)**3) )
    yl = GMlnd * ( (r3[1] - y)/(la.norm(r3 - r)**3) - r3[1]/(la.norm(r3)**3) )
    zl = GMlnd * ( (r3[2] - z)/(la.norm(r3 - r)**3) - r3[2]/(la.norm(r3)**3) ) 

    # Superimpose the accelerations
    apx = xj2 + xl
    apy = yj2 + yl
    apz = zj2 + zl

    # Project into the orbital frame (from inertial frame)
    apxo = apx*xx + apy*yx + apz*zx
    apyo = apx*xy + apy*yy + apz*zy
    apzo = apx*xz + apy*yz + apz*zz

    # Divide by a conventional factor the non-dimensional acceleration components
    a_px = apxo/(zeta3**4 * s**3)  # x-component of perturbing acceleration
    a_py = apyo/(zeta3**4 * s**3)  # y-component of perturbing acceleration
    a_pz = apzo/(zeta3**4 * s**3)  # z-component of perturbing acceleration

    # EOMs
    dSdsigma = [1/(zeta3**3 * s**2), 
                s * math.sin(sigma) * a_px   + (zeta1 + (1+s)*math.cos(sigma)) * a_py, 
                - s * math.cos(sigma) * a_px + (zeta2 + (1+s)*math.sin(sigma)) * a_py,
                - zeta3 * a_pz,
                1/2 * a_pz * (eta4 * math.cos(sigma) - eta3 * math.sin(sigma)),
                1/2 * a_pz * (eta3 * math.cos(sigma) + eta4 * math.sin(sigma)),
                1/2 * a_pz * (-eta2* math.cos(sigma) + eta1 * math.sin(sigma)),
                1/2 * a_pz * (-eta1* math.cos(sigma) - eta2 * math.sin(sigma))
                ]
    return dSdsigma


def main():
    """
    scipy.integrate.solve_ivp(fun, sigma_span, y0, 
    method='RK45', sigma_eval=None, dense_output=False, 
    events=None, vectorized=False, args=None, **options)
    Default values are 1e-3 for rtol and 1e-6 for atol.
    LSODA, RK45, DOP853, RK23, Radau, BDF
    outputs:
    t_events: list of ndarray or None
    Contains for each event type a list of arrays at which an event of 
    that type event was detected. None if events was None.
    y_events: list of ndarray or None
    For each value of t_events, the corresponding value of the solution. 
    None if events was None.
    """
    # solve ivp con tolleranza
    St = solve_ivp(dromo_perturbed, sigma_span, S0, method="RK45", t_eval=sigma_eval, 
                    events=event, dense_output=True, rtol=1e-13, atol=1e-13)
             
    print("The integration was: ", St.status) # 0: The solver successfully reached the interval end.
                                              # 1: A termination event occurred.
                                              # -1: Integration step failed.
    tout = St.t
    yout = St.y
    tevent = St.t_events
    print("final sigma (rad): ", tevent[0][0])
    ### ???
    # tevent (keplerian case = ) 311.2896081346541
    # tevent (perturbed case = ) 304.6235503622309
    yevent = St.y_events
    print("event day: ", yevent[0][0][0]* math.sqrt((a0**3)/GMe)/(24*3600) )

    # revent = dromo2rv(a0, tevent[0][0], *yevent[0][0][:])[:3]
    # vevent = dromo2rv(a0, tevent[0][0], *yevent[0][0][:])[-3:]
    # rp = r0/la.norm(r0)
    # ra = revent/la.norm(revent)
    status_event = dromo2rv(a0, tevent[0][0], *yevent[0][0][:])
    print("final state: ", status_event)

    dim1, dim2 = np.shape(yout)
    # print(dim1, dim2)
    df_dromo = pd.DataFrame(np.transpose(yout), columns=["tau", "z1", "z2", "z3", "h1", "h2", "h3", "h4"])
    # print(df_dromo.head(10))
    
    state = np.empty((6, dim2))
    # r = np.empty((3, dim2))
    # v = np.empty((3, dim2))
    for col in range(dim2):
        state[:, col] = dromo2rv(a0, sigma_eval[col], *yout[:, col])
    # r = state[:, :3]
    # v = state[:, -3:] 
    print("initial state", state[:, 0])
    print("final state", state[:, -1])

    tau = np.empty((1, dim2))
    for t, i in zip(df_dromo["tau"], range(dim2)):
        tau[0, i] = ( t*math.sqrt((a0**3)/GMe) )/(24*3600)
    print("tau finale", tau[:, -1])
    
    # df_pos = pd.DataFrame(np.transpose(r), columns=["x", "y", "z"])
    # print(df_pos.head())

    # ax = plt.axes(projection='3d')
    # ax.plot3D(df_pos["x"], df_pos["y"], df_pos["y"], color = 'k', label = 'orbit')
    # # ax.set_aspect('equal', 'box')
    # plt.show()

    # plt.plot(tau, df_pos["x"], 'tab:green')
    # plt.show()

    # fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
    # fig.suptitle(r'Evolution of components of position vector \textbf{r} (km)')
    # ax1.plot(tau, df_pos["x"], 'tab:green')
    # ax1.set(ylabel=r"$x$ (km)")
    # ax1.legend()
    # ax2.plot(tau, df_pos["y"], 'tab:green')
    # ax2.set(ylabel=r"$y$ (km)")
    # ax2.legend()
    # ax3.plot(tau, df_pos["z"], 'tab:green')
    # ax3.set(ylabel=r"$z$ (km)")
    # ax3.legend()
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


# Very dependent on rtol, atol, integration method, hard to compare results
# also odeint seems very dependent on rtol, atol
