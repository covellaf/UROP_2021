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

# Other imports
# import dromo_const as const
# from dromo_func import state2orb
#from astroplotlib import plot2d_x_ys

# Constants
GMe = 398601   #[km^3/sec^2]
Re  = 6371.22  #[km]
J2 = 0.00108265         # [-] second zonal harmonic
GMl = 4902.66 #[km^3/sec^2] for the Moon
rl = 384400   #[km] radius of the lunar orbit
omega_l = 2.665315780887e-6 #[1/s] lunar orbital angular velocity

# Auxiliary func
def state2orb(r0, v0, Gparam):
    """
    Converts from state vector to orbital parameters
    units:
    r0 in km
    v0 in km/s
    Gparam in km^3/s^2
    a in km
    e is [-]
    angles in radiants
    """
    h = np.cross(r0, v0)
    h_norm = np.linalg.norm(h)
    cos_inclination = h[2]/h_norm       # since h scalar product z = h_norm*1*cos(i) = h_3

    if np.linalg.norm(cos_inclination) >= 1:
        cos_inclination = np.sign(cos_inclination)
    inclination = math.acos(cos_inclination)

    if inclination == 0 or inclination == np.pi :
        node_line = [1, 0, 0] # None  # pick the x-axis as your line of Nodes, which is undefined as the orbital and equatorial plane coincide
        RAAN = 0  # None 
    else :
        node_line = np.cross([0, 0, 1], h)/(np.linalg.norm(np.cross([0, 0, 1], h))) # cross vector is not commutative
        cos_RAAN = node_line[0]
        if np.linalg.norm(cos_RAAN) >= 1:
            cos_RAAN = np.sign(cos_RAAN)
        RAAN = math.acos(cos_RAAN)
    if node_line[1] < 0:
        RAAN = 2*np.pi - RAAN

    # From the Laplace vector equation 
    e = (np.cross(v0, h))/Gparam - r0/np.linalg.norm(r0)
    e_norm = np.linalg.norm(e)

    if e_norm < math.pow(10, -5):
        # for circular orbits choose r0 as the apse line to count the true anomaly and define the argument of perigee
        cos_arg_perigee = np.dot(r0, node_line)/np.linalg.norm(r0)
        if np.linalg.norm(cos_arg_perigee) >= 1:
            cos_arg_perigee = np.sign(cos_arg_perigee)
        arg_perigee = math.acos(cos_arg_perigee)
        if r0[2] < 0:
            arg_perigee = 2*np.pi - arg_perigee
        # arg_perigee =  # None 
    else :
        cos_arg_perigee = np.dot(e, node_line)/e_norm
        if np.linalg.norm(cos_arg_perigee) >= 1:
            cos_arg_perigee = np.sign(cos_arg_perigee)
        arg_perigee = math.acos(cos_arg_perigee)
        if e[2] < 0: # e1,e2,e3 dot 0,0,1
            arg_perigee = 2*np.pi - arg_perigee

    perigee = (np.linalg.norm(h)**2/Gparam) * (1/(1+e_norm))
    apogee  = (np.linalg.norm(h)**2/Gparam) * (1/(1-e_norm))

    if apogee < 0:
        # in the case of an hyperbolic orbit
        apogee = - apogee

    semi_major_axis = (perigee+apogee)/2
    T = (2*np.pi/math.sqrt(Gparam)) * math.pow(semi_major_axis, 3/2)  # orbital period (s)

    if e_norm < math.pow(10, -5):
        true_anomaly = 0
    else :
        cos_true_anomaly = np.dot(e, r0)/(e_norm*np.linalg.norm(r0))
        if np.linalg.norm(cos_true_anomaly) >= 1:
            cos_true_anomaly = np.sign(cos_true_anomaly)
        true_anomaly = math.acos(cos_true_anomaly)

    u_r  = r0/np.linalg.norm(r0)
    if np.dot(v0, u_r) < 0:
        # past apogee
        true_anomaly = 2*np.pi - true_anomaly   
        true_anomaly_r = math.radians(true_anomaly)
    return semi_major_axis, e_norm, inclination, RAAN, arg_perigee, true_anomaly


def plot2d_x_ys(x, ys, line_colors, h_label, v_label, line_styles, line_widths,\
                labels=None, style='seaborn-paper', markers=None):
    """
    Can plot on the same x axis more than one curve, to perform comparisons
    Inputs:
    x: on x axis
    ys: a list for the y axis
    style: 'Solarize_Light2', '_classic_test_patch', 'bmh', 'classic', 'dark_background', 
    'fast', 'fivethirtyeight', 'ggplot', 'seaborn', 'seaborn-bright', 'seaborn-paper', 'seaborn-pastel', 
    'seaborn-colorblind','seaborn-darkgrid', 'seaborn-deep', 'seaborn-muted', 'seaborn-notebook', 
    'seaborn-poster','seaborn-talk', 'seaborn-ticks', 'seaborn-white', 'seaborn-whitegrid'
    """
    plt.figure()
    plt.style.use(style)
    if markers == None:
        markers = ['']*len(ys)
    if labels == None:
        labels = ['']*len(ys)
    else:
        plt.legend()
    for idx in range(len(ys)):
        plt.plot(x, 
                ys[idx], 
                color = line_colors[idx],
                linestyle = line_styles[idx], 
                linewidth = line_widths[idx],
                marker = markers[idx],
                label = labels[idx]
                )
    plt.grid(True)
    plt.xlabel(h_label)
    plt.ylabel(v_label)
    plt.show()


# Step 1: Define Initial Conditions (dimensional)
r0 = np.array([0.0, -5888.9727, -3400.0]) #km   (class 'numpy.ndarray')
v0 = np.array([10.691338, 0.0, 0.0])      #km/s

S0 = [*r0, *v0]
t0 = 0
tf = 288.12768941*24*3600                   # s
# delta_t = (tf-t0)/(N-1)
delta_t = 50                                # s (max time beween consecutive integration points)
N = math.floor((tf-t0)/delta_t - 1)
TSPAN = np.linspace(0, tf, N)               # duration of integration in seconds
print(N)


def relacc(S, t):
    """
    relative acceleration (kepleriam motion)
    define the function to be integrated
    """
    x, y, z, x_v, y_v, z_v = S
    dSdt = [x_v,
            y_v,
            z_v,
            (-GMe/((math.sqrt(x**2 + y**2 + z**2))**3)) * x,
            (-GMe/((math.sqrt(x**2 + y**2 + z**2))**3)) * y,
            (-GMe/((math.sqrt(x**2 + y**2 + z**2))**3)) * z
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
                    (-GMe/(r**3)) * x + ( (3/2)*J2*GMe*Re**2/r**4 ) * (x/r)*(5*(z**2/r**2) -1),
                    (-GMe/(r**3)) * y + ( (3/2)*J2*GMe*Re**2/r**4 ) * (y/r)*(5*(z**2/r**2) -1),
                    (-GMe/(r**3)) * z + ( (3/2)*J2*GMe*Re**2/r**4 ) * (z/r)*(5*(z**2/r**2) -3)
                    ]
    return dSdt_with_J2


# def relacc_with_third_body(S, t):
#     """
#     relative acceleration (perturbed motion) considering the third-body effect (Moon)
#     model taken from Urrutxua paper 2015
#     define the function to be integrated
#     """
#     x, y, z, x_v, y_v, z_v = S
#     dSdt_with_third_body = [x_v,
#                             y_v,
#                             z_v,
#                             ...,
#                             ...,
#                             ...
#                             ]
#     return dSdt_with_third_body


def main():
    """
    Functions:
    scipy.integrate.odeint(func, y0, t, args=(), ...)
    Integrate a system of ordinary differential equations.
    Outputs:
    some interesting facts about the orbit plotted
    """
    # St = odeint(relacc, S0, TSPAN)
    # pos = St[:, :3]
    # vel = St[:, -3:]
    St_with_J2 = odeint(relacc_with_J2, S0, TSPAN)
    pos_with_J2 = St_with_J2[:, :3]
    vel_with_J2 = St_with_J2[:, -3:]
    # print(St)
    # dim1, dim2 = np.shape(St)
    # a_out = np.empty((1, dim1))
    # enorm = np.empty((1, dim1))
    # inclination = np.empty((1, dim1))
    # RAAN = np.empty((1, dim1))
    # arg_per = np.empty((1, dim1))
    # true_anomaly = np.empty((1, dim1))
    # print("here")
    # for i in range(dim1):
    #     a_out[0, i], enorm[0, i], inclination[0, i], RAAN[0, i], arg_per[0, i], true_anomaly[0, i] = state2orb(St[i,:3], St[i,-3:], GMe)
    # #print(a_out[0, :], enorm[0, i], inclination[0, :], RAAN[0, :], arg_per[0, :], true_anomaly[0, :] )
    # t_days = np.empty((1, dim1))
    # for t, i in zip(TSPAN, range(dim1)):
    #     t_days[0, i] = t/(24*3600)
    # inclination_deg = np.empty((1, dim1))
    # RAAN_deg = np.empty((1, dim1))
    # arg_per_deg = np.empty((1, dim1))
    # for i in range(dim1):
    #     inclination_deg[0, i] = inclination[0, i] * 180/np.pi
    #     RAAN_deg[0, i] = RAAN[0, i] * 180/np.pi
    #     arg_per_deg[0, i] = arg_per[0, i] * 180/np.pi

    # plot2d_x_ys(t_days[0,:], [a_out[0,:]], ['blue'], 'time (days)', 'semi-major axis (km)', ['-'], ['2'])
    # plot2d_x_ys(t_days[0,:], [enorm[0,:]], ['blue'], 'time (days)', 'eccentricity', ['-'], ['2'])
    # plot2d_x_ys(t_days[0,:], [inclination_deg[0,:]], ['blue'], 'time (days)', 'inclination (deg)', ['-'], ['2'])
    # plot2d_x_ys(t_days[0,:], [RAAN_deg[0,:]], ['blue'], 'time (days)', 'RAAN (deg)', ['-'], ['2'])
    # plot2d_x_ys(t_days[0,:], [arg_per_deg[0,:]], ['blue'], 'time (days)', 'argument of perigee (deg)', ['-'], ['2'])
    # plot2d_x_ys(t_days[0,:], [true_anomaly[0,:]], ['blue'], 'time (days)', 'true anomaly (radians)', ['-'], ['2'])

    dim1_J2, dim2_J2 = np.shape(St_with_J2)
    a_out_J2 = np.empty((1, dim1_J2))
    enorm_J2 = np.empty((1, dim1_J2))
    inclination_J2 = np.empty((1, dim1_J2))
    RAAN_J2 = np.empty((1, dim1_J2))
    arg_per_J2 = np.empty((1, dim1_J2))
    true_anomaly_J2 = np.empty((1, dim1_J2))
    for i in range(dim1_J2):
        a_out_J2[0, i], enorm_J2[0, i], inclination_J2[0, i], RAAN_J2[0, i], arg_per_J2[0, i], true_anomaly_J2[0, i] = state2orb(St_with_J2[i,:3], St_with_J2[i,-3:], GMe)

    t_days = np.empty((1, dim1_J2))
    for t, i in zip(TSPAN, range(dim1_J2)):
        t_days[0, i] = t/(24*3600)

    inclination_deg_J2 = np.empty((1, dim1_J2))
    RAAN_deg_J2 = np.empty((1, dim1_J2))
    arg_per_deg_J2 = np.empty((1, dim1_J2))
    for i in range(dim1_J2):
        inclination_deg_J2[0, i] = inclination_J2[0, i] * 180/np.pi
        RAAN_deg_J2[0, i] = RAAN_J2[0, i] * 180/np.pi
        arg_per_deg_J2[0, i] = arg_per_J2[0, i] * 180/np.pi

    plot2d_x_ys(t_days[0,:], [a_out_J2[0,:]], ['red'], 'time (days)', 'semi-major axis (km)', ['-'], ['2'])
    plot2d_x_ys(t_days[0,:], [enorm_J2[0,:]], ['red'], 'time (days)', 'eccentricity', ['-'], ['2'])
    plot2d_x_ys(t_days[0,:], [inclination_deg_J2[0,:]], ['red'], 'time (days)', 'inclination (deg)', ['-'], ['2'])
    plot2d_x_ys(t_days[0,:], [RAAN_deg_J2[0,:]], ['red'], 'time (days)', 'RAAN (deg)', ['-'], ['2'])
    plot2d_x_ys(t_days[0,:], [arg_per_deg_J2[0,:]], ['red'], 'time (days)', 'argument of perigee (deg)', ['-'], ['2'])
    plot2d_x_ys(t_days[0,:], [true_anomaly_J2[0,:]], ['red'], 'time (days)', 'true anomaly (radians)', ['-'], ['2'])


if __name__ == "__main__":
    main()
