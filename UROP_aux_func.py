# =============================================================================
# Created By  : Francesca Covella
# Created Date: Tuesday 21 September 2021
# =============================================================================

"""
Auxiliary functions
"""

import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
# Other imports
from UROP_const import *


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


def orb2state(a, e_norm, i, RAAN, arg_perigee, true_anomaly):
    """
    This function takes as input the 6 orbital parameters and returns a state vector of the position and velocity
    a: semi-major axis (km)
    e_norm: the norm of the eccentricity vector
    i: inclination (rad)
    RAAN: right ascension of the ascending node (deg)
    arg_perigee: argument of perigee (deg)
    true_anomaly: theta (deg)
    """
    if e_norm < math.pow(10, -4):
        p = a
    elif e_norm > math.pow(10, -4) and e_norm < 1:
        b = a*math.sqrt(1 - e_norm**2)
        p = b**2/a
    elif e_norm == 1:
        p = 2*a
    elif e_norm > 1:
        b = a*math.sqrt(e_norm**2 - 1)
        p = b**2/a
    # p = h**2 / MU_earth
    h_norm = math.sqrt(p*GMe) 

    R3_Om = np.array( [[math.cos(RAAN), math.sin(RAAN), 0], [-math.sin(RAAN), math.cos(RAAN), 0], [0, 0, 1]] )
    R1_i  = np.array( [[1, 0, 0], [0, math.cos(i), math.sin(i)], [0, -math.sin(i), math.cos(i)]] )
    R3_om = np.array( [[math.cos(arg_perigee), math.sin(arg_perigee), 0], [-math.sin(arg_perigee), math.cos(arg_perigee), 0], [0, 0, 1]] )
    support_var = R3_om.dot(R1_i).dot(R3_Om)
    x = support_var[0, :]
    y = support_var[1, :]
    z = support_var[2, :]
    e_orb = e_norm * x
    h_orb = h_norm * z
    r_norm = (h_norm**2/GMe) * (1/(1+e_norm*math.cos(true_anomaly)))
    r_orb = r_norm*math.cos(true_anomaly)*x + r_norm*math.sin(true_anomaly)*y 
    u_radial = r_orb/r_norm
    u_normal = np.transpose( np.cross(z, u_radial)/np.linalg.norm(np.cross(z, u_radial)) )
    v_orb = (GMe/h_norm) * e_norm * math.sin(true_anomaly) * u_radial + (GMe/h_norm) * (1+e_norm*math.cos(true_anomaly)) * u_normal
    return r_orb, v_orb


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


# def Dromotime2time(tau):
#     # time in s
#     # page 14. from paper
#     time = t0 + tau * math.sqrt(a0d**3/GMe)
#     return time


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


# For function testing, to be commented
# def main():
#     Dromo2orbel()

# if __name__ == "__main__":
#     main()
