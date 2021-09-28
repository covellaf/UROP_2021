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


def rv2elorb(R, V, mu):
    """
    Programmed by: Claudio Bombardelli and Giulio Bau' (UPM & UniPD)
    Date:                  6/03/2010
    Revision:              1
    Tested by:
     Computes the inertial position and velocity given the orbital elements
    
     INPUT:     R = inertial position   (N x 3)
                V = inertial velocity   (N x 3)
                mu = grav constant  
    
     OUTPUT:   elorb = [a ecc inc raan a_per theta0]  (N x 6)
           
               where  a = pericenter radius     (N x 1)
                      ecc = eccentricity        (N x 1)
                      inc = inclination (rad)   (N x 1)
                      raan = right ascension of ascending node (rad)  (N x 1)
                      a_per = argument of perigee (rad)               (N x 1)
                      theta0 = true anomaly at epoch                  (N x 1)
    Adapted for Python by: Francesca Covella (Imperial College London)
    Date:                  24/09/2021
    """

    # POSITION
    # - vector
    X = R[0] 
    Y = R[1]
    Z = R[2]
    # - magnitiude
    r = (X**2 + Y**2 + Z**2)**(0.5)
    # - direction
    urx = np.divide(X, r)
    ury = np.divide(Y, r) 
    urz = np.divide(Z, r) 

    # VELOCITY
    # - vector
    Xd = V[0] 
    Yd = V[1] 
    Zd = V[2] 
    # - magnitude
    v = (Xd**2 + Yd**2 + Zd**2)**(0.5)
    # ENERGY
    En = 1/2*v**2 - mu/r

    # ANGULAR MOMENTUM VECTOR
    hx = Y*Zd - Z*Yd
    hy = Z*Xd - X*Zd
    hz = X*Yd - Y*Xd
    hmod = (hx**2 + hy**2 + hz**2)**(1/2)
    uhx = hx/hmod
    uhy = hy/hmod
    uhz = hz/hmod

    # ECCENTRICITY VECTOR
    vXhx = Yd*hz - Zd*hy
    vXhy = Zd*hx - Xd*hz
    vXhz = Xd*hy - Yd*hx

    ex = vXhx/mu - X/r
    ey = vXhy/mu - Y/r
    ez = vXhz/mu - Z/r

    # - vector
    ex = (1 - (np.abs(ex) <= np.spacing(1) ))*ex
    ey = (1 - (np.abs(ey) <= np.spacing(1) ))*ey
    ez = (1 - (np.abs(ez) <= np.spacing(1) ))*ez
    # - magnitude
    emod = (ex**2 + ey**2 + ez**2)**(1/2)

    # auxiliary quantities
    cr1 = np.sign(emod)   # The ith element of cr1 is:
                        # '0' if the orbit IS circular
                        # '1' if the orbit IS NOT circular
    cr2 = 1. - cr1     # The ith element of cr2 is
                        # '1' if the orbit IS circular
                        # '0' if the orbit IS NOT circular
    emod_aux2 = emod + cr2

    # - direction
    # uex, uey, uez are equal to 0 if the orbit is circular
    uex = ex/emod_aux2
    uey = ey/emod_aux2
    uez = ez/emod_aux2

    # AUXILIARY QUANTITIES
    eq1 = np.sign(1 - np.abs(uhz))   # The ith element of eq1 is:
                                # '0' if the orbit IS equatorial
                                # '1' if the orbit IS NOT equatorial
    eq2 = 1 - eq1              # The ith element of eq2 is:
                                # '1' if the the orbit IS equatorial 
                                # '0' if the orbit IS NOT equatorial

    # UNIT VECTOR NORMAL TO h AND e UNIT VECTORS
    # It coincides with the scalar product of the unit vectors N (node line direction)
    # and e multiplied by sin(i), where i is the inclination
    # We are interested only in the third component
    uhez = uhx*uey - uhy*uex

    # UNIT VECTOR NORMAL TO h AND e UNIT VECTORS
    # It coincides with the scalar product of the unit vectors N (node line direction)
    # and e multiplied by sin(i), where i is the inclination
    # We are interested only in the third component
    uhrz = uhx*ury - uhy*urx

    # COMPUTATION OF ORBITAL ELEMENTS
    # If the orbit is EQUATORIAL we assume:
    # - Omega = 0
    # - omega = angle between the inertial axis X and the eccentricity vector

    # If the orbit is CIRCULAR we assume:
    # - omega = 0
    # - nu    = angle between the node line and the position vector, if the
    #           orbit is equatorial then nu is the angle between the inertial X
    #           axis and the position vector

    a = - mu/(2*En)   # semimajor axis

    e = emod   # eccentricity

    i = np.arctan2((uhx**2 + uhy**2)**(1/2), uhz)   # inclination

    Omega = (np.arctan2(uhx, -uhy))*eq1   # right ascension of ascending node
    Omega = 2*np.pi*(Omega < 0) + Omega

    omega = eq1*(np.arctan2(uez, uhez)) + eq2*(np.arctan2(uey, uex))   # argument of pericenter
    omega = 2*np.pi*(omega < 0) + omega

    arg_lat_1 = np.arctan2(urz, uhrz)
    arg_lat_1 = 2*np.pi*(arg_lat_1 < 0) + arg_lat_1
    arg_lat_2 = np.arctan2(ury, urx)
    arg_lat_2 = 2*np.pi*(arg_lat_2 < 0) + arg_lat_2

    nu = eq1*(arg_lat_1-omega) + eq2*(arg_lat_2-omega)   # true anomaly
    nu = 2*np.pi*(nu < 0) + nu

    elorb = [a, e, i, Omega, omega, nu]   # NOTE: elorb: n_rows = N = size(R,1)
                                        #              n_columns = 6
    return elorb


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
        print("past apogee")
        # past apogee
        true_anomaly = 2*np.pi - true_anomaly   
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


# Initialise P matrix, page 4, formula (4):
def dromo2rv(Lc, sigma, tau, zeta1, zeta2, zeta3, eta1, eta2, eta3, eta4):
    """
    If you want to have r,v non dimensional from dromo, let Lc be 1.
    Dromo elements to cartesian representation of state vector (page 14)
    Outputs: Dimensional components of position and velocity
    """
    s = 1 + zeta1 * np.cos(sigma) + zeta2 * np.sin(sigma)
    alpha = (Lc / (zeta3**2 * s))
    omegacLc = (math.sqrt(GMe/Lc**3) * Lc)

    p11 = 1-2*(eta2**2 + eta3**2)
    p12 = 2*eta1*eta2 - 2*eta4*eta3
    p21 = 2*eta1*eta2 + 2*eta4*eta3
    p22 = 1-2*(eta1**2 + eta3**2)
    p31 = 2*eta1*eta3 - 2*eta4*eta2
    p32 = 2*eta3*eta2 + 2*eta4*eta1

    x = alpha * ( p11*np.cos(sigma) + p12*np.sin(sigma) )
    # x = np.round(x, 11)
    y = alpha * ( p21*np.cos(sigma) + p22*np.sin(sigma) )
    # y = np.round(y, 11)
    z = alpha * ( p31*np.cos(sigma) + p32*np.sin(sigma) ) 
    # z = np.round(z, 11)

    V1 = -zeta3*(np.sin(sigma)+zeta2)
    V2 =  zeta3*(np.cos(sigma)+zeta1)

    xv = omegacLc * ( p11*V1 + p12*V2 )
    # xv = np.round(xv, 11)
    yv = omegacLc * ( p21*V1 + p22*V2 )
    # yv = np.round(yv, 11)
    zv = omegacLc * ( p31*V1 + p32*V2 ) 
    # zv = np.round(zv, 11)

    return x,y,z, xv,yv,zv 


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


