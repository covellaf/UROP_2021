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
from scipy.integrate import odeint, solve_ivp

# Sys imports
import time

# Plot imports
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

"""
DROMO FORMULATION
m is the mass of particle M moving in a fixed intertial frame Ox1y1z1
O is the center of a celestial body
M is acted upon by the gravitational force of the celestial body (Keplerian motion) and 
the remaining forces which are included in the perturbing force
* means dimensional (d)
"""

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

def Dromo2orbel(sigma, tau, zeta1, zeta2, zeta3, eta1, eta2, eta3, eta4):
    """
    conversion from DROMO elements to classical orbital elements
    taken from page 15-16 of Urrutxua et al. paper 2015
    """
    a = - a0d / (zeta3**2 * (zeta1**2 + zeta2**2 -1))
    e_norm = math.sqrt(zeta1**2 + zeta2**2)
    beta = np.arctan2(zeta2, zeta1)
    i = 2*np.arccos(math.sqrt(eta3**2 + eta4**2))
    RAAN = - np.arctan2(eta3, eta4) + np.arctan2(eta2, eta1)
    omega_tilda = - np.arctan2(eta3, eta4) - np.arctan2(eta2, eta1)
    omega = omega_tilda + beta
    theta = sigma - beta
    return a, e_norm, i, RAAN, omega, theta

def Dromotime2time(tau):
    # time in s
    # page 14. from paper
    time = t0 + tau * math.sqrt(a0d**3/GMe)
    return time


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
S0_ivp = np.array([tau_0, zeta1_0, zeta2_0, zeta3_0, eta1_0, eta2_0, eta3_0, eta4_0])

print(S0_ivp)

# tf = 288.12768941*24*3600            # s
tf = 2*np.pi * 50                    # roughly 50 orbits
# NOTE:
# 314*math.sqrt(136000**3/398601) = 24944156.272633158
# 24944156.272633158/(24*3600) = 288.7055124147356
delta_t = 0.5
n_steps = math.floor((tf-t0)/delta_t - 1)
# duration of integration in seconds
tspan = np.linspace(0, tf, n_steps)
# print("tspan: ", tspan) ---> 0 to 314


dromo2orb_res = []
trange = np.array([0, tf])
def dromo_ivp(sigma, State, perturbation=None):
    """
    Equations to be integrated
    Propagate DROMO EOMs using odeint
    """
    tau, zeta1, zeta2, zeta3, eta1, eta2, eta3, eta4 = State
    # For every func evaluation save the correspondent orbital elements to be plotted
    #dict
    dromo2orb_res.append(Dromo2orbel(sigma, tau, zeta1, zeta2, zeta3, eta1, eta2, eta3, eta4))

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
        # transform the current dromo element in cartesian coord (Dromo2orbel and then orb2state)
        # calculate the dimensional acceleration due to J2
        # adimensionalise it and go back to dromo eq.            
        pos, vel = orb2state(*Dromo2orbel(sigma, tau, zeta1, zeta2, zeta3, eta1, eta2, eta3, eta4))
        # print(pos)
        x, y, z = pos[0], pos[1], pos[2]
        # xv, yv, zv = vel[0], vel[1], vel[2]
        r_norm = math.sqrt(x**2 + y**2 + z**2)
        apxd = ( (3/2)*J2*GMe*Re**2/r_norm**4 ) * (x/r_norm)*(5*(z**2/r_norm**2) -1) # x-component of dimensional perturbing acceleration
        apyd = ( (3/2)*J2*GMe*Re**2/r_norm**4 ) * (y/r_norm)*(5*(z**2/r_norm**2) -1) # y-component of dimensional perturbing acceleration
        apzd = ( (3/2)*J2*GMe*Re**2/r_norm**4 ) * (z/r_norm)*(5*(z**2/r_norm**2) -3) # z-component of dimensional perturbing acceleration
        apx = apxd * (a0d**2/GMe)
        apy = apyd * (a0d**2/GMe)
        apz = apzd * (a0d**2/GMe)
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

    sol = solve_ivp( dromo_ivp, trange, S0_ivp, args=(None, ), t_eval=tspan)


    # df = pd.DataFrame(dromo2orb_res, columns=["a", "e", "i", "RAAN", "omega", "theta"])
    # # print(df)
    # df.plot(subplots=True)
    # plt.tight_layout()
    # plt.show() 

    # # print(sol.y[1])#, sol.t, np.shape(sol.y))
    # sol_j2 = solve_ivp( dromo_ivp, trange, S0_ivp , args=("J2", ), t_eval=tspan)
    # df = pd.DataFrame(sol.sol, columns=["tau", "z1", "z2", "z3", "z4", "h1", "h2", "h3", "h4"])
    # # print(df)
    # df.plot(subplots=True)
    # plt.tight_layout()
    # plt.show()
    

    # t = sol.t
    # tau = sol.y[0]
    # z1 = sol.y[1]
    # z2 = sol.y[2] 
    # z3 = sol.y[3]
    # h1 = sol.y[4]
    # h2 = sol.y[5]
    # h3 = sol.y[6]
    # h4 = sol.y[7]
    # print(np.shape(sol.y))
    # # plt.plot(sol.t, sol.y.T)
    # plt.plot(t, tau, 'b-', label='tau')
    # plt.show()
    # plt.plot(t, z1, 'b-', label='z1')
    # plt.show()
    # plt.plot(t, z2, 'b-', label='z2')
    # plt.show()
    # plt.plot(t, z3, 'b-', label='z3')
    # plt.show()
    # plt.plot(t, h1, 'b-', label='h1')
    # plt.show()
    # plt.plot(t, h2, 'b-', label='h2')
    # plt.show()
    # plt.plot(t, h3, 'b-', label='h3')
    # plt.show()
    # plt.plot(t, h4, 'b-', label='h4')
    # plt.show()

    # t = sol_j2.t
    # tau = sol_j2.y[0]
    # z1 = sol_j2.y[1]
    # z2 = sol_j2.y[2] 
    # z3 = sol_j2.y[3]
    # h1 = sol_j2.y[4]
    # h2 = sol_j2.y[5]
    # h3 = sol_j2.y[6]
    # h4 = sol_j2.y[7]
    # print(np.shape(sol_j2.y))
    # plt.plot(t, tau, 'r-', label='tau')
    # plt.show()
    # plt.plot(t, z1, 'r-', label='z1')
    # plt.show()
    # plt.plot(t, z2, 'r-', label='z2')
    # plt.show()
    # plt.plot(t, z3, 'r-', label='z3')
    # plt.show()
    # plt.plot(t, h1, 'r-', label='h1')
    # plt.show()
    # plt.plot(t, h2, 'r-', label='h2')
    # plt.show()
    # plt.plot(t, h3, 'r-', label='h3')
    # plt.show()
    # plt.plot(t, h4, 'r-', label='h4')
    # plt.show()

    # # elementwise = list(zip(*dromo2orb_res))
    # df = pd.DataFrame(dromo2orb_res, columns=["a", "e", "i", "RAAN", "omega", "theta"])
    # # print(df)
    # df.plot(subplots=True)
    # plt.tight_layout()
    # plt.show()



if __name__ == "__main__":
    main()