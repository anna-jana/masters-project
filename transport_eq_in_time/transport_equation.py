# Based on https://arxiv.org/abs/2006.03148 for B - L (Lepto)genesis
# Warning: for different epoch change numbers!

import sys
import numpy as np
from scipy.optimize import root_scalar
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
from numba import njit

if ".." not in sys.path: sys.path.append("..")
from common import constants, cosmology

############# Rates for Different Processes ############

# Yukawa Rates
Y_tau = 1e-2
Y_top = 0.49
Y_bottom = 6.8e-3
kappa_tau = 1.7e-3
kappa_quarks_uptype = np.array([8e-3, 1e-2, 1.2e-2]) # \approx downtype
T_for_kappa_quarks_uptype = np.array([1e15, 1e12, 1e9])
kappa_quark_uptype_interp = interp1d(np.log(T_for_kappa_quarks_uptype), kappa_quarks_uptype, fill_value="extrapolate")

@njit
def calc_yukawa_rate(T, kappa, Y):
    return kappa * Y**2 * T**4

@njit
def calc_yukawa_rate_tau(T):
    return calc_yukawa_rate(T, kappa_tau, Y_tau)

def calc_yukawa_rate_quark(T, Y):
    kappa = kappa_quark_uptype_interp(np.log(T))
    return calc_yukawa_rate(T, kappa, Y)

def calc_yukawa_rate_top(T):
    return calc_yukawa_rate_quark(T, Y_top)

def calc_yukawa_rate_bottom(T):
    return calc_yukawa_rate_quark(T, Y_bottom)

# Sphaleron Rates
g2 = 0.55
g3 = 0.6

def g_to_alpha(g):
    return g**2 / (4*np.pi)

alpha_2 = g_to_alpha(g2)
alpha_3 = g_to_alpha(g3)

kappa_WS = 24 # for 1e12 GeV
kappa_SS = 2.7e2 # for 1e13 GeV

@njit
def calc_weak_sphaleron_rate(T):
    return kappa_WS / 2 * alpha_2**5 * T**4

@njit
def calc_strong_sphaleron_rate(T):
    return kappa_SS / 2 * alpha_3**5 * T**4

# Weinberg Rate
kappa_W = 3e-3
nu_EW = 174 # [GeV]
m_nu = 0.05e-9 # [GeV]

@njit
def calc_weinberg_op_rate(T):
    return kappa_W * m_nu**2 * T**6 / nu_EW**4

# calc all rates
def calc_rate_vector(T):
    return np.array((
        calc_weak_sphaleron_rate(T), # WS
        calc_strong_sphaleron_rate(T), # SS
        calc_yukawa_rate_tau(T), # Y_tau
        calc_yukawa_rate_top(T), # Y_top
        calc_yukawa_rate_bottom(T), # Y_bottom
        2 * calc_weinberg_op_rate(T), # W_12
        calc_weinberg_op_rate(T), # W_3
    ))

######### calc the equilibration (freezeout) temperatures of the processes #########
def eqi_temp(alpha, low=1e10, high=1e16):
    def ratio(T):
        gamma = calc_rate_vector(T)[alpha] / (T**3 / 6)
        expr = np.sum(charge_vector[alpha, :]**2 / dofs * gamma)
        H = cosmology.calc_hubble_parameter(cosmology.calc_radiation_energy_density(T)) # NOTE: this is not right (we are not always in rad dom but
        # we ignore this at this point
        return expr / H

    try:
        sol = root_scalar(lambda T: ratio(T) - 1, bracket=(high, low), rtol=1e-10, xtol=1e-10)
        if not sol.converged: return np.nan
    except ValueError:
        return np.nan
    return sol.root

def rate(T):
    if np.isclose(T, 0.0):
        return np.zeros(N_alpha)
    Gamma = calc_rate_vector(T) # rate per unit space time
    gamma = Gamma / (T**3 / 6) # rate per unit time
    return gamma

################################# Charge and Source Vectors #################################

charge_names = ["tau", "L12", "L3", "q12", "t", "b", "Q12", "Q3", "H"]
process_names = ["WS", "SS", "Y_tau", "Y_top", "Y_bottom", "W_12", "W_3"]
conserved_names = ["Y", "B12 - 3*B3"]

######### Charge Vectors #########
# Interactions:
charge_vector = np.array((
    # tau, L12, L3, q12, t,  b,  Q12, Q3,  H
    (0,    2,   1,  0,   0,  0,  6,   3,   0), # WS
    (0,    0,   0, -4,  -1, -1,  4,   2,   0), # SS
    (-1,   0,   1,  0,   0,  0,  0,   0,  -1), # Y_tau # maybe H charge 1 -> -1?
    (0,    0,   0,  0,  -1,  0,  0,   1,   1), # Y_top
    (0,    0,   0,  0,   0, -1,  0,   1,  -1), # Y_bottom
    (0,    2,   0,  0,   0,  0,  0,   0,   2), # Weinberg_12
    (0,    0,   2,  0,   0,  0,  0,   0,   2), # Weinberg_3
))

# Conserved Quantaties:
conserved = np.array((
    # tau, L12,  L3,   q12,   t,    b,  Q12,   Q3,  H
    ( -1, -1/2, -1/2,  1/6,  2/3, -1/3, 1/6,  1/6, 1/2), # Y
    (  0,  0,    0,    1/3, -2/3, -2/3, 1/3, -2/3, 0),   # B12 - 2*B3
))

N_alpha, N = charge_vector.shape
N_A = conserved.shape[0]

########## Source Vectors ########
#                                           WS SS  Ytau Yt Yb  W12  W3
source_vector_B_minus_L_current = np.array((0,  0,  0,  0,  0, -2, -2))
source_vector_weak_sphaleron    = np.array((1,  0,  0,  0,  0,  0,  0))
source_vector_strong_sphaleron  = np.array((0,  1,  0,  0,  0,  0,  0))
source_vector_none              = np.zeros(len(source_vector_B_minus_L_current))

########### internal degrees of freedom ############
dofs = np.array([1,4,2,12,3,3,12,6,4])

######### Linear Combinations #######
# tau, L12,  L3,   q12,   t,    b,  Q12,   Q3,  H
charge_vector_B_minus_L = np.array((-1, -1, -1, 1/3, 1/3, 1/3, 1/3, 1/3, 0))
charge_vector_B_plus_L = np.array((1, 1, 1, 1/3, 1/3, 1/3, 1/3, 1/3, 0))

def calc_B_minus_L(red_chems):
    return (dofs * charge_vector_B_minus_L) @ red_chems

def calc_B_plus_L(red_chems):
    return (dofs * charge_vector_B_plus_L) @ red_chems

############ safety checks #########
# only the Weinberg violates B - L
B_minus_L_violations = [name for alpha, name in enumerate(process_names) if not np.isclose(charge_vector[alpha, :] @ charge_vector_B_minus_L, 0.0)]
assert B_minus_L_violations == ["W_12", "W_3"], str(B_minus_L_violations)

# all conserved charge vectors are orthogonal to the charge vectors of the interactions
# only okay if we include the additional minus sign in the charge vector of the right handed tau
assert len([(process_name, conserved_name)
 for conserved_name, n_A in zip(conserved_names, conserved)
 for process_name, n_alpha in zip(process_names, charge_vector)
 if not np.isclose(n_A @ n_alpha, 0.0)]) == 0

####################################### Transport Equation ###################################
unit = 1e-9

def transport_eq_rhs(log_t, red_chem_pots, T_fn, H_fn, T_dot_fn, axion_fn, n_S):
    t = np.exp(log_t)
    T = T_fn(t)
    if T <= 0.0:
        return np.zeros(N)
    T_dot = T_dot_fn(t)
    H = H_fn(t)
    _, theta_dot = axion_fn(log_t)
    d_red_chem_pot_d_t = (
            - (rate(T) * (charge_vector @ red_chem_pots - n_S * theta_dot / T / unit)) @ charge_vector / dofs
            - red_chem_pots * 3 * (T_dot / T + H)
    )
    d_red_chem_pot_d_ln_t = d_red_chem_pot_d_t * t
    return d_red_chem_pot_d_ln_t

jac_mats = [1 / dofs[:, None] * np.outer(charge_vector[alpha, :], charge_vector[alpha, :]) for alpha in range(N_alpha)]
I = np.eye(N)

def transport_eq_jac(log_t, red_chem_pots, T_fn, H_fn, T_dot_fn, axion_fn, n_S):
    t = np.exp(log_t)
    T = T_fn(t)
    if T <= 0.0: return np.zeros((N, N))
    R = rate(T)
    T_dot = T_dot_fn(t)
    H = H_fn(t)
    return (- sum(jac_mats[alpha] * R[alpha] for alpha in range(N_alpha)) - 3 * (T_dot / T + H) * I) * t

def solve_transport_eq(t_start, t_end, initial_red_chem_pots, rtol, T_fn, H_fn, T_dot_fn, axion_fn, source_vector):
    sol = solve_ivp(transport_eq_rhs, (np.log(t_start), np.log(t_end)), initial_red_chem_pots / unit,
            # method="Radau", rtol=rtol, # jac=transport_eq_jac,
            method="BDF", rtol=rtol, jac=transport_eq_jac,
            args=(T_fn, H_fn, T_dot_fn, axion_fn, source_vector))
    return np.exp(sol.t), sol.y * unit
