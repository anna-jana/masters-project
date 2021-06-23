# Based on https://arxiv.org/abs/2006.03148 for B - L (Lepto)genesis
# Warning: for different epoch change numbers!

from collections import namedtuple
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar
from scipy.interpolate import interp1d

if ".." not in sys.path: sys.path.append("..")
from common import constants, cosmology
from numba import jit, njit

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
def eqi_temp(alpha, plot=True, low=1e10, high=1e16):
    def ratio(T):
        gamma = calc_rate_vector(T)[alpha] / (T**3 / 6)
        expr = np.sum(charge_vector[alpha, :]**2 / dofs * gamma)
        H = cosmology.calc_hubble_parameter(cosmology.calc_radiation_energy_density(T))
        return expr / H

    T_range = np.geomspace(high, low, 200)
    if plot: plt.loglog(T_range, [ratio(T) for T in T_range], label=process_names[alpha])

    try:
        sol = root_scalar(lambda T: ratio(T) - 1, bracket=(T_range[0], T_range[-1]), rtol=1e-10, xtol=1e-10)
        if not sol.converged: return np.nan
    except ValueError:
        return np.nan
    return sol.root

def rate(T):
    H = cosmology.calc_hubble_parameter(cosmology.calc_radiation_energy_density(T))
    Gamma = calc_rate_vector(T) # rate per unit space time
    gamma = Gamma / (T**3 / 6) # rate per unit time
    return gamma / H


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

def transport_eq_rhs(log_T, y, n_S, axion_rhs, calc_d2Vdtheta2, axion_decay_rate, axion_parameter):
    T = np.exp(log_T)
    red_chem_pot, axion = y[:N], y[N:]
    v = axion[1]
    theta_dot = v
    d_red_chem_pot_d_ln_T = (rate(T) * (charge_vector @ red_chem_pot - n_S * theta_dot / T / unit)) @ charge_vector / dofs
    d_axion_d_ln_T = axion_rhs(T, axion, axion_decay_rate, axion_parameter)
    return np.hstack([d_red_chem_pot_d_ln_T, d_axion_d_ln_T])

jac_mats = [1 / dofs[:, None] * np.outer(charge_vector[alpha, :], charge_vector[alpha, :]) for alpha in range(N_alpha)]

def transport_eq_jac(log_T, y, n_S, axion_rhs, calc_d2Vdtheta2, axion_decay_rate, axion_parameter):
    jac = np.zeros((y.size, y.size))
    T = np.exp(log_T)
    R = rate(T)
    H = cosmology.calc_hubble_parameter(cosmology.calc_radiation_energy_density(T))

    # transport equation jac
    for alpha in range(N_alpha):
        jac[:N, :N] += jac_mats[alpha] * R[alpha]

    # axion eq of motion jacobian
    # jac[N][N] = 0
    theta = y[N]
    jac[N + 1][N] = - 1 / H
    jac[N][N + 1] = calc_d2Vdtheta2(T, theta, *axion_parameter) / H
    jac[N + 1][N + 1] = axion_decay_rate / H + 3

    # source in the transport eq.
    for alpha in range(N_alpha):
        jac[:N, N + 1] += (R[alpha] * n_S[alpha]) * charge_vector[alpha]
    jac[:N, N + 1] /= dofs
    jac[:N, N + 1] /= -T

    return jac

TransportEqResult = namedtuple("TransportEqResult", ["T", "red_chem_pots", "red_chem_B_minus_L", "axion"])

def solve_transport_eq(T_RH, source_vector, axion_rhs, calc_d2Vdtheta2, axion_decay_rate, axion_parameter, axion_initial,
        solver="Radau", rtol=1e-8, atol=1e-6, T_end=1e10, num_steps=100):
    red_chem_pot_initial = np.zeros(charge_vector.shape[1])
    initial = np.hstack([red_chem_pot_initial, axion_initial])
    start, end = np.log(T_RH), np.log(T_end)
    if num_steps is not None:
        steps = np.linspace(start, end, num_steps); steps[0] = start; steps[-1] = end
    sol = solve_ivp(transport_eq_rhs, (start, end), initial,
             args=(source_vector, axion_rhs, calc_d2Vdtheta2, axion_decay_rate, axion_parameter),
             method=solver, rtol=rtol, atol=atol, t_eval=None if num_steps is None else steps, # jac=transport_eq_jac,
             )
    red_chem_pots = sol.y[:N] * unit
    axion = sol.y[N:]
    return TransportEqResult(T=np.exp(sol.t), red_chem_pots=red_chem_pots,
            red_chem_B_minus_L=calc_B_minus_L(red_chem_pots), axion=axion)

