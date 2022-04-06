# Based on https://arxiv.org/abs/2006.03148 for B - L (Lepto)genesis
# Warning: for different epoch (e.g. electroweak phasetransition) change numbers!

import time, importlib
import numpy as np, matplotlib.pyplot as plt
from scipy.optimize import root_scalar
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
import decay_process, axion_motion
decay_process = importlib.reload(decay_process)
axion_motion = importlib.reload(axion_motion)

############# Rates for Different Processes ############

# NOTE: numbers from the paper
# NOTE: all these number are unitless except for the T^4 terms
# Yukawa Rates
Y_tau = 1e-2
Y_top = 0.49
Y_bottom = 6.8e-3
kappa_tau = 1.7e-3
kappa_quarks_uptype = np.array([8e-3, 1e-2, 1.2e-2]) # \approx downtype
T_for_kappa_quarks_uptype = np.array([1e15, 1e12, 1e9])
kappa_quark_uptype_interp = interp1d(np.log(T_for_kappa_quarks_uptype), kappa_quarks_uptype, fill_value="extrapolate")
# Sphaleron Rates
g2 = 0.55
g3 = 0.6
def g_to_alpha(g): return g**2 / (4*np.pi)
alpha_2 = g_to_alpha(g2)
alpha_3 = g_to_alpha(g3)
kappa_WS = 24 # for 1e12 GeV
kappa_SS = 2.7e2 # for 1e13 GeV
# Weinberg Rate
# NOTE: these depend on nu_EW and m_nu in GeV
kappa_W = 3e-3
nu_EW = 174 # [GeV] electroweak scale
m_nu = 0.05e-9 # [GeV] neutrino mass
# NOTE: the factor 6 comes from gamma = Gamma / (T^3 / 6)
ws_const = kappa_WS / 2 * alpha_2**5 * 6
ss_const = kappa_SS / 2 * alpha_3**5 * 6
weinberg_const = kappa_W * m_nu**2 / nu_EW**4 * 6 # [GeV^-2]
Y_tau_const = kappa_tau * Y_tau**2 * 6
Y_top_const = Y_top**2 * 6
Y_bottom_const = Y_bottom**2 * 6

# NOTE: this is the rate per time (gamma in the paper) and not per time and space (Gamma in the paper) !!
# NOTE: unit allows you to use this directly with temperatures that are not in GeV!
def calc_rate_vector(T, unit=1.0):
    if np.isclose(T, 0.0):
        return np.zeros(N_alpha)
    # sphalerons
    ws_rate = ws_const * T
    ss_rate = ss_const * T
    # Yukawa
    Y_tau_rate = Y_tau_const * T
    A = kappa_quark_uptype_interp(np.log(T * unit)) * T
    Y_top_rate = A * Y_tau_const
    Y_bottom_rate = A * Y_bottom_const
    # Weinberg
    W_3 = (weinberg_const * unit**2) * T**3
    W_12 = 2*W_3
    return np.array((ws_rate, ss_rate, Y_tau_rate, Y_top_rate, Y_bottom_rate, W_12, W_3))

######### calc the equilibration (freezeout) temperatures of the processes #########
def calc_eqi_temp(alpha, low=1e10, high=1e16):
    H_const = np.pi/decay_process.M_pl*np.sqrt(decay_process.g_star)
    def goal_fn(T):
        gamma = calc_rate_vector(T)[alpha]
        expr = np.sum(charge_vector[alpha, :]**2 / dofs * gamma)
        # not FIXME: this is not right (we are not always in rad dom)
        # this is not too problematic as the important result do not depend on this
        # also otherwise this depends on the exact reheating process -> depends on H_inf, Gamma_inf
        H = H_const * T**2
        return expr / H - 1
    try:
        sol = root_scalar(goal_fn, bracket=(high, low), rtol=1e-10, xtol=1e-10)
        if not sol.converged: return np.nan
    except ValueError:
        return np.nan
    return sol.root

################################# Charge and Source Vectors #################################
# namens for different things
charge_names = [r"$\tau$", "$L_{12}$", "$L_{3}$", "$q_{12}$", "$t$", "$b$", "$Q_{12}$", "$Q_3$", "$H$"]
process_names = ["$WS$", "$SS$", r"$Y_\mathrm{tau}$", r"$Y_\mathrm{top}$", r"$Y_\mathrm{bottom}$", "$W_{12}$", "$W_3$"]
conserved_names = ["$Y$", r"$B_{12} - 3 \cdot B_3$"]

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
B_minus_L_violations = [name for alpha, name in enumerate(process_names)
        if not np.isclose(charge_vector[alpha, :] @ charge_vector_B_minus_L, 0.0)]
assert B_minus_L_violations == process_names[-2:], str(B_minus_L_violations)

# all conserved charge vectors are orthogonal to the charge vectors of the interactions
# only okay if we include the additional minus sign in the charge vector of the right handed tau
assert len([(process_name, conserved_name)
 for conserved_name, n_A in zip(conserved_names, conserved)
 for process_name, n_alpha in zip(process_names, charge_vector)
 if not np.isclose(n_A @ n_alpha, 0.0)]) == 0

####################################### Transport Equation ###################################
def rhs(log_t, red_chem_pots, T_and_H_and_T_dot_fn, axion_source, source_vector, unit, Gamma_inf):
    t = np.exp(log_t)
    T, H, T_dot = T_and_H_and_T_dot_fn(t)
    if T <= 0.0: return np.zeros(N)
    Gamma_inf2 = Gamma_inf**2
    T /= Gamma_inf
    H /= Gamma_inf
    T_dot /= Gamma_inf2
    theta_dot = axion_source(t) # axion source takes inf_time not axion_time!
    R = calc_rate_vector(T, unit=Gamma_inf)
    d_red_chem_pot_d_t = (
            - (R * (charge_vector @ red_chem_pots - source_vector * theta_dot / T / unit)) @ charge_vector / dofs
            - red_chem_pots * 3 * (T_dot / T + H)
    )
    d_red_chem_pot_d_ln_t = d_red_chem_pot_d_t * t
    return d_red_chem_pot_d_ln_t

jac_mats = [1 / dofs[:, None] * np.outer(charge_vector[alpha, :], charge_vector[alpha, :]) for alpha in range(N_alpha)]
I = np.eye(N)
def jac(log_t, red_chem_pots, T_and_H_and_T_dot_fn, axion_source, source_vector, unit, Gamma_inf):
    t = np.exp(log_t)
    T, H, T_dot = T_and_H_and_T_dot_fn(t)
    Gamma_inf2 = Gamma_inf**2
    T /= Gamma_inf
    H /= Gamma_inf
    T_dot /= Gamma_inf2
    if T <= 0.0: return np.zeros((N, N))
    R = calc_rate_vector(T, unit=Gamma_inf)
    return (- sum(jac_mats[alpha] * R[alpha] for alpha in range(N_alpha)) - 3 * (T_dot / T + H) * I) * t

M_inv = np.linalg.inv(np.vstack([charge_vector, dofs * conserved]))
def solve(t_inf_time, initial_red_chem_pots, T_and_H_and_T_dot_fn, axion_source, source_vector, Gamma_inf, conv_factor, debug=False):
    # determine the equilibrium solution in order to set the scale of the solution
    ts_inf = np.geomspace(decay_process.t0, decay_process.t0 + t_inf_time, 100)
    theta_dot = axion_source(ts_inf[1:])
    T, _, _ = T_and_H_and_T_dot_fn(ts_inf) # T is in GeV
    source = theta_dot / (T[1:] / Gamma_inf)
    b = M_inv @ np.hstack([source_vector, np.zeros(N_A)])
    red_chem_pots_eq = source[None, :] * b[:, None]
    units = np.max(red_chem_pots_eq, axis=1)
    unit = np.mean(units)
    # solve the transport eq.
    sol = solve_ivp(rhs, (np.log(decay_process.t0), np.log(decay_process.t0 + t_inf_time)),
            initial_red_chem_pots / unit, dense_output=True, method="LSODA", rtol=1e-6, jac=jac,
            args=(T_and_H_and_T_dot_fn, axion_source, source_vector, unit, Gamma_inf))
    if debug:
        plt.figure()
        t_W = decay_process.T_to_t(T_equis[-2], lambda t: T_and_H_and_T_dot_fn(t)[0], T[-1])
        red_chem_pots = sol.sol(np.log(ts_inf))
        for (i,  name) in enumerate(charge_names):
            plt.loglog(ts_inf, np.abs(unit * red_chem_pots[i, :]), label=name)
        plt.loglog(ts_inf, np.abs(calc_B_minus_L(unit * red_chem_pots)), label="B - L", color="black", lw=2)
        plt.xlabel(r"$t \cdot \Gamma_\mathrm{inf}$")
        plt.ylabel(r"$|\mu_i / T|$")
        plt.legend(ncol=3)
    return sol

############################################ test code ################################################
T_equis = [calc_eqi_temp(alpha) for alpha in range(N_alpha)]

def test(H_inf, Gamma_inf, m_a, tmax_axion_time=10.0):
    axion_parameter = (m_a,)
    calc_axion_mass = lambda T, m_a: m_a
    rho0 = 3*decay_process.M_pl**2*H_inf**2
    m_a_osc, conv_factor = axion_motion.calc_axion_timescale(calc_axion_mass, axion_parameter, Gamma_inf)
    tmax_inf_time = tmax_axion_time * conv_factor

    start_time = time.time()
    sol_rh = decay_process.solve_decay_eqs(tmax_inf_time, 0.0, rho0, Gamma_inf, debug=True)
    T_and_H_fn, T_and_H_and_T_dot_fn = decay_process.to_temperature_and_hubble_fns(sol_rh, rho0, Gamma_inf, debug=True)
    decay_time = time.time()
    print("decay done")
    print("took:", decay_time - start_time, "seconds")

    sol_axion = axion_motion.solve(axion_motion.make_single_field_rhs(lambda theta, T, m_a: m_a**2*theta),
            (1.0, 0.0), calc_axion_mass, axion_parameter, tmax_axion_time, T_and_H_fn, Gamma_inf, debug=True)
    axion_source = axion_motion.get_axion_source_single_field(sol_axion, conv_factor)
    axion_time = time.time()
    print("axion done")
    print("took:", axion_time - decay_time, "seconds")

    sol_transp_eq = solve(tmax_inf_time, np.zeros(N), T_and_H_and_T_dot_fn,
            axion_source, source_vector_weak_sphaleron, Gamma_inf, conv_factor, debug=True)
    trans_time = time.time()
    print("trans done")
    print("took:", trans_time - axion_time, "seconds")

    plt.show()


