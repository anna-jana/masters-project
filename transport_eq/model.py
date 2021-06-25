from collections import namedtuple

import numpy as np
from scipy.integrate import solve_ivp

import axion_motion
import transport_equation
import axion_decay

from common import cosmology, constants

Result = namedtuple("Result",
        ["T", "red_chem_pots", "red_chem_B_minus_L", "axion"])

AxionBaryogenesisModel = namedtuple("AxionBaryogenesisModel",
        ["source_vector", "axion_rhs", "calc_d2Vdtheta2", "axion_decay_rate", "axion_parameter"])

SolverOptions = namedtuple("SolverOptions",
        ["solver", "rtol", "atol", "num_steps"])

SimulationState = namedtuple("SimulationState",
        ["initial", "T_start", "T_end"])

default_solver_options = SolverOptions(solver="Radau", rtol=1e-4, atol=1e-6, num_steps=None)

# TODO: clean up the options passed to the rhs function, we could also just pass the AxionBaryogenesisModel object itself
def rhs(log_T, y, n_S, axion_rhs, calc_d2Vdtheta2, axion_decay_rate, axion_parameter):
    T = np.exp(log_T)
    red_chem_pot, axion = y[:transport_equation.N], y[transport_equation.N:]
    theta_dot = axion[1]
    d_red_chem_pot_d_ln_T = transport_equation.transport_eq_rhs(T, red_chem_pot, n_S, theta_dot)
    d_axion_d_ln_T = axion_rhs(T, axion, axion_decay_rate, axion_parameter)
    return np.hstack([d_red_chem_pot_d_ln_T, d_axion_d_ln_T])

def evolve(model, state, options):
    start, end = np.log(state.T_start), np.log(state.T_end)
    if options.num_steps is not None:
        steps = np.linspace(start, end, options.num_steps); steps[0] = start; steps[-1] = end
    sol = solve_ivp(rhs, (start, end), state.initial,
             args=(model.source_vector, model.axion_rhs, model.calc_d2Vdtheta2, model.axion_decay_rate, model.axion_parameter),
             method=options.solver, rtol=options.rtol, atol=options.atol, t_eval=None if options.num_steps is None else steps, # jac=transport_eq_jac,
             )
    red_chem_pots = sol.y[:transport_equation.N] * transport_equation.unit
    axion = sol.y[transport_equation.N:]
    return Result(T=np.exp(sol.t), red_chem_pots=red_chem_pots,
            red_chem_B_minus_L=transport_equation.calc_B_minus_L(red_chem_pots), axion=axion)

T_eqs = [transport_equation.eqi_temp(alpha, plot=False) for alpha in range(transport_equation.N_alpha)]

def calc_T_end(calc_axion_mass, axion_decay_rate, axion_parameter, T_end, T_RH):
    if calc_axion_mass is None and T_end is None:
        raise ValueError("requires either calc_axion_mass (for T_osc computation) or fixed T_end for the determination of T_end")
    elif calc_axion_mass is not None and T_end is not None:
        raise ValueError("both calc_axion_mass and T_end are passed to start_solving")
    elif calc_axion_mass is not None:
        T_osc = axion_motion.calc_T_osc(calc_axion_mass, axion_parameter)
        T_dec = cosmology.calc_temperature(cosmology.calc_energy_density_from_hubble(axion_decay_rate))
        T_end = min(T_osc, T_eqs[-1]) # min(min(T_eqs), T_osc)
        if T_end > T_RH:
            T_end = calc_next_T(calc_axion_mass, axion_parameter, T_RH, num_osc)
        return T_end
    else:
        return T_end

def start(model, T_RH, axion_initial, options=default_solver_options, calc_axion_mass=None, T_end=None):
    T_end = calc_T_end(calc_axion_mass, model.axion_decay_rate, model.axion_parameter, T_end, T_RH)
    state = SimulationState(
        initial = np.hstack([np.zeros(transport_equation.N), axion_initial]),
        T_start = T_RH,
        T_end   = T_end,
    )
    #print("domain:", f"{T_RH:e} {T_end:e}")
    return evolve(model, state, options)

T_shrink_factor = 1.5

C = np.sqrt(np.pi**2 * constants.g_star /  90) / constants.M_pl

def calc_next_T(calc_axion_mass, axion_parameter, T_1, num_osc):
    m_a = calc_axion_mass(T_1, *axion_parameter)
    Delta_t = num_osc * 2 * np.pi / m_a
    T_2 = (2 * C * Delta_t + 1 / T_1**2)**(-0.5)
    return T_2

num_osc = 10

def restart(res, calc_axion_mass, axion_parameter):
    new_T_start = res.T[-1]
    new_T_end = calc_next_T(calc_axion_mass, axion_parameter, new_T_start, num_osc)
    return SimulationState(
        initial = np.hstack([res.red_chem_pots.T[-1] / transport_equation.unit, res.axion.T[-1]]),
        T_start = new_T_start,
        T_end = new_T_end,
    )

def done(res, debug):
    y = res.red_chem_B_minus_L
    i = np.argmax(y)
    j = np.argmin(y)
    delta = np.abs((y[i] - y[j]) / ((y[i] + y[j]) / 2))
    if debug:
        print("delta:", f"{delta:e}", y[i], y[j])
    return delta < constants.convergence_epsilon

def final_result(res):
    return cosmology.calc_eta_B_final(res.red_chem_B_minus_L[-1], res.T[-1])


def solve_to_end(model, T_RH, axion_initial, options=default_solver_options, calc_axion_mass=None, debug=False):
    result = start(model, T_RH, axion_initial, options, calc_axion_mass)
    step = 0
    while not done(result, debug):
        #print("step:", step)
        if debug:
            print("T:", f"{result.T[-1]:e}")
        state = restart(result, calc_axion_mass, model.axion_parameter)
        result = evolve(model, state, options)
        step += 1
    eta_B = final_result(result)
    return eta_B, result.red_chem_B_minus_L[-1], result.T[-1], result.axion.T[-1]

def solve(model, T_RH, axion_initial, f_a, options=default_solver_options, calc_axion_mass=None, debug=False):
    eta_B, red_chem_pot_B_minus_L, T, (theta, theta_dot) = solve_to_end(model, T_RH, axion_initial, options=options, calc_axion_mass=calc_axion_mass, debug=debug)
    if model.axion_decay_rate != 0:
        m_a = calc_axion_mass(T, *model.axion_parameter) # this is not 100% correct
        return axion_decay.compute_axion_decay(T, red_chem_pot_B_minus_L, theta, theta_dot, m_a, f_a, model.axion_decay_rate)
    else:
        return eta_B
