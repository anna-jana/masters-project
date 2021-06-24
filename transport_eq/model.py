from collections import namedtuple

import numpy as np
from scipy.integrate import solve_ivp

import axion_motion
import transport_equation

from common import cosmology

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

def calc_T_end(calc_axion_mass, axion_decay_rate, axion_parameter, T_end):
    if calc_axion_mass is None and T_end is None:
        raise ValueError("requires either calc_axion_mass (for T_osc computation) or fixed T_end for the determination of T_end")
    elif calc_axion_mass is not None and T_end is not None:
        raise ValueError("both calc_axion_mass and T_end are passed to start_solving")
    elif calc_axion_mass is not None:
        T_osc = axion_motion.calc_T_osc(calc_axion_mass, axion_parameter)
        T_dec = cosmology.calc_temperature(cosmology.calc_energy_density_from_hubble(axion_decay_rate))
        return min(min(T_eqs), max(T_osc, T_dec))
    else:
        return T_end

def start(model, T_RH, axion_initial, options=default_solver_options, calc_axion_mass=None, T_end=None):
    state = SimulationState(
        initial = np.hstack([np.zeros(transport_equation.N), axion_initial]),
        T_start = T_RH,
        T_end   = calc_T_end(calc_axion_mass, model.axion_decay_rate, model.axion_parameter, T_end),
    )
    return evolve(model, state, options)

T_shrink_factor = 1.5

def restart(res):
    new_T_start = res.T[-1]
    new_T_end = new_T_start / T_shrink_factor
    return SimulationState(
        initial = np.hstack([res.red_chem_pots.T[-1] / transport_equation.unit, res.axion.T[-1]]),
        T_start = new_T_start,
        T_end = new_T_end,
    )

convergence_epsilon = 1e-2

def done(res, debug):
    y = res.red_chem_B_minus_L
    i = np.argmax(y)
    j = np.argmin(y)
    delta = np.abs((y[i] - y[j]) / ((y[i] + y[j]) / 2))
    if debug:
        print("delta:", f"{delta:e}", y[i], y[j])
    return delta < convergence_epsilon

C_sph = 8 / 23
def calc_eta_B_final(red_chem_B_minus_L, T):
    return - C_sph * T**3 / 6 * red_chem_B_minus_L / cosmology.calc_photon_number_density(T)

def final_result(res):
    return calc_eta_B_final(res.red_chem_B_minus_L[-1], res.T[-1])

def solve(model, T_RH, axion_initial, options=default_solver_options, calc_axion_mass=None, debug=False):
    result = start(model, T_RH, axion_initial, options, calc_axion_mass)
    while not done(result, debug):
        if debug:
            print("T:", f"{result.T[-1]:e}")
        state = restart(result)
        result = evolve(model, state, options)
    return final_result(result)

