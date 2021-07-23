from collections import namedtuple

import numpy as np
from scipy.integrate import solve_ivp

import axion_motion
import transport_equation
import axion_decay
import reheating

from common import cosmology, constants

Result = namedtuple("Result",
        ["t", "red_chem_pots", "red_chem_B_minus_L", "axion_fn", "T_fn", "rh_final"])

AxionBaryogenesisModel = namedtuple("AxionBaryogenesisModel",
        ["source_vector", "axion_rhs", "calc_d2Vdtheta2", "axion_decay_rate", "axion_parameter",
            "Gamma_phi", "H_inf"])

SolverOptions = namedtuple("SolverOptions",
        ["solver", "rtol", "rtol_axion", "atol", "num_osc", "num_osc_step"],
        defaults=["Radau", 1e-4, 1e-3, 1e-6, 50, 20],
        )

SimulationState = namedtuple("SimulationState",
        ["initial_reheating", "initial_axion", "initial_transport_eq", "t_start", "t_end"])

default_solver_options = SolverOptions()

def evolve(model, state, options):
    start, end = np.log(state.t_start), np.log(state.t_end)
    # solve reheating
    T_fn, H_fn, T_dot_fn, rh_final = \
            reheating.solve_reheating_eq(state.t_start, state.t_end, state.initial_reheating, model.Gamma_phi)
    # solve axion
    axion_fn = axion_motion.solve_axion_motion(model.axion_rhs, state.initial_axion, state.t_start, state.t_end,
            T_fn, H_fn, model.axion_parameter, options.rtol_axion)
    # solve transport equation
    ts, red_chem_pots = transport_equation.solve_transport_eq(state.t_start, state.t_end, state.initial_transport_eq,
            options.rtol, T_fn, H_fn, T_dot_fn, axion_fn, model.source_vector)
    # create result
    return Result(t=ts, red_chem_pots=red_chem_pots, T_fn=T_fn, rh_final=rh_final,
            red_chem_B_minus_L=transport_equation.calc_B_minus_L(red_chem_pots), axion_fn=axion_fn)

T_eqs = [transport_equation.eqi_temp(alpha) for alpha in range(transport_equation.N_alpha)]

def calc_t_end(calc_axion_mass, axion_parameter, t_end, Gamma_phi, H_inf, options):
    if t_end is not None:
        return t_end
    elif calc_axion_mass is not None:
        T_osc = axion_motion.calc_T_osc(calc_axion_mass, axion_parameter)
        T_end = min(T_osc, T_eqs[-1]) # min(min(T_eqs), T_osc)
        T_RH = cosmology.calc_reheating_temperature(Gamma_phi)
        t_inf = cosmology.calc_start_time(H_inf)
        if T_end > T_RH:
            t_end = calc_next_t(calc_axion_mass, axion_parameter, t_inf, T_RH, options.num_osc)
        H_end = cosmology.calc_hubble_parameter(cosmology.calc_radiation_energy_density(T_end))
        t_end = 1 / H_end
        return t_end
    else:
        raise ValueError("requires either calc_axion_mass (for T_osc computation) or fixed T_end for the determination of T_end")

def start(model, axion_initial, options, calc_axion_mass, t_end):
    t_start, rh_initial = reheating.calc_initial_reheating(model.H_inf)
    t_end = calc_t_end(calc_axion_mass, model.axion_parameter, t_end, model.Gamma_phi, model.H_inf, options)
    transport_eq_initial = np.zeros(transport_equation.N)
    state = SimulationState(
        initial_reheating = rh_initial,
        initial_axion = axion_initial,
        initial_transport_eq = transport_eq_initial,
        t_start = t_start,
        t_end   = t_end,
    )
    return evolve(model, state, options)

def calc_next_t(calc_axion_mass, axion_parameter, t, T, num_osc):
    m_a = calc_axion_mass(T, *axion_parameter)
    Delta_t = num_osc * 2 * np.pi / m_a
    return t + Delta_t

def restart(res, calc_axion_mass, axion_parameter, options):
    new_t_start = res.t[-1]
    T = res.T_fn(new_t_start)
    new_t_end = calc_next_t(calc_axion_mass, axion_parameter, new_t_start, T, options.num_osc_step)
    return SimulationState(
        initial_reheating = res.rh_final,
        initial_axion = res.axion_fn(np.log(res.t[-1])),
        initial_transport_eq = res.red_chem_pots.T[-1],
        t_start = new_t_start,
        t_end   = new_t_end,
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
    T = res.T_fn(res.t[-1])
    x = res.red_chem_B_minus_L[-1]
    return cosmology.calc_eta_B_final(x, T)

def solve_to_end(model, axion_initial, options=default_solver_options, calc_axion_mass=None, t_end=None, debug=False, collect=False):
    result = start(model, axion_initial, options, calc_axion_mass, t_end)
    if collect:
        steps = []
    while not done(result, debug):
        if collect:
            steps.append(result)
        if t_end is not None:
            break
        state = restart(result, calc_axion_mass, model.axion_parameter, options)
        result = evolve(model, state, options)
    if collect:
        steps.append(result)
        return steps
    else:
        eta_B = final_result(result)
        t_final = result.t[-1]
        T = result.T_fn(t_final)
        axion_final = result.axion_fn(np.log(t_final))
        return eta_B, result.red_chem_B_minus_L[-1], T, axion_final, t_final

def solve(model, axion_initial, f_a, options=default_solver_options, calc_axion_mass=None, debug=False):
    eta_B, red_chem_pot_B_minus_L, T, (theta, theta_dot), t_final = \
            solve_to_end(model, axion_initial, options=options, calc_axion_mass=calc_axion_mass, debug=debug)
    if model.axion_decay_rate != 0:
        m_a = calc_axion_mass(T, *model.axion_parameter) # NOTE: this is not 100% correct
        eta_B = axion_decay.compute_axion_decay(T, red_chem_pot_B_minus_L, theta, theta_dot, m_a, f_a, model.axion_decay_rate)
    return eta_B
