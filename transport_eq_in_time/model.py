from collections import namedtuple

import numpy as np

import axion_motion
import transport_equation
import axion_decay
import reheating

# yes this library is included with sys.path hacking. I am sorry
from common import cosmology, constants

# the result of a simulation of the leptogenesis process i.e. the time evolution
# of all relevant quantaties
Result = namedtuple("Result",
        ["t", "red_chem_pots", "red_chem_B_minus_L", "axion_fn", "T_fn", "rh_final"])

# the model i.e. all relevant parameters and potentials etc.
AxionBaryogenesisModel = namedtuple("AxionBaryogenesisModel",
        ["source_vector", # coupling of the axion to SM
         "axion_rhs", "calc_axion_mass", "axion_parameter", "axion_initial", # axion potential
         "Gamma_phi", "H_inf", # inflation model
         ])

# the state of the system at one point and the next time to integrate to
SimulationState = namedtuple("SimulationState",
        ["initial_reheating", "initial_axion", "initial_transport_eq", "t_start", "t_end"])

# convergence parameters
rtol_axion = 1e-4
rtol_transport_eq = 1e-4 # 1e-6
num_osc = 1
num_osc_step = 20
axion_solver = "RK45"

def evolve(model, state):
    """
    Simulate the system with the model `model` from an initial state `state`.
    """
    # solve reheating
    T_fn, H_fn, T_dot_fn, rh_final = \
            reheating.solve_reheating_eq(state.t_start, state.t_end, state.initial_reheating, model.Gamma_phi)
    # solve axion
    axion_fn = axion_motion.solve_axion_motion(model.axion_rhs, state.initial_axion, state.t_start, state.t_end,
            T_fn, H_fn, model.axion_parameter, rtol_axion, axion_solver)
    # solve transport equation
    ts, red_chem_pots = transport_equation.solve_transport_eq(state.t_start, state.t_end, state.initial_transport_eq,
            rtol_transport_eq, T_fn, H_fn, T_dot_fn, axion_fn, model.source_vector)
    # create result
    return Result(t=ts, red_chem_pots=red_chem_pots, T_fn=T_fn, rh_final=rh_final,
            red_chem_B_minus_L=transport_equation.calc_B_minus_L(red_chem_pots), axion_fn=axion_fn)

# list of the equilibration temperatures
T_eqs = [transport_equation.eqi_temp(alpha) for alpha in range(transport_equation.N_alpha)]

def calc_t_end(calc_axion_mass, axion_parameter, t_end, Gamma_phi, H_inf):
    """
    calculate our guess for the end of the leptogenesis process. Return t_end is it is not None.
    """
    if t_end is not None:
        return t_end
    T_osc = axion_motion.calc_T_osc(calc_axion_mass, axion_parameter)
    T_end = min(T_osc, T_eqs[-1]) # min(min(T_eqs), T_osc)
    T_RH = cosmology.calc_reheating_temperature(Gamma_phi)
    t_inf = cosmology.calc_start_time(H_inf)
    if T_end > T_RH:
        t_end = calc_next_t(calc_axion_mass, axion_parameter, t_inf, T_RH, num_osc)
    H_end = cosmology.calc_hubble_parameter(cosmology.calc_radiation_energy_density(T_end))
    t_end = 1 / H_end
    return t_end

def start(model, t_end):
    """
    setup the intitial state of the simulation
    """
    t_start, rh_initial = reheating.calc_initial_reheating(model.H_inf)
    t_end = calc_t_end(model.calc_axion_mass, model.axion_parameter, t_end, model.Gamma_phi, model.H_inf)
    transport_eq_initial = np.zeros(transport_equation.N)
    state = SimulationState(
        initial_reheating = rh_initial,
        initial_axion = model.axion_initial,
        initial_transport_eq = transport_eq_initial,
        t_start = t_start,
        t_end   = t_end,
    )
    return evolve(model, state)

def calc_next_t(calc_axion_mass, axion_parameter, t, T, number_of_oscillations):
    """
    Calculate the next time point to integrate to given a start time `t` and temperature `T`
    from a given axion mass `calc_axion_mass` with parameters `axion_parameter`.
    Use the `number_of_oscillations` oscillations.
    """
    m_a = calc_axion_mass(T, *axion_parameter)
    Delta_t = number_of_oscillations * 2 * np.pi / m_a
    return t + Delta_t

def restart(res, calc_axion_mass, axion_parameter):
    """
    Given by integration over some interval. Create a new state at the end of the integrated
    interval to continue the simulation.
    """
    new_t_start = res.t[-1]
    T = res.T_fn(new_t_start)
    new_t_end = calc_next_t(calc_axion_mass, axion_parameter, new_t_start, T, num_osc_step)
    return SimulationState(
        initial_reheating = res.rh_final,
        initial_axion = res.axion_fn(np.log(res.t[-1])),
        initial_transport_eq = res.red_chem_pots.T[-1],
        t_start = new_t_start,
        t_end   = new_t_end,
    )

def done(res, debug):
    """
    Check if within the result of the simulation of a interval the
    B - L number has converged to a constant value.
    """
    y = res.red_chem_B_minus_L
    i = np.argmax(y)
    j = np.argmin(y)
    delta = np.abs((y[i] - y[j]) / ((y[i] + y[j]) / 2))
    if debug:
        print("delta:", delta, "vs", "epsilon:", constants.convergence_epsilon)
    return delta < constants.convergence_epsilon

def solve(model, t_end=None, converge=True, debug=False, collect=False):
    """
    Solve the baryogenesis model
    """
    result = start(model, t_end)
    if collect:
        steps = []
    while not done(result, debug):
        if collect:
            steps.append(result)
        if not converge:
            break
        state = restart(result, model.calc_axion_mass, model.axion_parameter)
        if debug:
            print("state:", state)
        result = evolve(model, state)
    if collect:
        steps.append(result)
        return steps
    else:
        t_final = result.t[-1]
        T_final = result.T_fn(t_final)
        axion_final = result.axion_fn(np.log(t_final))
        return result.red_chem_B_minus_L[-1], T_final, axion_final

def compute_final_asymmetry(model, red_chem_pot_B_minus_L, T, axion, axion_decay_rate, f_a):
    if axion_decay_rate != 0:
        m_a = model.calc_axion_mass(T, *model.axion_parameter) # NOTE: this is not 100% correct
        theta, theta_dot = axion[:2]
        eta_B = axion_decay.compute_axion_decay(T, red_chem_pot_B_minus_L, theta, theta_dot, m_a, f_a, axion_decay_rate)
    else:
        eta_B = cosmology.calc_eta_B_final(red_chem_pot_B_minus_L, T)
    return eta_B
