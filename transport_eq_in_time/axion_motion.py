import sys
if ".." not in sys.path: sys.path.append("..")
from collections import namedtuple

import numpy as np
from numba import njit
from scipy.optimize import root
from scipy.integrate import solve_ivp

from common import cosmology

def make_single_axion_rhs(calc_dVdtheta_over_f_a_squared):
    calc_dVdtheta_over_f_a_squared = njit(calc_dVdtheta_over_f_a_squared)
    def axion_rhs_standard(log_t, y, T_fn, H_fn, axion_parameter):
        theta, v = y
        t = np.exp(log_t)
        T = T_fn(t)
        H = H_fn(t)
        d_theta_d_ln_t = v * t
        d_v_d_t = - 3 * H * v - calc_dVdtheta_over_f_a_squared(T, theta, *axion_parameter)
        d_v_d_ln_t = d_v_d_t * t
        return (d_theta_d_ln_t, d_v_d_ln_t)
    return axion_rhs_standard

# calc_dVdtheta_over_f_a_squared(T, theta, *axion_parameter)

axion_rhs_simple = make_single_axion_rhs(lambda T, theta, m_a: m_a**2 * theta)

@njit
def calc_one_instanton_axion_mass(T, m_a, Lambda, p):
    return m_a * np.min((Lambda / T)**p, 1.0)

@njit
def calc_const_axion_mass(T, m_a):
    return m_a

axion_rhs_one_instanton = make_single_axion_rhs(
        lambda T, theta, m_a, Lambda, p: calc_one_instanton_axion_mass(T, m_a, Lambda, p)**2 * np.sin(theta)
        )

def calc_T_osc(calc_axion_mass, axion_parameter, N=2):
    m_a_0 = calc_axion_mass(0, *axion_parameter)
    # we assume raditation domination, since this is only an initial guess it will not invalidate the final result if its
    # not perfectly correct
    T_osc_initial_guess = cosmology.calc_temperature(cosmology.calc_energy_density_from_hubble(m_a_0))
    goal_fn = lambda T: np.log(calc_axion_mass(T, *axion_parameter) /
            (N * cosmology.calc_hubble_parameter(cosmology.calc_radiation_energy_density(T))))
    res = root(goal_fn, T_osc_initial_guess)
    if not res.success:
        return np.nan
    return res.x[0]

def solve_axion_motion(axion_rhs, axion_initial, t_start, t_end, T_fn, H_fn, axion_parameter, rtol, solver):
    sol = solve_ivp(axion_rhs, (np.log(t_start), np.log(t_end)), axion_initial, dense_output=True,
            args=(T_fn, H_fn, axion_parameter,),
            method=solver, rtol=rtol)
    assert sol.success, sol.message
    return sol.sol

def get_standard_axion_source(field_fn, _p):
    return lambda log_t: field_fn(log_t)[1]
