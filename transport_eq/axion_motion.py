import sys
if ".." not in sys.path: sys.path.append("..")

import numpy as np
from numba import jit, njit

from common import cosmology

def make_single_axion_rhs(calc_dVdtheta_over_f_a_squared):
    calc_dVdtheta_over_f_a_squared = njit(calc_dVdtheta_over_f_a_squared)
    def axion_rhs_standard(T, axion, axion_decay_rate, axion_parameter):
        theta, v = axion
        H = cosmology.calc_hubble_parameter(cosmology.calc_radiation_energy_density(T))
        d_theta_d_ln_T = - v / H
        d_v_d_ln_T = 3 * v + calc_dVdtheta_over_f_a_squared(T, theta, *axion_parameter) / H + axion_decay_rate * v / H
        return (d_theta_d_ln_T, d_v_d_ln_T)
    return axion_rhs_standard

# calc_dVdtheta_over_f_a_squared(T, theta, *axion_parameter)

axion_rhs_simple = make_single_axion_rhs(lambda T, theta, m_a: m_a**2 * theta)

@njit
def calc_axion_mass(T, m_a, Lambda, p):
    return m_a * np.min((Lambda / T)**p, 1.0)

axion_rhs_one_instanton = make_single_axion_rhs(
        lambda T, theta, m_a, Lambda, p: calc_axion_mass(T, m_a, Lambda, p)**2 * np.sin(theta)
        )

# calc_d2Vdtheta2_over_f_a_squared(T, theta, *axion_parameter)

@njit
def calc_d2Vdtheta2_simple(T, theta, m_a): return m_a**2

@njit
def calc_d2Vdtheta2_one_instanton(T, theta, m_a, Lambda, p): return calc_axion_mass(T, m_a, Lambda, p)**2 * np.cos(theta)
