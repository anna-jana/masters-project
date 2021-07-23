import sys
import numpy as np
from scipy.integrate import solve_ivp

from common import cosmology, constants

def rhs_axion_decay(log_t, y, Gamma_a):
    rho_R, rho_a, R = np.exp(y)
    t = np.exp(log_t)
    H = cosmology.calc_hubble_parameter(rho_a + rho_R)
    d_log_rho_R_d_log_t = - t * (4 * H - Gamma_a * rho_a / rho_R)
    d_log_rho_a_d_log_t = - t * (3 * H + Gamma_a)
    d_log_R_d_log_t = t * H
    return d_log_rho_R_d_log_t, d_log_rho_a_d_log_t, d_log_R_d_log_t

def compute_axion_decay(T_start, red_chem_B_minus_L, theta, theta_dot, m_a, f_a, axion_decay_rate):
    # initial condition
    R_0 = 1.0
    rho_R   = cosmology.calc_radiation_energy_density(T_start)
    rho_kin = 0.5 * f_a**2 * theta_dot**2
    rho_pot = 0.5 * f_a**2 * m_a**2 * theta**2
    rho_a   = rho_kin + rho_pot
    initial = np.log([rho_R, rho_a, R_0])

    # we start a some fake time (not the cosmological one)
    n_B_start = cosmology.red_chem_pot_to_B_density_final(red_chem_B_minus_L, T_start)
    H = cosmology.calc_hubble_parameter(rho_R + rho_a)
    t_start = 1 / (2*H)
    t_end = 1 / axion_decay_rate
    start = np.log(t_start)
    end = np.log(t_end)
    n_B_start = cosmology.red_chem_pot_to_B_density_final(red_chem_B_minus_L, T_start)

    step = 0
    last_eta_B = np.nan
    while True:
        sol = solve_ivp(rhs_axion_decay, (start, end), initial, args=(axion_decay_rate,),
                method="Radau", rtol=1e-5)
        if not sol.success:
            #print("%e" % f_a, start, end, step)
            return last_eta_B

        t = np.exp(sol.t[-3:])
        rho_R, rho_a, R = np.exp(sol.y[:, -3:])
        T = cosmology.calc_temperature(rho_R)
        n_B = n_B_start * (R_0 / R)**3
        eta_B = cosmology.calc_asym_parameter(T, n_B) # this works for all asymmetry parameters

        try:
            deriv = (eta_B[-1] - eta_B[-2]) / (t[-1] - t[-2])
            deriv2 = (eta_B[-3] - 2 * eta_B[-2] + eta_B[-1]) / (t[-1] - t[-2])**2
            delta = np.abs(deriv / eta_B[-1] * t[-1])
            delta2 = np.abs(deriv2 / eta_B[-1] * t[-1]**2)
        except IndexError:
            return eta_B[-1] # np.nan
        if delta < constants.convergence_epsilon and delta2 < constants.convergence_epsilon:
            return eta_B[-1]
        else:
            last_eta_B = eta_B[-1]

        start, end = end, end - 0.5
        step += 1
