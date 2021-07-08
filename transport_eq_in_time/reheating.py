import sys
sys.path.append("..")
import numpy as np
from scipy.integrate import solve_ivp
from common import cosmology

def calc_temperature(y):
    rho_phi, rho_tot = np.exp(y)
    rho_rad = rho_tot - rho_phi
    T = cosmology.calc_temperature(rho_rad)
    return T

def rhs(log_t, y, Gamma_phi):
    rho_phi, rho_tot = np.exp(y)
    t = np.exp(log_t)
    H = cosmology.calc_hubble_parameter(rho_tot)
    d_log_rho_phi_d_log_t = - t * (3 * H + Gamma_phi)
    d_log_rho_tot_d_log_t = - H * t * (4 - rho_phi / rho_tot)
    return (d_log_rho_phi_d_log_t, d_log_rho_tot_d_log_t)

def solve_reheating_eq(t_start, t_end, initial, Gamma_phi):
    interval = np.log((t_start, t_end))
    sol = solve_ivp(rhs, interval, initial, dense_output=True, rtol=1e-5, args=(Gamma_phi,))

    def T_fn(t):
        y = sol.sol(np.log(t))
        return calc_temperature(y)

    final = sol.sol(interval[-1])

    return T_fn, final

def calc_initial_reheating(H_inf):
    rho_phi_inf = cosmology.calc_energy_density_from_hubble(H_inf)
    t_start = 1 / H_inf
    initial = (np.log(rho_phi_inf), np.log(rho_phi_inf))
    return t_start, initial
