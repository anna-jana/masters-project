import numpy as np
from numba import jit

from .constants import *

@jit(nopython=True)
def eta_L_a_to_eta_B_0(eta_L_a):
    return L_to_B_final_factor * eta_L_a # formula from paper (*)

@jit(nopython=True)
def calc_temperature(rho_R):
    # in the paper its
    # (np.pi**2 / 3 * g_star * rho_R)**(1/4)
    return (rho_R / g_star * 30 / np.pi**2)**(1/4)

@jit(nopython=True)
def calc_radiation_energy_density(T):
    return np.pi**2 / 30 * g_star * T**4

@jit(nopython=True)
def calc_hubble_parameter(rho_total):
    return np.sqrt(rho_total) / (np.sqrt(3) * M_pl) # Friedmann

@jit(nopython=True)
def calc_rho_R(rho_phi, rho_tot):
    return rho_tot - rho_phi # neglegt axion

@jit(nopython=True)
def calc_energy_density_from_hubble(H):
    return 3 * M_pl**2 * H**2 # Friedmann eq.

@jit(nopython=True)
def calc_lepton_asym_in_eqi(T, mu_eff):
    return 4 / np.pi**2 * mu_eff * T**2 # boltzmann thermodynamics

zeta3 = 1.20206
g_photon = 2
@jit(nopython=True)
def calc_photon_number_density(T):
    return zeta3 / np.pi**2 * g_photon * T**3 # K&T (3.52)

@jit(nopython=True)
def calc_asym_parameter(T, n_L):
    n_gamma = calc_photon_number_density(T)
    return n_L / n_gamma # definition

@jit(nopython=True)
def n_L_to_eta_B_final(T, n_L):
    return -eta_L_a_to_eta_B_0(calc_asym_parameter(T, n_L)) # -sign from defintion of (anti)matter

