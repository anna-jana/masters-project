import numpy as np
from numba import jit

from .constants import *

@jit(nopython=True)
def eta_L_a_to_eta_B_0(eta_L_a):
    return L_to_B_final_factor * eta_L_a # formula from paper (*)

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
    " this function is for the old code "
    return -eta_L_a_to_eta_B_0(calc_asym_parameter(T, n_L)) # -sign from defintion of (anti)matter

def calc_eta_B_final(red_chem_B_minus_L, T):
    return red_chem_pot_to_B_density_final(red_chem_B_minus_L, T)/ calc_photon_number_density(T)

def red_chem_pot_to_B_density_final(red_chem_B_minus_L, T):
    return - g_star_0 / g_star * C_sph * T**3 / 6 * red_chem_B_minus_L



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
def calc_lepton_asym_in_eqi(T, mu_eff, prefactor=1, g=4):
    # return 4 / np.pi**2 * mu_eff * T**2 # boltzmann thermodynamics
    return g / 6 * prefactor * mu_eff * T**2 # boltzmann thermodynamics

@jit(nopython=True)
def calc_start_time(H_inf):
    return 1 / H_inf

# analyical results for reheating
def calc_rho_phi_analytical(t0, t, R_osc, R, rho_phi_0, Gamma_phi):
    return rho_phi_0 * (R / R_osc)**-3 * np.exp(- Gamma_phi * (t - t0)) # K&T (8.30)

def calc_rho_R_analytical(rho_R, t0, t, R_osc, R, rho_phi_0, Gamma_phi):
    start = (6 / np.pi)**0.5 / 10 * m_pl * Gamma_phi * rho_phi_0**0.5 * (R / R_osc)**(-3/2) * (1 - (R / R_osc)**(-5/2)) # K&T (8.32)
    t_RH = 1 / Gamma_phi
    rad_dom = rho_R[-1] * (R / R[-1])**(-4)
    return np.where(t < t_RH, start, rad_dom)

def calc_reheating_temperature(Gamma_phi, result="paper"):
    if result == "paper":
        return 2e13 * np.sqrt(Gamma_phi / 1e9)
    elif result == "K&T":
        return 0.55 * g_star**(-1/4) * (m_pl * Gamma_phi)**0.5
    else:
        raise ValueError(f"invalid result {result}")

def calc_maximal_temperature(Gamma_phi, H_inf, result="paper"):
    if result == "paper":
        return 5e13 * (Gamma_phi / 1e9)**(1/4) * (H_inf / 1e11)**(1/2)
    elif result == "K&T":
        rho_phi_0 = calc_energy_density_from_hubble(H_inf)
        return 0.8 * g_star**(-1/4) * rho_phi_0**(1/8) * (Gamma_phi * m_pl)**(1/4)
    else:
        raise ValueError(f"invalid result {result}")

def switch_hubble_and_time_rad_dom(x):
    return 1 / (2*x)
