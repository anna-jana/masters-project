from numba import jit
import numpy as np

from .constants import alpha

paper_sigma_eff = 1e-31 # [GeV^-2] from paper heavy neutrino exchange

def calc_Gamma_a_SU2(m_a, f_a):
    return float(alpha**2 / (64 * np.pi**3) * m_a**3 / f_a**2) # from paper

def calc_Gamma_a_U1(m_a, f_a):
    return float(alpha_1**2 / (32 * np.pi**3) * m_a**3 / f_a**2) # from paper

@jit(nopython=True)
def calc_Gamma_L(T, sigma_eff):
    n_l_eq = 2 / np.pi**2 * T**3
    return 4 * n_l_eq * sigma_eff # is this term only active in a certain range?
