import astropy.units as u
import astropy.constants as c
import numpy as np
from scipy.special import zeta

T0 = 2.7255 * u.K
T0_err =
Omega_b_h_sq = 0.02233
Omega_b_h_sq_err =
H0 = 67.37 * u.km / (u.s * u.Mpc)
H0_err =

h = 0.674

n_gamma = zeta(3) / np.pi**2 * 2 * T0**3 * (c.k_B / c.hbar / c.c)**3

rho_c = 3 * H0**2 / (8*np.pi*c.G)
n_B = Omega_b_h_sq / h**2 * rho_c / c.m_p

eta_B = n_B / n_gamma
eta_B = eta_B.to("1").value

n_B_err = (
    (Omega_b_h_sq_err / h**2 * rho_c / c.m_p)**2 +
    (Omega_b_h_sq / h**2 / c.m_p * 3 * 2 * H0 / (8*np.pi*c.G) * H0_err)**2)**0.5
n_gamma_err = zeta(3) / np.pi**2 * 2 * 3 * T0**2 * (c.k_B / c.hbar / c.c)**3 * T0_err
eta_B_err = ((n_B_err / n_gamma)**2 + (n_gamma_err * n_B / n_gamma**2)**2)**0.5

