import importlib
import numpy as np, matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar
from scipy.special import ellipj, ellipk, ellipkinc, ellipkm1
from scipy.constants import hbar, electron_volt
import axion_motion; axion_motion = importlib.reload(axion_motion)

############################## helper functions #########################
def sc(x, y):
    sn, cn, _dn, _ph = ellipj(x, y)
    return sn / cn

##################### conversion between parameters #####################
def calc_f_eff(f, eps):
    return 2 / np.pi * f * ellipkm1(eps**2)

def calc_f(f_eff, eps):
    return np.pi / 2 * f_eff / ellipkm1(eps**2)

def calc_eps(mR):
    return np.exp(- np.pi * mR)

def calc_m_phi(mR, M):
    return calc_eps(mR) * M

def calc_mass(_T, eps, M):
    return eps * M

def calc_mass_scale(m_phi, eps):
    return m_phi / eps

############### calculation of V_eff and d V_eff / d phi ################
def theta_to_phi_over_f(theta, eps):
    return 2 / eps * ellipkinc(theta / 2, - 1 / eps**2)

def calc_u(phi_over_f, eps):
    return eps * sc(phi_over_f / 2, 1 - eps**2)

def calc_du_dphi_times_f(phi_over_f, eps):
    _sn, cn, dn, _ph = ellipj(phi_over_f / 2, 1 - eps**2)
    return eps / 2 * dn / cn**2

def calc_d2u_dphi_times_f_sq(phi_over_f, eps):
    sn, cn, dn, _ph = ellipj(phi_over_f / 2, 1 - eps**2)
    return eps / 4 * (2 * cn**-3 * dn**2 * sn - (1 - eps**2) * sn * cn**-1)

def calc_V_eff_over_f_sq(phi_over_f, eps, M):
    return M**2 * 2 / (1 + 1 / calc_u(phi_over_f, eps)**2)

def calc_dV_eff_dphi_over_f(phi_over_f, eps, M):
    u = calc_u(phi_over_f, eps)
    return 4 * M**2 * calc_du_dphi_times_f(phi_over_f, eps) / (u**3 * (1 + 1 / u**2)**2)

def calc_d2V_eff_dphi2(phi_over_f, eps, M):
    u = calc_u(phi_over_f, eps)
    du_f = calc_du_dphi_times_f(phi_over_f, eps)
    d2u_f2 = calc_d2u_dphi_times_f_sq(phi_over_f, eps)
    X = 1 + 1 / u**2
    X2 = X**2
    B = u**3 * X2
    return 4*M**2 * (d2u_f2 / B + du_f**2 * (4*X - 3*u**2*X2) / B**2)

def calc_theta_dot(phi_over_f, phi_dot_over_f, eps, M):
    MM = 2*M**2
    A = calc_V_eff_over_f_sq(phi_over_f, eps, M) / MM
    return (
        1 / MM * np.abs(calc_dV_eff_dphi_over_f(phi_over_f, eps, M)) * phi_dot_over_f
        / ((1 - A)*A)**0.5
    )

########################## define the field class ##############################
class ClockworkAxionField(axion_motion.SingleAxionField):
    does_decay = False
    has_relic_density = True

    def calc_pot_deriv(self, phi_over_f, __T, eps, M):
        return calc_dV_eff_dphi_over_f(phi_over_f, eps, M)

    def calc_pot(self, phi_over_f, __T, eps, M):
        return calc_V_eff_over_f_sq(phi_over_f, eps, M)

    def find_dynamical_scale(self, eps, M):
        return M

    def calc_source(self, y, conv_factor, eps, M):
        phi_over_f, phi_dot_over_f = y
        phi_dot_over_f /= conv_factor
        return calc_theta_dot(phi_over_f, phi_dot_over_f, eps, M)

    def get_energy(self, y, f_a, eps, M):
        phi_over_f, phi_dot_over_f = y
        energy_scale = self.find_dynamical_scale(eps, M)
        return f_a**2 * (0.5 * (phi_dot_over_f * energy_scale)**2 + calc_V_eff_over_f_sq(phi_over_f, eps, M))

    def find_H_osc(self, eps, M): return eps / 3
    def find_mass(self, T, eps, M): return eps / 2

clockwork_axion_field = ClockworkAxionField()

############################## checking if the axion field interferce with inflation ###########################
def is_pot_curvature_too_large(mR, m_phi, H_inf, theta_i):
    eps = calc_eps(mR)
    phi_over_f_i = theta_to_phi_over_f(theta_i, eps)
    d2_Veff = calc_d2V_eff_dphi2(phi_over_f_i, eps, calc_mass_scale(m_phi, eps))
    return np.sqrt(np.abs(d2_Veff)) - H_inf

def get_max_mR(m_phi, H_inf, theta_i):
    sol = root_scalar(lambda mR: is_pot_curvature_too_large(mR, m_phi, H_inf, theta_i) / H_inf, bracket=(0, 20))
    return sol.root

####################### check if the axion might decay (then it cant be dark matter) ###################
def calc_decay_time(mR, m_phi, f_eff):
    eps = calc_eps(mR)
    f = calc_f(f_eff, eps)
    decay_rate = constants.alpha**2 / (64*np.pi**3) * eps**2 * m_phi**3 / f**2
    return 1 / decay_rate

def to_seconds(natural_time_in_GeV):
    return hbar * 1/electron_volt * 1e-9 * natural_time_in_GeV

def get_min_mR(m_phi, f_eff):
    try:
        sol = root_scalar(lambda mR: np.log10(to_seconds(calc_decay_time(mR, m_phi, f_eff))) - constants.log_min_decay_time, bracket=(0, 15))
    except ValueError:
        return np.nan
    if sol.converged:
        return sol.root
    else:
        return np.nan

################################ detection ##############################
#e_sq = (constants.g_1 * constants.g_2)**2 / (constants.g_1**2 + constants.g_2**2)

#def calc_axion_photon_coupling(mR, f_eff):
#    eps = calc_eps(mR)
#    f = calc_f(f_eff, eps)
#    return e_sq * eps / (16*np.pi**2 * f)
