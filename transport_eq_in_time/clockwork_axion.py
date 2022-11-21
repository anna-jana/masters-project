import importlib, os, itertools
import numpy as np, matplotlib.pyplot as plt
import tqdm
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar
from scipy.special import ellipj, ellipk, ellipkinc, ellipkm1
from scipy.constants import hbar, electron_volt
import axion_motion, generic_alp, util, transport_equation, decay_process, observables, axion_decay_rate
axion_motion, generic_alp, util, transport_equation, decay_process, observables, axion_decay_rate = \
    map(importlib.reload, (axion_motion, generic_alp, util, transport_equation, decay_process, observables, axion_decay_rate))

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

def compute_max_mR(m_phi, H_inf, theta_i):
    sol = root_scalar(lambda mR: is_pot_curvature_too_large(mR, m_phi, H_inf, theta_i) / H_inf, bracket=(0, 20))
    return sol.root

####################### check if the axion might decay (then it cant be dark matter) ###################
def calc_decay_time(mR, m_phi, f_eff, source_vector):
    eps = calc_eps(mR)
    f = calc_f(f_eff, eps)
    decay_rate = axion_decay_rate.get_axion_decay_rate(source_vector, f_eff, m_phi)
    return 1 / decay_rate

def to_seconds(natural_time_in_GeV):
    return hbar * 1/electron_volt * 1e-9 * natural_time_in_GeV

log_min_decay_time = 26 # [seconds]

def compute_min_mR(m_phi, f_eff, source_vector):
    try:
        sol = root_scalar(lambda mR: np.log10(to_seconds(calc_decay_time(mR, m_phi, f_eff, source_vector))) - log_min_decay_time, bracket=(0, 15))
    except ValueError:
        return np.nan
    if sol.converged:
        return sol.root
    else:
        return np.nan

default_f_eff = 1e12 # arbitary value since only Omega depends on f_eff and it is ~ f^2

################################ detection ##############################
#e_sq = (constants.g_1 * constants.g_2)**2 / (constants.g_1**2 + constants.g_2**2)

def calc_axion_photon_coupling(mR, f_eff):
    eps = calc_eps(mR)
    f = calc_f(f_eff, eps)
    return e_sq * eps / (16*np.pi**2 * f)


####################################### postprocessing #################################
def compute_example_trajectory(H_inf, Gamma_inf, nsource, f, m_phi, mR):
    m_phi /= 1e9
    eps = calc_eps(mR)
    M = m_phi / eps
    phi0_over_f = theta_to_phi_over_f(1.0, eps)
    conv_factor = Gamma_inf / clockwork_axion_field.find_dynamical_scale(eps, M)

    H_osc = clockwork_axion_field.find_H_osc(eps, M)
    t_osc = 2 * (1 / H_osc + 1 / (H_inf / M))
    tmax_ax = t_osc + 1/eps*100
    tmax_inf = tmax_ax * conv_factor

    phi0_over_f = theta_to_phi_over_f(1.0, eps)
    _, T_and_H_fn, _ = decay_process.solve(tmax_inf, 0.0, 3*H_inf**2*decay_process.M_pl**2,
                                           decay_process.find_scale(Gamma_inf), Gamma_inf)
    sol = clockwork_axion_field.solve((phi0_over_f, 0.0), (eps, M), tmax_ax, T_and_H_fn, Gamma_inf)
    relic_ts = np.geomspace(sol.t[1], sol.t[-1], 400)
    phi_over_f = sol.sol(relic_ts)[0,:]

    background_sols, axion_sols, red_chem_pot_sols = observables.compute_observables(
                        H_inf, Gamma_inf, (eps, M), f, clockwork_axion_field,
                        (phi0_over_f, 0), source_vector_axion=transport_equation.source_vectors[nsource],
                        calc_init_time=True, return_evolution=True, isocurvature_check=False)
    collected = []
    tstart = 0
    for red_chem_pot_sol, axion_sol, T_and_H_and_T_dot_fn in zip(red_chem_pot_sols, axion_sols, background_sols):
        tmax_axion = axion_sol.t[-1]
        tmax_inf = tmax_axion * conv_factor
        tinfs = np.geomspace(decay_process.t0, decay_process.t0 + tmax_inf, 300)
        taxs = (tinfs - decay_process.t0) / conv_factor
        plot_ts = tstart + tinfs
        tstart += tmax_inf
        red_chem_pots = red_chem_pot_sol(np.log(tinfs))
        B_minus_L = transport_equation.calc_B_minus_L(red_chem_pots)
        Ts, Hs, _ = T_and_H_and_T_dot_fn(tinfs)
        theta_dots = axion_sol.sol(taxs)[1, :]
        gammas = [transport_equation.calc_rate_vector(T) for T in Ts]
        source =  (
            - theta_dots * clockwork_axion_field.find_dynamical_scale(eps, M) / Ts *
            [gamma @ transport_equation.source_vectors[nsource] for gamma in gammas]
        )
        rate = - np.array([gamma @ transport_equation.charge_vector @ transport_equation.charge_vector_B_minus_L
                    for gamma in gammas]) / Hs
        collected.append((B_minus_L, source, rate, plot_ts))
    B_minus_L = np.concatenate([x[0] for x in collected])
    source = np.concatenate([x[1] for x in collected])
    rate = np.concatenate([x[2] for x in collected])
    ts = np.concatenate([x[3] for x in collected])

    return B_minus_L, source, rate, ts, phi_over_f, relic_ts

interesting_points = [(1e-3, 13), (1e3, 10), (1e1, 2), (1e-2, 8)]
example_trajectories_filename = os.path.join(util.datadir, "example_trajectories_cw.pkl")

def compute_all_example_trajectories():
    source_vectors = [1] # source vector
    Gamma_inf_indicies = [2] # TODO loop
    output = {}
    for v, Gamma_inf_index in itertools.product(source_vectors, Gamma_inf_indicies):
        data = util.load_data("clockwork_mR_vs_mphi", v)
        H_inf = data["H_inf"][0]
        nsource = int(data["nsource_vector"][0])
        source_name = transport_equation.source_vector_names[nsource]
        f = 1e12
        Gamma_inf = data["Gamma_inf"][Gamma_inf_index]
        interesting_solutions = [compute_example_trajectory(H_inf, Gamma_inf, nsource, f, *p)
                                for p in tqdm.tqdm(interesting_points)]
        output[(v, Gamma_inf_index)] = interesting_solutions
    util.save_pkl(output, example_trajectories_filename)
