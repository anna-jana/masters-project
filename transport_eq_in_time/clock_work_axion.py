import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar
from scipy.special import ellipj, ellipk, ellipkinc, ellipkm1
from scipy.constants import hbar, electron_volt
import model
import transport_equation
from common import constants, cosmology

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

##################################### define the rhs ####################################
def rhs(t, y, eps, M, H_fn):
    phi_over_f, phi_dot_over_f = y
    H = H_fn(t)
    phi_dot_dot_over_f = - 3 * H * phi_dot_over_f - calc_dV_eff_dphi_over_f(phi_over_f, eps, M)
    return phi_dot_over_f, phi_dot_dot_over_f

def rhs_log_t(log_t, y, _T_fn, H_fn, p):
    eps, M = p
    t = np.exp(log_t)
    phi_dot_over_f, phi_dot_dot_over_f = rhs(t, y, eps, M, H_fn)
    return phi_dot_over_f * t, phi_dot_dot_over_f * t

#################### caclulate theta for coupling to the (weak) shaleron #################
def calc_theta_dot(phi_over_f, phi_dot_over_f, eps, M):
    MM = 2*M**2
    A = calc_V_eff_over_f_sq(phi_over_f, eps, M) / MM
    return (
        1 / MM * np.abs(calc_dV_eff_dphi_over_f(phi_over_f, eps, M)) * phi_dot_over_f
        / ((1 - A)*A)**0.5
    )

def get_axion_source_clockwork(field_fn, p):
    eps, M = p
    def source_fn(log_t):
        phi_over_f, phi_dot_over_f = field_fn(log_t)
        return calc_theta_dot(phi_over_f, phi_dot_over_f, eps, M)
    return source_fn

############################### relic density computation ################################
def calc_abundance(phi_over_f, phi_dot_over_f, T, eps, mR, f, M):
    rho = f**2 * (0.5 * phi_dot_over_f**2 + calc_V_eff_over_f_sq(phi_over_f, eps, M))
    m_phi = calc_m_phi(mR, M)
    n = rho / m_phi
    s = cosmology.calc_entropy_density(T)
    return n / s

def evolve(t_start, t_end, initial, args, steps=None):
    sol = solve_ivp(rhs, (t_start, t_end), initial, t_eval=steps, method="RK45", rtol=1e-8, args=args)
    assert sol.success
    return sol

def compute_relic_density(field_initial_over_f, T_initial, t_initial, f, mR, M,
                          num_osc_per_step=5, convergence_epsilon=1e-2, debug=False, max_steps=50):
    # integrate unitl t_osc ~ about the start of oscillations
    T_fn_rad_dom, H_fn_rad_dom = cosmology.make_rad_dom_H_fn(t_initial, T_initial)
    m_phi = calc_m_phi(mR, M)
    t_osc = cosmology.switch_hubble_and_time_rad_dom(m_phi)
    Delta_t = 2*np.pi / m_phi * num_osc_per_step
    eps = calc_eps(mR)
    args = (eps, M, H_fn_rad_dom)
    sol = evolve(t_initial, t_osc, field_initial_over_f, args)
    step = 0
    last = -1
    if debug:
        plt.subplot(2,1,1)
        plt.semilogx(sol.t, sol.y[0])
        plt.subplot(2,1,2)
        plt.loglog(sol.t, calc_abundance(*sol.y, T_fn_rad_dom(sol.t), eps, mR, f, M))
    # main converence loop
    while True:
        # now integrate num_osc_per_step oscillations at the time
        t_start = sol.t[-1]
        t_end = t_start + Delta_t
        t_steps = np.linspace(t_start, t_end, num_osc_per_step * 10)
        t_steps[0] = t_start; t_steps[-1] = t_end
        sol = evolve(t_start, t_end, sol.y[:, -1], args, steps=t_steps)
        if debug:
            plt.subplot(2,1,1)
            plt.semilogx(sol.t, sol.y[0])
            plt.subplot(2,1,2)
            plt.loglog(sol.t, calc_abundance(*sol.y, T_fn_rad_dom(sol.t), eps, mR, f, M))
        # get n/s
        Y = calc_abundance(*sol.y, T_fn_rad_dom(sol.t), eps, mR, f, M)
        # find the local minima and maxima in the solution
        is_max = np.where((Y[:-2] < Y[1:-1]) & (Y[2:] < Y[1:-1]))[0]
        is_min = np.where((Y[:-2] > Y[1:-1]) & (Y[2:] > Y[1:-1]))[0]
        if len(is_max) > 0 and len(is_min) > 0:
            Y_min, Y_max = Y[is_min[-1] + 1], Y[is_max[-1] + 1]
        else:
            # if no local extrema are found then the hasn't started oscillating yet.
            # We skip the convergence check and continue with the next integration interval.
            if debug:
                print("no oscillations:", "Y =", Y[-1], "t_end =", t_end)
            step += 1
            if max_steps is not None and step > max_steps:
                break
            else:
                continue
        # check if the relative change of the mean value of Y between this and the last interval
        # is below our threshold.
        Y_mean = (Y_max + Y_min) / 2
        if last is not None:
            change = np.abs(Y_mean - last) / Y_mean
            if debug:
                print("change:", change, "Y:", Y_mean, "convergence_eps:", convergence_epsilon)
            if change < convergence_epsilon:
                break
            else:
                last = Y_mean
        step += 1
        if max_steps is not None and step > max_steps:
            raise RuntimeError(f"iteration for relic density took too long. last change:") #  {change}")
    # once we are converged redshift the abundance to today and compute the relic density as
    # the density parameter
    n_today = Y_mean * cosmology.calc_entropy_density(constants.T_CMB, constants.g_star_0)
    rho_today = m_phi * n_today
    Omega_h_sq = rho_today * (1e9)**4 / constants.rho_c * constants.h**2 # includes conversion between eV and GeV since rho_c is in eV
    return Omega_h_sq

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

###################################### compute all observables ####################################
def compute_observables(m_phi, mR, f_eff, Gamma_phi, H_inf, theta_i=1.0, sbg_kwargs={}, relic_kwargs={}):
    try:
        eps = calc_eps(mR)
        f = calc_f(f_eff, eps)
        M = calc_mass_scale(m_phi, eps)
        m = model.AxionBaryogenesisModel(
            source_vector=transport_equation.source_vector_weak_sphaleron,
            get_axion_source=get_axion_source_clockwork,
            axion_rhs=rhs_log_t,
            calc_axion_mass=calc_mass,
            axion_parameter=(eps, M),
            axion_initial=(theta_to_phi_over_f(theta_i, eps), 0),
            Gamma_phi=Gamma_phi,
            H_inf=H_inf,
        )
        red_chem_pot_B_minus_L, T_final, axion_final = model.solve(m, **sbg_kwargs)
        t_final = cosmology.switch_hubble_and_time_rad_dom(cosmology.calc_hubble_parameter(cosmology.calc_radiation_energy_density(T_final)))
        eta_B = cosmology.calc_eta_B_final(red_chem_pot_B_minus_L, T_final)
        Omega_a_h_sq = compute_relic_density(axion_final, T_final, t_final, f, mR, M, **relic_kwargs)
        return eta_B, Omega_a_h_sq
    except Exception as e:
        print(e)
        return np.nan, np.nan


################################
e_sq = (constants.g_1 * constants.g_2)**2 / (constants.g_1**2 + constants.g_2**2)

def calc_axion_photon_coupling(mR, f_eff):
    eps = calc_eps(mR)
    f = calc_f(f_eff, eps)
    return e_sq * eps / (16*np.pi**2 * f)
