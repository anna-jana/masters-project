import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.special import ellipj, ellipk, ellipkinc, ellipkm1
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
def calc_u(phi_over_f, eps):
    return eps * sc(phi_over_f / 2, 1 - eps**2)

def calc_V_eff_over_f_sq(phi_over_f, eps, M):
    return M**2 * 2 / (1 + 1 / calc_u(phi_over_f, eps)**2)

def theta_to_phi_over_f(theta, eps):
    return 2 / eps * ellipkinc(theta / 2, - 1 / eps**2)

def calc_du_dphi_times_f(phi_over_f, eps):
    _sn, cn, dn, _ph = ellipj(phi_over_f / 2, 1 - eps**2)
    return eps / 2 * dn / cn**2

def calc_dV_eff_dphi_over_f(phi_over_f, eps, M):
    u = calc_u(phi_over_f, eps)
    return 4 * M**2 * calc_du_dphi_times_f(phi_over_f, eps) / (u**3 * (1 + 1 / u**2)**2)

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
    sol = solve_ivp(rhs, (t_start, t_end), initial, t_eval=steps, method="Radau", rtol=1e-5, args=args)
    assert sol.success
    return sol

def compute_relic_density(field_initial_over_f, T_initial, t_initial, f, mR, M,
                          num_osc_per_step=5, convergence_epsilon=1e-2, debug=False, max_steps=50):
    T_fn_rad_dom, H_fn_rad_dom = cosmology.make_rad_dom_H_fn(t_initial, T_initial)
    m_phi = calc_m_phi(mR, M)
    t_osc = cosmology.switch_hubble_and_time_rad_dom(m_phi)
    Delta_t = 2*np.pi / m_phi * num_osc_per_step
    eps = calc_eps(mR)
    args = (eps, M, H_fn_rad_dom)
    sol = evolve(t_initial, t_osc, field_initial_over_f, args)
    if debug:
        plt.plot(sol.t, sol.y[0])
    change = 1
    step = 0
    last = -1
    while True:  # change > convergence_epsilon:
        t_start = sol.t[-1]
        t_end = t_start + Delta_t
        t_steps = np.linspace(t_start, t_end, num_osc_per_step * 10)
        t_steps[0] = t_start; t_steps[-1] = t_end
        sol = evolve(t_start, t_end, sol.y[:, -1], args, steps=t_steps)
        Y = calc_abundance(*sol.y, T_fn_rad_dom(sol.t), eps, mR, f, M)
        is_max = np.where((Y[:-2] < Y[1:-1]) & (Y[2:] < Y[1:-1]))[0]
        is_min = np.where((Y[:-2] > Y[1:-1]) & (Y[2:] > Y[1:-1]))[0]
        if len(is_max) > 0 and len(is_min) > 0:
            Y_min, Y_max = Y[is_min[-1] + 1], Y[is_max[-1] + 1]
        else:
            if debug:
                print("no oscillations:", np.min(sol.y[0] / f))
                plt.plot(sol.t, sol.y[0])
            step += 1
            if max_steps is not None and step > max_steps:
                break
            else:
                continue
            # Y_max, Y_min = np.max(Y), np.min(Y)
        Y_mean = (Y_max + Y_min) / 2
        if debug:
            plt.plot(sol.t, Y)
            plt.xscale("log")
            plt.yscale("log")
        if step > 0:
            change = np.abs(Y_mean - last) / Y_mean
            if debug:
                print("change:", change, "Y:", Y_mean, "convergence_eps:", convergence_epsilon)
            if change < convergence_epsilon:
                break
            else:
                last = Y_mean
        step += 1
        if max_steps is not None and step > max_steps:
            raise RuntimeError(f"iteration for relic density took too long. last change: {change}")
    n_today = Y_mean * cosmology.calc_entropy_density(constants.T_CMB, constants.g_star_0)
    rho_today = m_phi * n_today
    Omega_h_sq = rho_today * (1e9)**4 / constants.rho_c * constants.h**2
    return Omega_h_sq

def compute_observables(m_phi, mR, f_eff, Gamma_phi, H_inf, debug=False, relic_kwargs={}):
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
            axion_initial=(theta_to_phi_over_f(1.0, eps), 0),
            Gamma_phi=Gamma_phi,
            H_inf=H_inf,
        )
        red_chem_pot_B_minus_L, T_final, axion_final = model.solve(m, debug=debug)
        if debug:
            print("baryogenesis done")
        t_final = cosmology.switch_hubble_and_time_rad_dom(cosmology.calc_hubble_parameter(cosmology.calc_radiation_energy_density(T_final)))
        eta_B = cosmology.calc_eta_B_final(red_chem_pot_B_minus_L, T_final)
        Omega_a_h_sq = compute_relic_density(axion_final, T_final, t_final, f, mR, M, debug=debug, **relic_kwargs)
        if debug:
            print("relic density done")
        return eta_B, Omega_a_h_sq
    except Exception as e:
        print(e)
        return np.nan, np.nan
