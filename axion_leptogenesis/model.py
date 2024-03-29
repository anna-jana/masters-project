# based on arXiv:1412.2043v2
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from collections import namedtuple
from numba import jit

sys.path.append("..")
from common.constants import *
from common.cosmology import *
from common.rh_neutrino import *
from common import util

# all units should be natural units and in GeV unless otherwise noted

global_epsilon = 1e-3 # global default relative error for convergence check

@jit(nopython=True)
def calc_end_time(m_a, Gamma_phi, num_osc, larger_than_reheating_by):
    t_osc = 1/(2*m_a) if Gamma_phi >= m_a else 2/(3*m_a)
    axion_period = 2*np.pi/m_a
    t_axion = t_osc + axion_period * num_osc
    t_reheating = 1 / Gamma_phi * larger_than_reheating_by
    return t_axion # max((t_axion, t_reheating))

# Analytical Solution
def calc_Delta_a(m_a, f_a, Gamma_phi, theta0):
    a0 = theta0 * f_a
    Delta_a_prime = 2*np.pi**2 / alpha * f_a * a0**2 / (m_a * M_pl**2) * min((1, (Gamma_phi / m_a)**0.5))
    Delta_a = max((1, Delta_a_prime))
    return Delta_a

def compute_B_asymmetry_analytic(m_a, f_a, Gamma_phi, sigma_eff=paper_sigma_eff, theta0=1):
    Delta_phi_prime = (m_a / Gamma_phi)**(5 / 4)
    Delta_phi = max((1, Delta_phi_prime))
    Delta_a = calc_Delta_a(m_a, f_a, Gamma_phi, theta0)
    a0 = f_a * theta0
    eta_L_max = sigma_eff * a0 / (g_star**0.5 * f_a) * m_a * M_pl * min((1, (Gamma_phi / m_a)**0.5))
    T_RH = 2e13*(Gamma_phi / 1e9)**0.5
    T_L = g_star**0.5 / (np.pi * M_pl * sigma_eff)
    kappa = np.where(m_a > Gamma_phi, T_RH / T_L, 0)
    C = np.where(m_a > Gamma_phi, 2.2, 1.5) # factor determined in paper
    eta_L_a = C * Delta_a**-1 * Delta_phi**-1 * eta_L_max * np.exp(-kappa)
    return eta_L_a_to_eta_B_0(eta_L_a)


# Numerical Simulation
SimulationResult = namedtuple("SimulationResult",
    ["t", "rho_phi", "rho_R", "rho_tot", "T", "H", "R", "theta", "theta_dot", "n_L"])

## numerical implementation of the complete model
theta_index = 3
theta_diff_index = theta_index + 1
n_L_index = theta_diff_index + 1
R_osc = 1.0

p = 8

def calc_hidden_sector_scale(m_a, f_a):
    return np.sqrt(m_a * f_a) # Lambda^2 / f_a = m_a, ignore the prefactor

# returns the axions mass squared"
@jit(nopython=True)
def calc_axion_power_law_mass_squared(T, m_a, Lambda, p):
    return m_a**2 if T < Lambda else m_a**2 * (T / Lambda)**(-p)

def make_rhs(use_cosine_potential, use_temp_dep_axion_mass):
    @jit(nopython=True)
    def rhs(log_t, y, Gamma_phi, m_a, sigma_eff, Lambda, mu_eff_prefactor):
        # coordinate transformation
        t = np.exp(log_t)
        rho_phi, rho_tot, R = np.exp(y[:theta_index])
        theta = y[theta_index]
        d_theta_d_log_t = y[theta_diff_index]
        theta_dot = d_theta_d_log_t / t
        rho_R = calc_rho_R(rho_phi, rho_tot)
        T = calc_temperature(rho_R)
        n_L = y[n_L_index]

        # Friedmann
        H = calc_hubble_parameter(rho_tot)
        d_log_R_d_log_t = t * H

        # reheating energy equations rewritten in rho_phi and roh_tot instead of rho_phi and phi_R and in loglog space
        d_log_rho_phi_d_log_t = - t * (3 * H + Gamma_phi)
        d_log_rho_tot_d_log_t = - H * t * (4 - rho_phi / rho_tot)

        # axion eom (Klein Gordon) in theta and log t
        if use_cosine_potential:
            U = np.sin(theta)
        else:
            U = theta
        if use_temp_dep_axion_mass:
            M = calc_axion_power_law_mass_squared(T, m_a, Lambda, p)
        else:
            M = m_a**2

        theta_dot2         = - 3 * H * theta_dot - M * U
        d2_theta_d_log_t_2 = d_theta_d_log_t + t**2 * theta_dot2

        # Boltzmann eq. for lepton asymmetry
        mu_eff = mu_eff_prefactor * theta_dot
        n_L_eq = calc_lepton_asym_in_eqi(T, mu_eff)
        Gamma_L = calc_Gamma_L(T, sigma_eff)
        d_n_L_d_log_t = t * (- 3 * H * n_L - Gamma_L * (n_L - n_L_eq))

        # final result
        return (
            d_log_rho_phi_d_log_t, d_log_rho_tot_d_log_t,
            d_log_R_d_log_t,
            d_theta_d_log_t, d2_theta_d_log_t_2,
            d_n_L_d_log_t,
        )
    return rhs

rhss = {(use_cosine_potential, use_temp_dep_axion_mass) : make_rhs(use_cosine_potential, use_temp_dep_axion_mass)
        for use_cosine_potential in (True, False) for use_temp_dep_axion_mass in (True, False)}


def simulate(m_a, f_a, Gamma_phi, H_inf,
             theta0=1.0, sigma_eff=paper_sigma_eff, use_cosine_potential=False, use_temp_dep_axion_mass=False, mu_eff_prefactor=1.0,
             start=None, end=None, num_osc=15, larger_than_reheating_by=5, solver="DOP853",
             samples=500, fixed_samples=True, converge=True, convergence_epsilon=global_epsilon, debug=False):
    # setup
    # integration interval
    if start is None: start = calc_start_time(H_inf)
    if end is None: end = calc_end_time(m_a, Gamma_phi, num_osc, larger_than_reheating_by)
    interval = (start, end)
    # lookup right hand side
    rhs = rhss[use_cosine_potential, use_temp_dep_axion_mass]
    # initial condtions
    rho_phi_0 = calc_energy_density_from_hubble(H_inf)
    initial_conditions = np.array([np.log(rho_phi_0), np.log(rho_phi_0), np.log(R_osc), theta0, 0.0, 0.0])
    # step
    axion_periode = 2*np.pi / m_a
    # arrays for integration step collection
    ys = [np.array([initial_conditions]).T] # np.array([initial_conditions]).T
    ts = [np.array([np.log(start)])] ## np.array([start])
    first = True
    # hidden sector energy scale
    Lambda = calc_hidden_sector_scale(m_a, f_a)

    # integrate until convergence of asymmetry (end of leptogensis)
    while True:
        if debug:
            print("interval:", interval, "initial conditions:", initial_conditions, "arguments:", (Gamma_phi, m_a, sigma_eff, Lambda))
        sol = solve_ivp(rhs, np.log(interval), initial_conditions,
                        args=(Gamma_phi, m_a, sigma_eff, Lambda, mu_eff_prefactor),
                        t_eval=np.log(np.geomspace(*interval, samples))[:-1] if fixed_samples else None,
                        method=solver)
        # collect integration steps
        ys.append(sol.y[:, 1:])
        ts.append(sol.t[1:])

        # stop the loop once we are done
        if converge:
            interval = (np.exp(sol.t[-1]), np.exp(sol.t[-1]) + axion_periode * num_osc)
            initial_conditions = sol.y[:, -1]
            if first: # reduce number of samples in the integration result once we start to converge
                samples = max((samples // 10, 10))
                first = False
            else:
                n_L = sol.y[n_L_index]
                rho_phi, rho_tot = np.exp(sol.y[:theta_index - 1])
                T = calc_temperature(calc_rho_R(rho_phi, rho_tot))
                eta_B = n_L_to_eta_B_final(T, n_L)
                i = np.argmax(eta_B)
                j = np.argmin(eta_B)
                t = np.exp(sol.t)
                delta = np.abs((eta_B[i] - eta_B[j]) / ((eta_B[i] + eta_B[j]) / 2))
                if debug:
                    print("convergence:", delta, "vs", convergence_epsilon)
                if delta < convergence_epsilon:
                    break # stop once convergence criterion is fulfilled
        else:
            break # dont use the convergence loop if converge == False

    # final result
    t = np.exp(np.concatenate(ts))
    y = np.hstack(ys)
    rho_phi, rho_tot, R = np.exp(y[:theta_index])
    theta, n_L = y[theta_index], y[n_L_index]
    theta_dot = y[theta_diff_index] / t
    rho_R = calc_rho_R(rho_phi, rho_tot)
    T = calc_temperature(rho_R)
    H = calc_hubble_parameter(rho_tot)
    return SimulationResult(t=t, rho_R=rho_R, rho_phi=rho_phi, rho_tot=rho_tot, H=H, R=R, T=T, theta=theta, theta_dot=theta_dot, n_L=n_L)


# ## Axion Decay and Entropy Production
def rhs_axion_decay(log_t, y, Gamma_a):
    rho_R, rho_a, R = np.exp(y)
    t = np.exp(log_t)
    H = calc_hubble_parameter(rho_a + rho_R)
    #assert rho_R != 0.0,  f"t = {t}, rho_a = {rho_a}"
    d_log_rho_R_d_log_t = - t * (4 * H - Gamma_a * rho_a / rho_R)
    d_log_rho_a_d_log_t = - t * (3 * H + Gamma_a)
    d_log_R_d_log_t = t * H
    return d_log_rho_R_d_log_t, d_log_rho_a_d_log_t, d_log_R_d_log_t

R_0 = 1.0

AxionDecayResult = namedtuple("AxionDecayResult", ["t", "rho_R", "rho_a", "R", "T", "n_L"])

def simulate_axion_decay(m_a, f_a, bg_sol, end=None, solver="Radau", debug=False, calc_Gamma_a_fn=calc_Gamma_a_SU2,
                         use_cosine_potential=False, use_temp_dep_axion_mass=False, log_time_step=1, initial_end_factor=1e1,
                         samples=500, fixed_samples=True, converge=False, convergence_epsilon=global_epsilon):
    # integration range
    Gamma_a = calc_Gamma_a_fn(m_a, f_a)
    start = np.log(bg_sol.t[-1])
    t_axion_decay = 1 / Gamma_a
    end = np.log(end) if end is not None else np.log(t_axion_decay * initial_end_factor)
    interval = (start, end)
    if start >= end:
        end += 1

    # intitial conditions
    rho_kin = 0.5 * f_a**2 * bg_sol.theta_dot[-1]**2
    if use_cosine_potential:
        U = 1 - np.cos(bg_sol.theta[-1])
    else:
        U = 0.5 * bg_sol.theta[-1]**2
    if use_temp_dep_axion_mass:
        M = calc_axion_power_law_mass_squared(bg_sol.T[-1], m_a, Lambda, p)
    else:
        M = m_a**2
    rho_a_initial = rho_kin + f_a**2 * M * U
    rho_R_initial = bg_sol.rho_R[-1]
    initial_conditions = (np.log(rho_R_initial), np.log(rho_a_initial), np.log(R_0))

    sols = []

    def calc_n_L_during_decay(R):
        n_L_start = bg_sol.n_L[-1]
        n_L = n_L_start * (R_0 / R)**3
        return n_L

    while True:
        # solve eq. system
        if debug:
            print("integrating:", interval, initial_conditions)
        points = np.linspace(*interval, samples) if fixed_samples else None
        axion_decay_sol = solve_ivp(rhs_axion_decay, interval, initial_conditions,
                        args=(Gamma_a,), t_eval=points, method=solver)

        sols.append(axion_decay_sol)

        # convergence check and final result
        if converge:
            t = np.exp(axion_decay_sol.t[-3:])
            rho_R, rho_a, R = np.exp(axion_decay_sol.y[:, -3:])
            T = calc_temperature(rho_R)
            n_L = calc_n_L_during_decay(R)
            eta_B = n_L_to_eta_B_final(T, n_L)
            deriv = (eta_B[-1] - eta_B[-2]) / (t[-1] - t[-2])
            deriv2 = (eta_B[-3] - 2 * eta_B[-2] + eta_B[-1]) / (t[-1] - t[-2])**2
            delta = np.abs(deriv / eta_B[-1] * t[-1])
            delta2 = np.abs(deriv2 / eta_B[-1] * t[-1]**2)
            if debug:
                print("convergence:", delta, delta2, "has to be <", convergence_epsilon)
            if delta < convergence_epsilon and delta2 < convergence_epsilon:
                break
        else:
            break

        # compute next integration step interval
        initial_conditions = axion_decay_sol.y[:, -1]
        interval = (axion_decay_sol.t[-1], axion_decay_sol.t[-1] + log_time_step)

    t = np.exp(np.concatenate([sol.t for sol in sols]))
    rho_R = np.exp(np.concatenate([sol.y[0] for sol in sols]))
    rho_a = np.exp(np.concatenate([sol.y[1] for sol in sols]))
    R = np.exp(np.concatenate([sol.y[2] for sol in sols]))
    T = calc_temperature(rho_R)
    n_L = calc_n_L_during_decay(R)

    return AxionDecayResult(t=t, rho_R=rho_R, rho_a=rho_a, R=R, T=T, n_L=n_L)


# Final $\eta_B$ numerical
def compute_B_asymmetry(m_a, f_a, Gamma_phi, H_inf, do_decay=True, bg_kwargs={}, decay_kwargs={}):
    # leptogenesis process
    bg_options = dict(theta0=1.0, debug=False, fixed_samples=False)
    bg_options.update(bg_kwargs)
    bg_res = simulate(m_a, f_a, Gamma_phi, H_inf, **bg_options)
    # decay process of the axion
    if do_decay:
        decay_options = dict(debug=False, fixed_samples=False, converge=True)
        decay_options.update(decay_kwargs)
        axion_decay_res = simulate_axion_decay(m_a, f_a, bg_res, **decay_options)
        return n_L_to_eta_B_final(axion_decay_res.T[-1], axion_decay_res.n_L[-1])
    else:
        return n_L_to_eta_B_final(bg_res.T[-1], bg_res.n_L[-1])
