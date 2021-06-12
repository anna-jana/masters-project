import sys
from collections import namedtuple

import numpy as np
from scipy.integrate import solve_ivp
from numba import jit

sys.path.append("..")
from common.constants import *
from common.cosmology import *
from common.rh_neutrino import *
from common import util

# all units should be natural units and in GeV unless otherwise noted

global_epsilon = 1e-2 # global default relative error for convergence check
# we don't need a high precision since the observed asymmetry is also only given by one decimal place 6e-10

# Numerical Simulation
SimulationResult = namedtuple("SimulationResult",
    ["t", "rho_phi", "rho_R", "rho_tot", "T", "H", "R", "theta1", "theta1_dot", "theta2", "theta2_dot", "n_L"])

## numerical implementation of the complete model
log_index = 3

R_osc = 1.0

################################# functions for integration in log t #########################################
@jit(nopython=True)
def axion_field_eom(t, H, theta, d_theta_d_log_t, other_theta, Lambda1, Lambda2, self_coeff1, other_coeff1, self_coeff2, other_coeff2):
    theta_dot = d_theta_d_log_t / t
    V_deriv = (
        self_coeff1 * Lambda1**4 * np.sin(self_coeff1 * theta + other_coeff1 * other_theta) +
        self_coeff2 * Lambda2**4 * np.sin(self_coeff2 * theta + other_coeff2 * other_theta)
    )
    theta_dot_dot = - 3 * H * theta_dot - V_deriv
    d_theta_d_log_t2 = d_theta_d_log_t + t**2 * theta_dot_dot
    return d_theta_d_log_t2

@jit(nopython=True)
def rhs(log_t, y, Gamma_phi, sigma_eff, Lambda1, Lambda2, a11, a12, a21, a22):
    # coordinate transformation
    t = np.exp(log_t)
    rho_phi, rho_tot, R = np.exp(y[:log_index])
    theta_1, d_theta_1_d_log_t, theta_2, d_theta_2_d_log_t, n_L = y[log_index:]

    rho_R = calc_rho_R(rho_phi, rho_tot)
    T = calc_temperature(rho_R)

    # Friedmann
    H = calc_hubble_parameter(rho_tot)
    d_log_R_d_log_t = t * H

    # reheating energy equations rewritten in rho_phi and roh_tot instead of rho_phi and phi_R and in loglog space
    d_log_rho_phi_d_log_t = - t * (3 * H + Gamma_phi)
    d_log_rho_tot_d_log_t = - H * t * (4 - rho_phi / rho_tot)

    # axion eoms
    d2_theta_1_d_log_t_2 = axion_field_eom(t, H, theta_1, d_theta_1_d_log_t, theta_2, Lambda1, Lambda2, a11, a12, a21, a22)
    d2_theta_2_d_log_t_2 = axion_field_eom(t, H, theta_2, d_theta_2_d_log_t, theta_1, Lambda1, Lambda2, a12, a11, a22, a21)

    # Boltzmann eq. for lepton asymmetry
    mu_eff = d_theta_1_d_log_t / t
    n_L_eq = calc_lepton_asym_in_eqi(T, mu_eff)
    Gamma_L = calc_Gamma_L(T, sigma_eff)
    d_n_L_d_log_t = t * (- 3 * H * n_L - Gamma_L * (n_L - n_L_eq))

    # final result
    return (
        d_log_rho_phi_d_log_t, d_log_rho_tot_d_log_t,
        d_log_R_d_log_t,
        d_theta_1_d_log_t, d2_theta_1_d_log_t_2,
        d_theta_2_d_log_t, d2_theta_2_d_log_t_2,
        d_n_L_d_log_t,
    )

######################################## main integration routine #############################################
def simulate(Gamma_phi, H_inf,
             Lambda1, Lambda2, a11, a12, a21, a22,
             sigma_eff=paper_sigma_eff,
             theta0_1=1.0, theta0_2=1.0,
             step=0.5, start=None, end=1e-4,
             solver="DOP853", rtol=1e-5, samples=500, fixed_samples=True,
             converge=True, convergence_epsilon=global_epsilon, debug=False):
    # integration interval
    if start is None: start = calc_start_time(H_inf)
    interval = (start, end)
    # initial condtions
    rho_phi_0 = calc_energy_density_from_hubble(H_inf)
    initial_conditions = np.array([np.log(rho_phi_0), np.log(rho_phi_0), np.log(R_osc), theta0_1, 0.0, theta0_2, 0.0, 0.0])
    # arrays for integration step collection
    ys = [np.array([initial_conditions]).T]
    ts = [np.array([np.log(start)])]
    first = True
    steps_taken = 0

    # integrate until convergence of asymmetry (end of leptogensis)
    while True:
        if debug:
            print("interval:", interval, "initial conditions:", initial_conditions, "arguments:",
                    (Gamma_phi, sigma_eff, Lambda1, Lambda2, a11, a12, a21, a22))
        # steps to be computed by the integrator
        if fixed_samples:
            steps = np.log(np.geomspace(*interval, samples))
            steps[0] = np.log(interval[0])
            steps[-1] = np.log(interval[-1])
        else:
            steps = None
        # integrate the current interval
        sol = solve_ivp(rhs, np.log(interval), initial_conditions,
                        args=(Gamma_phi, sigma_eff, Lambda1, Lambda2, a11, a12, a21, a22),
                        t_eval=steps,
                        method=solver, rtol=rtol)
        # collect integration steps
        ys.append(sol.y[:, 1:])
        ts.append(sol.t[1:])

        # stop the loop once we are done
        if converge:
            interval = (np.exp(sol.t[-1]), np.exp(sol.t[-1] + step))
            initial_conditions = sol.y[:, -1]
            if first: # reduce number of samples in the integration result once we start to converge
                samples = max((samples // 10, 10))
                first = False
            else:
                n_L = sol.y[-1]
                rho_phi, rho_tot = np.exp(sol.y[:log_index - 1])
                T = calc_temperature(calc_rho_R(rho_phi, rho_tot))
                eta_B = n_L_to_eta_B_final(T, n_L)
                i = np.argmax(eta_B)
                j = np.argmin(eta_B)
                delta = np.abs((eta_B[i] - eta_B[j]) / ((eta_B[i] + eta_B[j]) / 2))
                if debug:
                    print("convergence:", delta, "vs", convergence_epsilon)
                if delta < convergence_epsilon:
                    break # stop once convergence criterion is fullfilled
        else:
            break # dont use the convergence loop if converge == False
        steps_taken += 1


    # final result
    t = np.exp(np.concatenate(ts))
    y = np.hstack(ys)

    rho_phi, rho_tot, R = np.exp(y[:log_index])
    theta1, d_theta1_d_log_t, theta2, d_theta2_d_log_t, n_L = y[log_index:]

    theta1_dot = d_theta1_d_log_t / t
    theta2_dot = d_theta2_d_log_t / t

    rho_R = calc_rho_R(rho_phi, rho_tot)
    T = calc_temperature(rho_R)
    H = calc_hubble_parameter(rho_tot)

    return SimulationResult(t=t, rho_R=rho_R, rho_phi=rho_phi, rho_tot=rho_tot, H=H, R=R, T=T,
            theta1=theta1, theta1_dot=theta1_dot,
            theta2=theta2, theta2_dot=theta2_dot,
            n_L=n_L)


####################################### Axion Decay and Entropy Production ###################################
# # TODO: what happens here????
# def rhs_axion_decay(log_t, y, Gamma_a):
#     rho_R, rho_a1, rho_a2, R = np.exp(y)
#     t = np.exp(log_t)
#     H = calc_hubble_parameter(rho_a1 + rho_a2 + rho_R)
#     #assert rho_R != 0.0,  f"t = {t}, rho_a = {rho_a}"
#     d_log_rho_R_d_log_t = - t * (4 * H - Gamma_a * rho_a / rho_R)
#     d_log_rho_a1_d_log_t = - t * (3 * H + Gamma_a)
#     d_log_rho_a2_d_log_t = - t * 3 * H
#     d_log_R_d_log_t = t * H
#     return d_log_rho_R_d_log_t, d_log_rho_a1_d_log_t, d_log_rho_a2_d_log_t, d_log_R_d_log_t
#
# R_0 = 1.0
#
# AxionDecayResult = namedtuple("AxionDecayResult", ["t", "rho_R", "rho_a1", "rho_a2", "R", "T", "n_L"])
#
# def calc_axion_energy_density(theta, theta_dot, other_theta, f_self, f_other, Lambda1, Lambda2, a11, a12, a21, a22):
#     T = 0.5 * f_self**2 * theta_dot**2
#     V = Lambda1**4 * np.cos(a11 * theta1 + a12 * thet
#     return T + V
#
# def simulate_axion_decay(Lambda1, Lambda2, a11, a12, a21, a22, bg_sol, end=None, solver="Radau", debug=False, calc_Gamma_a_fn=calc_Gamma_a_SU2,
#                          log_time_step=1, initial_end_factor=1e1,
#                          samples=500, fixed_samples=True, converge=False, convergence_epsilon=global_epsilon):
#     # integration range
#     Gamma_a = calc_Gamma_a_fn(m_a, f_a)
#     start = np.log(bg_sol.t[-1])
#     t_axion_decay = 1 / Gamma_a
#     end = np.log(end) if end is not None else np.log(t_axion_decay * initial_end_factor)
#     interval = (start, end)
#     if start >= end:
#         end += 1
#
#     # intitial conditions
#     rho_kin = 0.5 * f_a**2 * bg_sol.theta_dot[-1]**2
#     rho_pot = 0.5 * f_a**2 * m_a**2 * bg_sol.theta[-1]**2 + g * bg_sol.theta[-1]**2 * f_a**2 * bg_sol.chi[-1]**2
#     rho_a_initial = rho_kin + rho_pot
#     rho_R_initial = bg_sol.rho_R[-1]
#     initial_conditions = (np.log(rho_R_initial), np.log(rho_a_initial), np.log(R_0))
#
#     sols = []
#
#     def calc_n_L_during_decay(R):
#         n_L_start = bg_sol.n_L[-1]
#         n_L = n_L_start * (R_0 / R)**3
#         return n_L
#
#     while True:
#         # solve eq. system
#         if debug:
#             print("integrating:", interval, initial_conditions)
#         points = np.linspace(*interval, samples) if fixed_samples else None
#         axion_decay_sol = solve_ivp(rhs_axion_decay, interval, initial_conditions,
#                         args=(Gamma_a, p), t_eval=points, method=solver)
#
#         sols.append(axion_decay_sol)
#
#         # convergence check and final result
#         if converge:
#             t = np.exp(axion_decay_sol.t[-3:])
#             rho_R, rho_a, R = np.exp(axion_decay_sol.y[:, -3:])
#             T = calc_temperature(rho_R)
#             n_L = calc_n_L_during_decay(R)
#             eta_B = n_L_to_eta_B_final(T, n_L)
#             deriv = (eta_B[-1] - eta_B[-2]) / (t[-1] - t[-2])
#             deriv2 = (eta_B[-3] - 2 * eta_B[-2] + eta_B[-1]) / (t[-1] - t[-2])**2
#             delta = np.abs(deriv / eta_B[-1] * t[-1])
#             delta2 = np.abs(deriv2 / eta_B[-1] * t[-1]**2)
#             if debug:
#                 print("convergence:", delta, delta2, "has to be <", convergence_epsilon)
#             if delta < convergence_epsilon and delta2 < convergence_epsilon:
#                 break
#         else:
#             break
#
#         # compute next integration step interval
#         initial_conditions = axion_decay_sol.y[:, -1]
#         interval = (axion_decay_sol.t[-1], axion_decay_sol.t[-1] + log_time_step)
#
#     t = np.exp(np.concatenate([sol.t for sol in sols]))
#     rho_R = np.exp(np.concatenate([sol.y[0] for sol in sols]))
#     rho_a = np.exp(np.concatenate([sol.y[1] for sol in sols]))
#     R = np.exp(np.concatenate([sol.y[2] for sol in sols]))
#     T = calc_temperature(rho_R)
#     n_L = calc_n_L_during_decay(R)
#
#     return AxionDecayResult(t=t, rho_R=rho_R, rho_a=rho_a, R=R, T=T, n_L=n_L)
#
#
# # Final $\eta_B$ numerical
# def compute_B_asymmetry(m_a, f_a, Gamma_phi, H_inf, chi0, m_chi, g=1.0, do_decay=True, bg_kwargs={}, decay_kwargs={}):
#     # leptogenesis process
#     bg_options = dict(theta0=1.0, debug=False, fixed_samples=False)
#     bg_options.update(bg_kwargs)
#     bg_res = simulate(m_a, f_a, Gamma_phi, H_inf, chi0, m_chi, g=g, **bg_options)
#     # decay process of the axion
#     if do_decay:
#         decay_options = dict(debug=False, fixed_samples=False, converge=True)
#         decay_options.update(decay_kwargs)
#         axion_decay_res = simulate_axion_decay(m_a, f_a, bg_res, **decay_options)
#         return n_L_to_eta_B_final(axion_decay_res.T[-1], axion_decay_res.n_L[-1])
#     else:
#         return n_L_to_eta_B_final(bg_res.T[-1], bg_res.n_L[-1])
