import numpy as np
from scipy.integrate import solve_ivp

M_pl = 2.435e18 # reduced Planck mass [GeV] from wikipedia (https://en.wikipedia.org/wiki/Planck_mass)
g_star = 427/4 # during reheating from paper
convergence_epsilon = 1e-3
zeta3 = 1.20206
g_photon = 2
R_0 = R0 = 1.0
g_star_0 = 43/11 # from paper
C_sph = 8 / 23


def calc_H(rho_a, rho_R):
    return np.sqrt(rho_R + rho_a) / (M_pl * np.sqrt(3))

def rhs_axion_decay(log_t, y, Gamma_a):
    rho_R, rho_a, R = np.exp(y)
    t = np.exp(log_t)
    H = calc_H(rho_a, rho_R)
    d_log_rho_R_d_log_t = - t * (4 * H - Gamma_a * rho_a / rho_R)
    d_log_rho_a_d_log_t = - t * (3 * H + Gamma_a)
    d_log_R_d_log_t = t * H
    return d_log_rho_R_d_log_t, d_log_rho_a_d_log_t, d_log_R_d_log_t

# TODO: better units

def compute_axion_decay(T_start, red_chem_B_minus_L, theta, theta_dot, m_a, f_a, axion_decay_rate):
    # initial condition
    rho_R   = np.pi**2/30 * g_star * T_start**4
    rho_kin = 0.5 * f_a**2 * theta_dot**2
    rho_pot = 0.5 * f_a**2 * m_a**2 * theta**2
    rho_a   = rho_kin + rho_pot
    initial = np.log([rho_R, rho_a, R_0])
    n_B_start = - g_star_0 / g_star * C_sph * T_start**3 / 6 * red_chem_B_minus_L
    H = calc_H(rho_a, rho_R)
    t_start = 1 / (2*H)
    t_end = 10 / axion_decay_rate
    start = np.log(t_start)
    end = np.log(t_end)
        
    # calc eta_B
    def calc_eta(rho_R, R):
        T = (rho_R / (np.pi**2 / 30 * g_star))**(1/4)
        n_B = n_B_start * (R_0 / R)**3
        n_gamma = zeta3 / np.pi**2 * g_photon * T**3 
        eta_B = n_B / n_gamma 
        return eta_B
    eta_B_start = calc_eta(rho_R, R0)

    # convergence loop
    while True:
        print("interval:", start, end)
        ts = np.linspace(start, end, 100)
        ts[0] = start
        ts[-1] = end
        sol = solve_ivp(rhs_axion_decay, (start, end), initial, args=(axion_decay_rate,), 
                        method="Radau", rtol=1e-5, t_eval=ts)
        assert sol.success
        rho_R, _, R = np.exp(sol.y)
        eta_B = calc_eta(rho_R, R)
        delta = np.abs((np.max(eta_B) - np.min(eta_B)) / eta_B[-1])
        print("delta =", delta, "vs", convergence_epsilon)
        if delta < convergence_epsilon:
            return eta_B[-1] / eta_B_start
        initial = sol.y[:, -1]
        t_start, t_end = t_end, t_end + 1 / axion_decay_rate
        start = np.log(t_start)
        end = np.log(t_end)

# HOW DID THIS EVER WORKED???????????????????       
# import numpy as np
# from scipy.integrate import solve_ivp
# 
# M_pl = 2.435e18 # reduced Planck mass [GeV] from wikipedia (https://en.wikipedia.org/wiki/Planck_mass)
# g_star = 427/4 # during reheating from paper
# 
# def calc_H(rho_a, rho_R):
#     return np.sqrt(rho_R + rho_a) / (M_pl * np.sqrt(3))
# 
# def rhs_axion_decay(log_t, y, Gamma_a):
#     rho_R, rho_a, R = np.exp(y)
#     t = np.exp(log_t)
#     H = calc_H(rho_a, rho_R)
#     d_log_rho_R_d_log_t = - t * (4 * H - Gamma_a * rho_a / rho_R)
#     d_log_rho_a_d_log_t = - t * (3 * H + Gamma_a)
#     d_log_R_d_log_t = t * H
#     return d_log_rho_R_d_log_t, d_log_rho_a_d_log_t, d_log_R_d_log_t
# 
# convergence_epsilon = 1e-3
# 
# def compute_axion_decay(T_start, red_chem_B_minus_L, theta, theta_dot, m_a, f_a, axion_decay_rate):
#     # initial condition
#     R_0 = 1.0
#     rho_R   = np.pi**2/30 * g_star * T_start**4
#     rho_kin = 0.5 * f_a**2 * theta_dot**2
#     rho_pot = 0.5 * f_a**2 * m_a**2 * theta**2
#     rho_a   = rho_kin + rho_pot
#     initial = np.log([rho_R, rho_a, R_0])
# 
#     # we start a some fake time (not the cosmological one)
#     # cosmology.red_chem_pot_to_B_density_final(red_chem_B_minus_L, T_start)
#     n_B_start = - g_star_0 / g_star * C_sph * T_start**3 / 6 * red_chem_B_minus_L
#     H = calc_H(rho_a, rho_R)
#     t_start = 1 / (2*H)
#     t_end = 10 / axion_decay_rate
#     start = np.log(t_start)
#     end = np.log(t_end)
# 
#     step = 0
#     last_eta_B = np.nan
#     while True:
#         # TODO: wtf we never reset initial?????????
#         sol = solve_ivp(rhs_axion_decay, (start, end), initial, args=(axion_decay_rate,),
#                 method="Radau", rtol=1e-5)
#         if not sol.success:
#             #print("%e" % f_a, start, end, step)
#             return last_eta_B # TODO: wtf
# 
#         t = np.exp(sol.t[-3:])
#         rho_R, rho_a, R = np.exp(sol.y[:, -3:])
#         T = (rho_R / (np.pi**2 / 30 * g_star))**(1/4) # cosmology.calc_temperature(rho_R)
#         n_B = n_B_start * (R_0 / R)**3
#         zeta3 = 1.20206
#         g_photon = 2
#         n_gamma = zeta3 / np.pi**2 * g_photon * T**3 # K&T (3.52)
#         eta_B = n_B / n_gamma # definition calc_asym_parameter(T, n_B)
# 
#         try:
#             deriv = (eta_B[-1] - eta_B[-2]) / (t[-1] - t[-2])
#             deriv2 = (eta_B[-3] - 2 * eta_B[-2] + eta_B[-1]) / (t[-1] - t[-2])**2
#             delta = np.abs(deriv / eta_B[-1] * t[-1])
#             delta2 = np.abs(deriv2 / eta_B[-1] * t[-1]**2) # TODO: wtf
#         except IndexError:
#             return np.nan # TODO: wtf
#         if delta < convergence_epsilon and delta2 < convergence_epsilon:
#             return eta_B[-1]
#         else:
#             last_eta_B = eta_B[-1] # TODO wtf this is never used except for the case of failure???????
# 
#         start, end = end, end - 0.5 # why do the decrease end??????
#         step += 1