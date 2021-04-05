import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

from model import *
from util import *
import implicit_curve

def make_time_plots(m_a, f_a, Gamma_phi, H_inf, show_all=True, save=False, add_title=True, **kwargs):
    sol = t, rho_phi, rho_R, rho_tot, T, H, R, theta, theta_dot, n_L = simulate(m_a, f_a, Gamma_phi, H_inf, **kwargs)

    rho_phi_0 = calc_energy_density_from_hubble(H_inf)
    time_label = r"Time, $t \cdot \mathrm{GeV}$"
    T_RH = 2e13 * np.sqrt(Gamma_phi / 1e9) # paper
    T_max = 5e13 * (Gamma_phi / 1e9)**(1/4) * (H_inf / 1e11)**(1/2) # paper
    T_RH_KT = 0.55 * g_star**(-1/4) * (m_pl * Gamma_phi)**0.5 # K&T
    T_max_KT = 0.8 * g_star**(-1/4) * rho_phi_0**(1/8) * (Gamma_phi * m_pl)**(1/4) # K&T
    t_RH = 1 / Gamma_phi
    rho_R_max_Weinberg = 0.139 * Gamma_phi / H_inf * rho_phi_0
    T_max_Weinberg = calc_temperature(rho_R_max_Weinberg)
    osc = True
    try:
        t_osc = t[np.where(m_a > H)[0][0]]
    except:
        osc = False

    if show_all:
        # energy densities
        plt.figure()
        plt.loglog(t, rho_phi, color="tab:blue",   label=r"Numerical: $\rho_\phi$")
        plt.loglog(t, rho_R,   color="tab:orange", label=r"Numerical: $\rho_R$")
        #plt.loglog(t, rho_tot, label=r"Numerical: $\rho_\mathrm{tot}$")
        plt.loglog(t, calc_rho_phi_analytical(calc_start_time(H_inf), t, R_osc, R, rho_phi_0, Gamma_phi),
                   ls="--", color="tab:blue", label=r"Analytical: $\rho_\phi$")
        plt.loglog(t, calc_rho_R_analytical(rho_R, calc_start_time(H_inf), t, R_osc, R, rho_phi_0, Gamma_phi),
                   ls="--", color="tab:orange", label=r"Analytical: $\rho_R$")
        plt.axvline(t_RH, label=r"Reheating time, $t_\mathrm{RH}$", color="black", ls="--")
        #plt.axhline(rho_R_max_Weinberg, color="green", ls="-", label="Weinberg max rho_R")
        #plt.axhline(calc_radiation_energy_density(T_RH_KT), color="green", ls="--", label="K&T max rho")
        plt.legend()
        plt.xlabel(time_label)
        plt.ylabel(r"Energy Density, $\rho / \mathrm{GeV}^4$")
        plt.ylim(1e-5, plt.ylim()[1])
        plt.show()

        # temperature
        plt.figure()
        plt.loglog(t, T, label="Temperature", color="tab:blue") # from simulation
        # analytic results for different epochs
        try:
            i_RH = np.where(t_RH <= t)[0][0]
            max_T_idx = np.argmax(T)
            plt.loglog(t[:i_RH], T[max_T_idx]*(R[:i_RH] / R[max_T_idx])**(-3/8), color="tab:blue", ls="--", label="Reheating")
            plt.loglog(t[i_RH:], T[-1] * (R[i_RH:]/R[-1])**-1, color="tab:blue", ls=":", label="Radiation Domination")
        except:
            pass
        # reheating and maximal temperature
        plt.axvline(t_RH, label="Reheating Time", ls="--", color="black")
        #plt.axhline(T_RH, label="Reheating (paper)", ls=":", color="grey") # ~ same as K&T
        plt.axhline(T_RH_KT, label="Reheating T (K&T)", ls="--", color="black")
        #plt.axhline(T_max, label="Max. T (paper)", ls=":", color="red")
        plt.axhline(T_max_KT, label="Max. T (K&T)", ls="--", color="red")
        #plt.axhline(T_max_Weinberg, label="Max. T (Weinberg)", ls="-.", color="red")
        # labels
        plt.legend() # ncol=2)
        plt.xlabel(time_label)
        plt.ylabel(r"Temperature, $T / \mathrm{GeV}$")
        #plt.ylim(plt.ylim()[0], np.max(T) * 2)
        plt.show()

        # scale factor
        plt.figure()
        plt.subplot(1,2,1)
        plt.loglog(t, R / R_osc)
        plt.xlabel(time_label)
        plt.ylabel(r"Scale Parameter, $R / R_\mathrm{osc}$")
        # Hubble parameter
        plt.subplot(1,2,2)
        plt.loglog(t, H, label="Numerical")
        plt.loglog(t, 1 / (2*t), label="Radiation $H = 1/2t$")
        plt.xlabel(time_label)
        plt.ylabel(r"Hubble parameter, $H / \mathrm{GeV}$")
        plt.legend()
        plt.tight_layout()
        plt.show()

        # axion field
        plt.figure()
        plt.semilogx(t, theta, label=r"Axion field $a(t) / f_a = \theta(t)$")
        plt.xlabel(time_label)
        plt.ylabel(r"Axion angle, $\theta = a(t) / f_a$")
        if osc: plt.axvline(t_osc, label=r"Oscillation Onset: $H_\mathrm{osc} = m_a$", color="black", ls="--")
        plt.axhline(0, label=r"$\theta = 0$", color="black", ls=":")
        plt.legend()
        plt.show()

    # asymmetry parameter
    plt.figure()
    # the minus sign comes from the fact that we are actually computing B - L
    eta_B = n_L_to_eta_B_final(T[1:], n_L[1:])
    plt.loglog(t[1:], eta_B, label="Boltzmann")
    plt.axhline(eta_B[-1], color="black", ls=":", label="Final Value")
    if osc: plt.axvline(t_osc, color="black", ls="--", label="Oscillation Onset")
    plt.xlabel(time_label)
    plt.ylabel(r"Projected Baryon Asymmetry, $\eta_B$")
    plt.legend() # loc="lower right")
    if add_title:
        plt.title(f"$m_a =$ {m_a:.1e} GeV, $\\Gamma_\\phi =$ {Gamma_phi:.1e} GeV,\n$f_a =$ {f_a:.1e} GeV, $H_\\mathrm{{inf}} =$ {H_inf:.1e} GeV")
    plt.tight_layout()
    if save:
        plt.savefig(util.make_plot_path(f"lepto_axion_osc_m_a={m_a:.1e}Gamma_phi={Gamma_phi:.1e}f_a={f_a:.1e}H_inf={H_inf:.1e}_plot.pdf"))
    plt.show()

    print("axion oscillations:", count_oscillations(theta))
    print("final asymmetry:", eta_B[-1])
    return sol

def make_decay_plots(m_a, f_a, Gamma_phi, H_inf, do_plot=True, **kwargs):
    bg_sol = simulate(m_a, f_a, Gamma_phi, H_inf)
    decay_sol = t, rho_R, rho_a, R, T, n_L = simulate_axion_decay(m_a, f_a, bg_sol, end=np.log(1e20),
                                                                 **kwargs)
    eta_B_final_proj = n_L_to_eta_B_final(T, n_L)
    t_axion_decay = 1 / calc_Gamma_a(m_a, f_a)

    if do_plot:
        plt.figure()
        plt.loglog(t, rho_a, label=r"Axion $\rho_a$")
        plt.loglog(t, rho_R, label=r"Radiation $\rho_R$")
        plt.axvline(t_axion_decay, color="black", ls="--", label="Axion Decay Time")
        plt.ylim(1e-20, rho_R[0]*10)
        time_label = r"Time, $t [\mathrm{GeV}^{-1}]$"
        plt.xlabel(time_label)
        plt.ylabel(r"Energy Density, $\rho [\mathrm{GeV}^4]$")
        plt.legend()
        plt.show()

        plt.figure()
        T = calc_temperature(rho_R)
        plt.loglog(t, T, label="Temperature")
        plt.axvline(t_axion_decay, color="black", ls="--", label="Axion Decay Time")
        plt.xlabel(time_label)
        plt.ylabel(r"Temperature, $T$ / GeV")
        plt.legend()
        plt.show()

        plt.figure()
        plt.loglog(t, eta_B_final_proj)
        plt.xlabel(time_label)
        plt.ylabel(r"$\eta_B^\mathrm{final}$ projection")
        plt.show()

    return bg_sol, decay_sol

def sample_parameter_space(func, f_a, H_inf, min_Gamma_phi=1e6, max_Gamma_phi=1e10, min_m_a=2e5, max_m_a=1e10,
                         num_m_a_samples=30, num_Gamma_phi_samples=31):
    Gamma_phi_s = np.geomspace(min_Gamma_phi, max_Gamma_phi, num_Gamma_phi_samples)
    m_a_s = np.geomspace(min_m_a, max_m_a, num_m_a_samples)
    eta_B_s = np.array([[func(m_a, f_a, Gamma_phi, H_inf) for m_a in m_a_s] for Gamma_phi in tqdm(Gamma_phi_s)])
    return m_a_s, Gamma_phi_s, eta_B_s

def sample_parameter_space_numerical(f_a, H_inf, **kwargs):
    return sample_parameter_space(compute_B_asymmetry, f_a, H_inf, **kwargs)

def compute_correct_curve(f_a, H_inf, min_val=5e5):
    m_a_bounds = np.log10((min_val, H_inf))
    Gamma_phi_bounds = np.log10((min_val, H_inf))
    analytic_goal_fn = lambda p: np.log10(compute_B_asymmetry_analytic(10**p[0], f_a, 10**p[1])) - np.log10(eta_B_observed)
    goal_fn = lambda p: np.log10(compute_B_asymmetry(10**p[0], f_a, 10**p[1], H_inf)) - np.log10(eta_B_observed)
    clueless_guess = (np.mean(m_a_bounds), np.mean(Gamma_phi_bounds))
    analytic = implicit_curve.find_root(analytic_goal_fn, clueless_guess, m_a_bounds, Gamma_phi_bounds)
    lg_m_a, lg_Gamma_phi = implicit_curve.find_implicit_curve(goal_fn, m_a_bounds, Gamma_phi_bounds, analytic,
                                               h=0.01, eps=0.01, step_length=0.1)
    m_a_curve = 10**lg_m_a
    Gamma_phi_curve = 10**lg_Gamma_phi
    return m_a_curve, Gamma_phi_curve

def find_minimal_m_a(H_inf_max_dist=10, current_f_a=1e13):
    current_min_m_a = np.inf
    current_min_Gamma_phi = np.inf
    more_curves = []
    curve_eps = 0.1
    step = 0
    while True:
        step += 1
        print("step:", step)
        H_inf = calc_H_inf_max(current_f_a) / H_inf_max_dist
        m_a_curve, Gamma_phi_curve = compute_correct_curve(current_f_a, H_inf)
        min_m_a = np.min(m_a_curve)
        if min_m_a < current_min_m_a:
            min_Gamma_phi = np.min(Gamma_phi_curve)
            assert min_Gamma_phi < current_min_Gamma_phi
            delta = (current_min_m_a - min_m_a) / min_m_a
            print("delta:", delta)
            current_min_m_a = min_m_a
            current_min_Gamma_phi = min_Gamma_phi
            if delta <= curve_eps:
                break
        current_f_a /= 5


