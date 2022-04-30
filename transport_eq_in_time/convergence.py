import time, importlib, itertools
import numpy as np, matplotlib.pyplot as plt, matplotlib.colors as mcolors
import decay_process, axion_motion, transport_equation
from scipy.optimize import root
decay_process = importlib.reload(decay_process)
axion_motion = importlib.reload(axion_motion)
transport_equation = importlib.reload(transport_equation)

zeta3 = 1.20206
g_photon = 2
C_sph = 8 / 23
eta_B_observed = 6e-10 # from paper
g_star_0 = 43/11 # from paper
asym_const = - g_star_0 * C_sph * np.pi**2 / (6 * decay_process.g_star * zeta3 * g_photon)

def red_chem_pot_to_asymmetry(red_chem_pot_B_minus_L):
    #n_B_minus_L = T**3 / 6 * red_chem_pot_B_minus_L
    #n_B = - C_sph * n_B_minus_L
    #n_B_today = g_star_0 / g_star * n_B
    #n_gamma = zeta3 / np.pi**2 * g_photon * T**3 # K&T (3.52)
    #eta_B = n_B_today / n_gamma
    #eta_B = g_star_0 / g_star * (- C_sph) * T**3 / 6 * red_chem_pot_B_minus_L / (zeta3 / np.pi**2 * g_photon * T**3)
    #= g_star_0 / g_star * (- C_sph) * 1 / 6 * red_chem_pot_B_minus_L / (zeta3 / np.pi**2 * g_photon) * red_chem_pot_B_minus_L
    # = - g_star_0 * C_sph * np.pi**2 / (6 * g_star * zeta3 * g_photon) * red_chem_pot_B_minus_L
    return asym_const * red_chem_pot_B_minus_L

def compute_asymmetry(H_inf, Gamma_inf, axion_parameter, f_a,
        axion_model=axion_motion.realignment_axion_field, axion_init=(1.0, 0.0),
        start_tmax_axion_time=10.0, step_tmax_axion_time=2*2*np.pi,
        source_vector_axion=transport_equation.source_vector_weak_sphaleron,
        axion_decay_time=10.0, debug=False, convergence_rtol=1e-3, nsamples=100, calc_init_time=False):
    # this is a bit useless but I keep it to make it work like the general case
    energy_scale = axion_model.find_dynamical_scale(*axion_parameter)
    if energy_scale > H_inf:
        return np.nan # oscillation starts before the end of inflation
        #(invalidates assumtion that the axion dosnt iterfere with inflation)

    conv_factor = Gamma_inf / energy_scale
    rho_R_init = 0.0
    rho_inf_init = 3 * decay_process.M_pl**2 * H_inf**2
    red_chem_pots_init = np.zeros(transport_equation.N)
    tmax_axion_time = start_tmax_axion_time # initial time to integrate
    step = 1
    scale = decay_process.find_scale(Gamma_inf)

    axion_sols = []
    red_chem_pot_sols = []
    background_sols = []

    while True:
        rh_start = time.time()
        tmax_inf_time = tmax_axion_time * conv_factor
        if debug: print("step =", step)
        # reheating process
        sol_rh, T_and_H_fn, T_and_H_and_T_dot_fn = decay_process.solve(tmax_inf_time, rho_R_init, rho_inf_init, scale, Gamma_inf)
        if calc_init_time and step == 1:
            T_eq_general = 1e12 # [GeV]
            Tmax = (
                0.8 * decay_process.g_star**(-1/4) * rho_inf_init**(1/8)
                * (Gamma_inf * decay_process.M_pl / (8*np.pi))**(1/4)
            ) # [GeV]
            if T_and_H_fn(tmax_inf_time)[0] > T_eq_general:
                H_in_GeV = np.sqrt(np.pi**2 / 30 * decay_process.g_star / (3*decay_process.M_pl**2)) * T_eq_general**2
                H = H_in_GeV / Gamma_inf
                t_eq = (1/H - 1/(H_inf / Gamma_inf)) / 2 + decay_process.t0 # [1/Gamma_inf]
            elif Tmax < T_eq_general:
                t_eq = np.nan
            else:
                goal_fn = lambda log_t: np.log(T_and_H_fn(np.exp(log_t))[0] / T_eq_general)
                sol = root(goal_fn, np.log(decay_process.t0 + tmax_inf_time  / 2), method="lm")
                t_eq = np.exp(sol.x[0] if sol.success else np.nan) # [1/Gamma_inf]
            t_axion = 1.0 + 2*np.pi*10 # integrate 10 axion oscillations [1/m_a]
            t_RH = 1.0 + decay_process.t0 # we want to reach reheating [1/Gamma_inf]

            tmax_inf_time = max(t_eq, t_RH)
            tmax_axion_time = tmax_inf_time / conv_factor
            sol_rh, T_and_H_fn, T_and_H_and_T_dot_fn = decay_process.solve(tmax_inf_time, 0.0, rho_inf_init, scale, Gamma_inf)
            if debug:
                print("calculcated initial integration time:")
                print("tmax_inf_time =", tmax_inf_time, "tmax_axion_time =", tmax_axion_time)
        if debug:
            background_sols.append(T_and_H_and_T_dot_fn)
            rh_end = time.time()
            print("rh:", rh_end - rh_start)

        # evolution of the axion field
        sol_axion = axion_model.solve(axion_init, axion_parameter, tmax_axion_time, T_and_H_fn, Gamma_inf)
        if debug:
            axion_sols.append(sol_axion)
            ax_end = time.time()
            print("axion:", ax_end - rh_end)
        axion_source = axion_model.get_source(sol_axion, conv_factor, *axion_parameter)

        # transport eq. for standard model charges
        sol_transp_eq = transport_equation.solve(tmax_inf_time, red_chem_pots_init, T_and_H_and_T_dot_fn,
                axion_source, source_vector_axion, Gamma_inf, conv_factor)
        if debug:
            red_chem_pot_sols.append(sol_transp_eq)
            trans_end = time.time()
            print("transport eq.:", trans_end - ax_end)

        # check for convergence of the B - L charge
        log_ts_inf = np.linspace(np.log(decay_process.t0), np.log(decay_process.t0 + tmax_inf_time), nsamples)
        red_chem_pots = sol_transp_eq(log_ts_inf) # returned function converts internal unit
        B_minus_L_red_chem = transport_equation.calc_B_minus_L(red_chem_pots)

        # convergence by timescale
        d_red_chem_B_minus_L_dt = (B_minus_L_red_chem[-1] - B_minus_L_red_chem[-2]) / (np.exp(log_ts_inf[-1]) - np.exp(log_ts_inf[-2]))
        time_scale_B_minus_L = np.abs(B_minus_L_red_chem[-1] / d_red_chem_B_minus_L_dt)
        time_scale_source = conv_factor
        rel_timescale = time_scale_B_minus_L / time_scale_source
        delta_ = 1 / rel_timescale
        if debug:
            print("change by timescale:", delta_, "vs", convergence_rtol)

        # convergence by change within the last integration interval
        a, b = np.max(B_minus_L_red_chem), np.min(B_minus_L_red_chem)
        delta = np.abs((a - b) / np.mean(B_minus_L_red_chem))
        if debug:
            print("B-L range:", b, a)
            print("delta =", delta, "convergence_rtol =", convergence_rtol)
        if not np.isfinite(delta):
            #print(sol_axion)
            red_chem_pots_final = red_chem_pots[:, -1]
            break
        if delta < convergence_rtol or step > 15:
            red_chem_pots_final = red_chem_pots[:, -1] # 1 (unit has been removed)
            break
        # set initial conditions to our end state and continue
        rho_R_init = decay_process.find_end_rad_energy(sol_rh, scale) # [GeV^4]
        rho_inf_init = decay_process.find_end_field_energy(sol_rh, rho_inf_init) # [GeV^4]
        axion_init = sol_axion.y[:, -1] # (1, energy_scale_axion (independent of initial condition and starting time))
        red_chem_pots_init = red_chem_pots[:, -1] # 1 (unit has been removed)
        tmax_axion_time = step_tmax_axion_time # continue to integrate with the step time

        step += 1

    if debug:
        # background cosmology
        plt.figure()
        tend = 0
        for i, (axion_sol, T_and_H_and_T_dot_fn) in enumerate(zip(axion_sols, background_sols)):
            t_inf_max = conv_factor * axion_sol.t[-1]
            ts_inf = np.geomspace(decay_process.t0, decay_process.t0 + t_inf_max, 500)
            T, H, T_dot = T_and_H_and_T_dot_fn(ts_inf)
            plt.loglog(tend + ts_inf, H / Gamma_inf, label="numerical solution for reheating" if i == 0 else None)
            tend += conv_factor * axion_sol.t[-1]
        H0 = background_sols[0](decay_process.t0)[1]
        ts_inf = np.geomspace(decay_process.t0, tend, 500)
        plt.loglog(ts_inf, (1.0 / (2*((ts_inf - decay_process.t0) / Gamma_inf) + 1/H0)) / Gamma_inf,
                label="analytical radiation domination", color="black", ls="--")
        plt.legend()
        plt.xlabel(r"$t \cdot \Gamma_\mathrm{inf}$")
        plt.ylabel(r"H / \Gamma_\mathrm{inf}$")

        plt.figure()
        tend = 0
        for i, (axion_sol, T_and_H_and_T_dot_fn) in enumerate(zip(axion_sols, background_sols)):
            t_inf_max = conv_factor * axion_sol.t[-1]
            ts_inf = np.geomspace(decay_process.t0, decay_process.t0 + t_inf_max, 500)
            T, H, T_dot = T_and_H_and_T_dot_fn(ts_inf)
            plt.loglog(tend + ts_inf, T)
            tend += conv_factor * axion_sol.t[-1]
        plt.legend()
        plt.xlabel(r"$t \cdot \Gamma_\mathrm{inf}$")
        plt.ylabel(r"T / GeV")

        # axion plot
        plt.figure()
        tend = 0
        for axion_sol in axion_sols:
            ts_ax = np.linspace(0.0, axion_sol.t[-1], 500)
            ts = tend + ts_ax
            tend += axion_sol.t[-1]
            plt.plot(ts, axion_sol.sol(ts_ax)[0, :])
        if tend > 1.0:
            plt.axvline(1.0, color="black", ls="--")
        plt.axhline(0.0, color="black", ls="-")
        plt.xscale("log")
        plt.xlabel(r"$t \cdot m_a(T_\mathrm{osc})$")
        plt.ylabel(r"$\varphi / f_a$")

        plt.figure()
        tend = 0
        for axion_sol in axion_sols:
            ts_ax = np.linspace(0.0, axion_sol.t[-1], 500)
            ts = tend + ts_ax
            tend += axion_sol.t[-1]
            plt.loglog(ts, [axion_model.get_energy(y, f_a, Gamma_inf, *axion_parameter) for y in axion_sol.sol(ts_ax).T])
        plt.xlabel(r"$t \cdot m_a(T_\mathrm{osc})$")
        plt.ylabel(r"~ energy density")

        plt.figure()
        tend = 0
        for axion_sol in axion_sols:
            ts_ax = np.linspace(0.0, axion_sol.t[-1], 500)
            ts = tend + ts_ax
            tend += axion_sol.t[-1]
            plt.plot(ts, axion_sol.sol(ts_ax)[1, :])
        if tend > 1.0:
            plt.axvline(1.0, color="black", ls="--")
        plt.axhline(0.0, color="black", ls="-")
        plt.xscale("log")
        plt.xlabel(r"$t \cdot m_a(T_\mathrm{osc})$")
        plt.ylabel(r"$\dot{\varphi} / f_a$")

        # transport eq. plot
        plt.figure()
        tend = 0
        for j, (axion_sol, red_chem_pot_sol, ls) in enumerate(zip(axion_sols, red_chem_pot_sols, itertools.cycle(("-", "--")))):
            t_inf_max = conv_factor * axion_sol.t[-1]
            ts_inf = np.linspace(decay_process.t0, decay_process.t0 + t_inf_max, 500)
            red_chem_pots = red_chem_pot_sol(np.log(ts_inf))
            for i, (name, color) in enumerate(zip(transport_equation.charge_names, mcolors.TABLEAU_COLORS)):
                plt.plot(ts_inf + tend, np.abs(red_chem_pots[i, :]), ls=ls, color=color, label=name if j == 0 else None)
            B_minus_L = np.abs(transport_equation.calc_B_minus_L(red_chem_pots))
            plt.plot(ts_inf + tend, B_minus_L,
                    label="B - L" if j == 0 else None, color="black", lw=2, ls=ls)
            tend += conv_factor * axion_sol.t[-1]
        plt.xscale("log")
        plt.yscale("log")
        #plt.ylim(1e-29, 1e-26)
        plt.xlabel(r"$t \cdot \Gamma_\mathrm{inf}$")
        plt.ylabel(r"$|\mu_i / T|$")
        plt.legend(ncol=3, framealpha=1)

    if axion_model.does_decay:
        if debug:
            start_decay = time.time()
        # dilution factor from axion decay
        # we don't do converence check for this part right now
        Gamma_axion = axion_model.get_decay_constant(f_a, *axion_parameter) # [GeV]
        axion_scale = decay_process.find_scale(Gamma_axion) # GeV

        rho_end_axion = axion_model.get_energy(sol_axion.y[:, -1], f_a, *axion_parameter) # [GeV^4]
        rho_end_rad = decay_process.find_end_rad_energy(sol_rh, scale) # [GeV^4]

        sol_axion_decay, T_and_H_fn_axion, _ = decay_process.solve(axion_decay_time, rho_end_rad, rho_end_axion, axion_scale, Gamma_axion)

        t = np.exp(sol_axion_decay.t[-1])
        f = decay_process.find_dilution_factor(sol_axion_decay, T_and_H_fn_axion, t)
        Omega = 0.0

        if debug:
            end_decay = time.time()
            print("axion decay took:", end_decay - start_decay, "seconds")
            plt.figure()
            ts = np.geomspace(decay_process.t0, decay_process.t0 + axion_decay_time, 100) # ts is in axion decay units
            fs = decay_process.find_dilution_factor(sol_axion_decay, T_and_H_fn_axion, ts)
            plt.axhline(fs[0], ls="--", color="black", label="initial")
            plt.axhline(fs[-1], ls="-", color="black", label="final")
            plt.loglog(ts, fs, label="evolution")
            plt.xlabel(r"$\Gamma_a \cdot t$")
            plt.ylabel(r"dilution factor $f = (T(t_0) a(t_0) / T(t) a(t))^3$")
            plt.legend()

    elif axion_model.has_relic_density:
        f = 1.0
        axion_init = axion_sol.y[:, -1]
        H0 = T_and_H_fn(t_end)
        H_osc = axion_model.find_H_osc(*axion_parameter) # [GeV]
        t_osc = 1 / 2 * (1/H_osc + 1/H0) + t0
        n_relic_oscs = 10
        period = axion_model.find_osc_period(*axion_parameter)
        tmax = t_osc + n_relic_oscs * period
        axion_model.solve()

    else:
        Omega = 1.0
        f = 1.0


    return red_chem_pot_to_asymmetry(transport_equation.calc_B_minus_L(red_chem_pots_final)), f, Omega

