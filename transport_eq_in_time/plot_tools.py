import numpy as np, matplotlib.pyplot as plt, matplotlib.colors as mcolors
import importlib, itertools
import decay_process, transport_equation
decay_process = importlib.reload(decay_process)
transport_equation = importlib.reload(transport_equation)

def plot_asymmetry_time_evolution(axion_model, conv_factor, Gamma_inf, axion_parameter, f_a, background_sols, axion_sols, red_chem_pot_sols):
    print(axion_model, axion_parameter)
    # background cosmology
    plt.figure()
    plt.subplot(2,1,1)
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
    plt.ylabel(r"$H / \Gamma_\mathrm{inf}$")
    plt.subplot(2,1,2)
    tend = 0
    for i, (axion_sol, T_and_H_and_T_dot_fn) in enumerate(zip(axion_sols, background_sols)):
        t_inf_max = conv_factor * axion_sol.t[-1]
        ts_inf = np.geomspace(decay_process.t0, decay_process.t0 + t_inf_max, 500)
        T, H, T_dot = T_and_H_and_T_dot_fn(ts_inf)
        plt.loglog(tend + ts_inf, T)
        tend += conv_factor * axion_sol.t[-1]
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
    plt.subplot(2,1,1)
    tend = 0
    for axion_sol in axion_sols:
        ts_ax = np.linspace(0.0, axion_sol.t[-1], 500)
        ts = tend + ts_ax
        tend += axion_sol.t[-1]
        plt.loglog(ts, [axion_model.get_energy(y, f_a, *axion_parameter) for y in axion_sol.sol(ts_ax).T])
    plt.xlabel(r"$t \cdot m_a(T_\mathrm{osc})$")
    plt.ylabel(r"~ energy density")
    plt.subplot(2,1,2)
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
        
def plot_dilution_factor_time_evolution(sol_axion_decay, T_and_H_fn_axion):
    axion_decay_time = sol_axion_decay.t[-1] - sol_axion_decay.t[0]
    plt.figure()
    ts = np.geomspace(decay_process.t0, decay_process.t0 + axion_decay_time, 100) # ts is in axion decay units
    fs = decay_process.find_dilution_factor(sol_axion_decay, T_and_H_fn_axion, ts)
    plt.axhline(fs[0], ls="--", color="black", label="initial")
    plt.axhline(fs[-1], ls="-", color="black", label="final")
    plt.loglog(ts, fs, label="evolution")
    plt.xlabel(r"$\Gamma_a \cdot t$")
    plt.ylabel(r"dilution factor $f = (T(t_0) a(t_0) / T(t) a(t))^3$")
    plt.legend()

def plot_relic_density_time_evolution(conv_factor, t_advance_inf, advance_sol_axion, Y_samples):
    # advance to oscillation start
    if advance_sol_axion is not None:
        plt.figure()
        ts_inf = np.geomspace(decay_process.t0, decay_process.t0 + t_advance_inf, 400)
        ts_ax = (ts_inf - decay_process.t0) / conv_factor
        plt.semilogx(ts_ax, advance_sol_axion.sol(ts_ax)[0, :])
        plt.xlabel("t * M")
        plt.ylabel("axion field")
        plt.title("advancing to oscillation")

    # convergence of abundance Y = n/s
    tend = 0
    plt.figure()
    for (ts_ax, Y, is_min, is_max, Y_estimate) in Y_samples:
        ls = plt.plot(ts_ax + tend, Y)
        plt.plot(ts_ax[1:-1][is_min] + tend, Y[1:-1][is_min], "ob")
        plt.plot(ts_ax[1:-1][is_max] + tend, Y[1:-1][is_max], "or")
        plt.plot([ts_ax[0] + tend, ts_ax[-1] + tend], [Y_estimate, Y_estimate], color=ls[0].get_color())
        tend += ts_ax[-1]
    plt.xlabel("t*M")
    plt.ylabel("n/s")

    