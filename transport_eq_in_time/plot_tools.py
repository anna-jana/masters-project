import numpy as np, matplotlib.pyplot as plt, matplotlib.colors as mcolors, matplotlib as mpl
import importlib, itertools, os
import decay_process, transport_equation
decay_process = importlib.reload(decay_process)
transport_equation = importlib.reload(transport_equation)

plotpath = "plots"
if not os.path.exists(plotpath):
    os.mkdir(plotpath)
    
mpl.rcParams["font.size"] = 15

def latex_exponential_notation(value, digits=1):
    exponent = int(np.floor(np.log10(np.abs(value))))
    prefix = value / 10**exponent
    rounded_prefix = np.round(np.abs(prefix) * 10**digits) / 10.0**digits
    format_string_prefix = r"%." + str(digits) + "f"
    rounded_prefix_string = format_string_prefix % rounded_prefix
    while rounded_prefix_string and rounded_prefix_string[-1] == "0":
        rounded_prefix_string = rounded_prefix_string[:-1]
    if rounded_prefix_string and rounded_prefix_string[-1] == ".":
        rounded_prefix_string = rounded_prefix_string[:-1]
        if rounded_prefix_string and rounded_prefix_string[-1] == "1":
            rounded_prefix_string = ""
    if rounded_prefix_string:
        latex_string = rounded_prefix_string + r"\cdot 10^{%i}" % exponent
    else:
        latex_string = "10^{%i}" % exponent
    if value < 0:
        latex_string = "-" + latex_string
    return latex_string

def plot_background_cosmology(conv_factor, Gamma_inf, background_sols, axion_sols, show_steps=True):
    color = None if show_steps else "tab:blue"
    fig = plt.figure()
    fig.subplots_adjust(hspace=0)
    plt.subplot(2,1,1)
    tend = 0
    for i, (axion_sol, T_and_H_and_T_dot_fn) in enumerate(zip(axion_sols, background_sols)):
        t_inf_max = conv_factor * axion_sol.t[-1]
        ts_inf = np.geomspace(decay_process.t0, decay_process.t0 + t_inf_max, 500)
        T, H, T_dot = T_and_H_and_T_dot_fn(ts_inf)
        plt.loglog(tend + ts_inf, H / Gamma_inf, label="numerical solution for reheating" if i == 0 else None, color=color)
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
        plt.loglog(tend + ts_inf, T, color=color)
        tend += conv_factor * axion_sol.t[-1]
    plt.xlabel(r"$t \cdot \Gamma_\mathrm{inf}$")
    plt.ylabel(r"T / GeV")
    
def get_samples(axsol, tend):
    if hasattr(axsol, "sol"):
        ts = np.linspace(0.0, axsol.t[-1], 500)
        ys = axsol.sol(ts)
    else:
        ts, ys = axsol
    return tend + ts, ys, tend + ts[-1]

def plot_axion_field_evolution(axion_model, axion_parameter, f_a, axion_sols, 
                               show_steps=True, field_name="\\varphi", show_energy=True, show_derivative=True, logtime=True):
    color = None if show_steps else "tab:blue"
    fig = plt.figure()
    n = 3 if show_energy else 2
    n = 2 if show_derivative else 1

    plt.subplot(n,1,1)
    fig.subplots_adjust(hspace=0)
    
    zero_lw = 0.5

    tend = 0
    for axion_sol in axion_sols:
        ts, ys, tend = get_samples(axion_sol, tend)
        N = ys.shape[0] // 2
        for i, (c, ax) in enumerate(zip(mcolors.TABLEAU_COLORS, ys[:N, :])):
            plt.plot(ts, ax, color=None if show_steps else c, label=f"axion {i + 1}" if not show_steps else None)
    if tend > 1.0:
        plt.axvline(1.0, color="black", ls="--")
    plt.axhline(0.0, color="black", lw=zero_lw, ls=":")
    plt.xscale("log" if logtime else "linear")
    plt.ylabel(f"${field_name} / f_a$")
    plt.legend()
    if not show_derivative:
        plt.xlabel(r"$t \cdot m_a(T_\mathrm{osc})$")
    
    if show_derivative:
        plt.subplot(n,1,2)
        tend = 0
        for axion_sol in axion_sols:
            ts, ys, tend = get_samples(axion_sol, tend)
            N = ys.shape[0] // 2
            for c, ax in zip(mcolors.TABLEAU_COLORS, ys[N:, :]):
                plt.plot(ts, ax, color=None if show_steps else c)
        if tend > 1.0:
            plt.axvline(1.0, color="black", ls="--")
        plt.axhline(0.0, color="black", lw=zero_lw, ls=":")
        plt.xscale("log" if logtime else "linear")
        plt.ylabel(r"$\dot{" + field_name + r"} / f_a / m_a(T_\mathrm{osc})$")
        if not show_energy:
            plt.xlabel(r"$t \cdot m_a(T_\mathrm{osc})$")
    
    if show_energy:
        plt.subplot(n,1,3)
        tend = 0
        for axion_sol in axion_sols:
            ts, ys, tend = get_samples(axion_sol, tend)
            plt.loglog(ts, [axion_model.get_energy(y, f_a, *axion_parameter) for y in ys.T], color=color)
        if tend > 1.0:
            plt.axvline(1.0, color="black", ls="--")
        plt.xlabel(r"$t \cdot m_a(T_\mathrm{osc})$")
        plt.ylabel(r"$\rho / f_a^2 / \mathrm{GeV}^2$")

def plot_charge_evolution(conv_factor, axion_sols, red_chem_pot_sols, show_steps=True, fig=None, ax=None, show_legend=True):
    color = None if show_steps else "tab:blue"
    if fig is None:
        fig, ax = plt.subplots()
    tend = 0
    for j, (axion_sol, red_chem_pot_sol, ls) in enumerate(zip(axion_sols, red_chem_pot_sols, itertools.cycle(("-", ":")))):
        t_inf_max = conv_factor * axion_sol.t[-1]
        ts_inf = np.geomspace(decay_process.t0, decay_process.t0 + t_inf_max, 100)
        red_chem_pots = red_chem_pot_sol(np.log(ts_inf))
        for i, (name, color) in enumerate(zip(transport_equation.charge_names, mcolors.TABLEAU_COLORS)):
            ax.plot(ts_inf + tend, np.abs(red_chem_pots[i, :]), ls=ls if show_steps else "-", color=color, label=name if j == 0 else None)
        B_minus_L = np.abs(transport_equation.calc_B_minus_L(red_chem_pots))
        ax.plot(ts_inf + tend, B_minus_L,
                label="B - L" if j == 0 else None, color="black", lw=2, ls=ls if show_steps else "-")
        tend += conv_factor * axion_sol.t[-1]
    ax.set_xscale("log")
    ax.set_yscale("log")
    #plt.ylim(1e-29, 1e-26)
    ax.set_xlabel(r"$t \cdot \Gamma_\mathrm{inf}$")
    ax.set_ylabel(r"$|\mu_i / T|$")
    if show_legend:
        ax.legend(ncol=3, framealpha=1)

def plot_asymmetry_time_evolution(axion_model, conv_factor, Gamma_inf, axion_parameter, f_a, background_sols, axion_sols, red_chem_pot_sols, show_steps=True):
    plot_background_cosmology(conv_factor, Gamma_inf, background_sols, axion_sols, show_steps=True)
    plot_axion_field_evolution(axion_model, axion_parameter, f_a, axion_sols, show_steps=True)
    plot_charge_evolution(conv_factor, axion_sols, red_chem_pot_sols, show_steps=True)
        
def plot_dilution_factor_time_evolution(sol_axion_decay, T_and_H_fn_axion):
    axion_decay_time = sol_axion_decay.t[-1] - sol_axion_decay.t[0]
    plt.figure()
    ts = np.geomspace(decay_process.t0, decay_process.t0 + axion_decay_time, 100) # ts is in axion decay units
    fs = decay_process.find_dilution_factor(sol_axion_decay, T_and_H_fn_axion, ts)
    plt.axhline(fs[0], ls="--", color="black", label="initial")
    plt.axhline(fs[-1], ls="-", color="black", label="final")
    plt.semilogx(ts, fs, label="evolution")
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

def plot_config_space_2d(ts, var1, var2, name1, name2, name, V_fn, parameter,
                        skip_percent=0.0, interval=np.pi / 8):
    skip = (ts[-1] - ts[0]) * skip_percent
    s = skip_steps = int(np.ceil(skip / (ts[1] - ts[0])))

    def calc_range(x):
        return np.linspace(np.floor(np.min(x) / interval) * interval, np.ceil(np.max(x) / interval) * interval, 100)
    
    plt.figure(figsize=(6,5))
    range1, range2 = calc_range(var1[skip_steps:]), calc_range(var2[skip_steps:])
    V = np.array([[V_fn([t1, t2], *parameter) for t1 in range1] for t2 in range2])
    plt.contourf(range1, range2, V, levels=15) # , cmap="OrRd")
    plt.plot(var1[skip_steps:], var2[skip_steps:], color="red")
    plt.plot([var1[skip_steps]], [var2[skip_steps]], "bo")
    plt.xlabel(name1)
    plt.ylabel(name2)
    plt.colorbar(label=f"$V({name1[1:-1]}, {name2[1:-1]})$")
    plt.title(name)