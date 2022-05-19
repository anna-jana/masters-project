import numpy as np, matplotlib.pyplot as plt, matplotlib.colors as mcolors
import importlib, itertools
import decay_process, transport_equation
decay_process = importlib.reload(decay_process)
transport_equation = importlib.reload(transport_equation)

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

def plot_asymmetry_time_evolution(axion_model, conv_factor, Gamma_inf, axion_parameter, f_a, background_sols, axion_sols, red_chem_pot_sols):
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
    plt.ylabel(r"energy density / f_a^2")
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

def make_contour_plot(xrange, yrange, vals, nlines, padding, label, cmap, ls, fts):
    contour = plt.contour(xrange, yrange, vals, levels=nlines, cmap=cmap, linestyles=ls)
    cbar = plt.colorbar(pad=padding)
    cbar.set_label(label, fontsize=fts)

def double_contour_plot(xrange, yrange, A, B, Alabel, Blabel, num_lines_A=10, num_line_B=10, fts=15):
    make_contour_plot(xrange, yrange, A, num_lines_A, 0.08, Alabel, "viridis", "-", fts)
    make_contour_plot(xrange, yrange, B, num_line_B, None, Blabel, "plasma", "--", fts)
    

def plot_evolution():
    # create figure and subplots
    fig, (ax2, ax1) = plt.subplots(2, 1, figsize=(10,10), gridspec_kw=dict(height_ratios=[1, 2]))
    fig.subplots_adjust(hspace=0)

    # plot the evolution of the asymmetry for both codes
    for i, res in enumerate(ress):
        l1, = ax1.loglog(res.t, -transport_equation.calc_B_minus_L(res.red_chem_pots),
                   color="black", label="asymmetry" if i == len(ress) - 1 else None)

    if asymmetry_limits is not None:
        ax1.set_ylim(*asymmetry_limits)
    ax1.set_ylabel(r"$\mu_{B - L} / T$", fontsize=15)

    # plot the source from the axion field on a second axis on the right
    ax_sec = ax1.twinx()
    for i, res in enumerate(ress):
        T = res.T_fn(res.t)
        _, theta_dot = res.axion_fn(np.log(res.t))
        y = theta_dot / T / source_scale
        l3, = ax_sec.semilogx(res.t, y,
            label="source" if i == len(ress) - 1 else None, color="tab:blue")

    ax_sec.get_yaxis().get_major_formatter().set_useOffset(False)
    ax_sec.set_ylabel(r"$\dot{\theta} / T \cdot " + latex_exponential_notation(source_scale, 2) + "$", fontsize=fs)
    ax_sec.set_xscale("log")

    # legend for the main plot
    lines = [l1, l3]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, framealpha=1, fontsize=fs - 2) # loc="center left"

    # plot the temperature evolution in a second subfigure on the top
    for i, res in enumerate(ress):
        ax2.loglog(res.t, res.T_fn(res.t), color="tab:blue")

    ax1.set_xlabel(r"$t \cdot \mathrm{GeV}$", fontsize=15)
    ax2.set_ylabel(r"$T / \mathrm{GeV}$", fontsize=15)

    # added labels for the equilibration times
    ax2.xaxis.tick_top()
    T_max = max([np.max(res.T_fn(res.t)) for res in ress])
    T_min = min([np.min(res.T_fn(res.t)) for res in ress])
    t_eqs = []

    t_min_plot, t_max_plot = ax2.get_xlim()
    arrow_percent_dx = 0.03
    arrow_height = 0.3

    for alpha, T_eq in enumerate(model.T_eqs):
        if T_min > T_eq  or T_eq > T_max:
            continue
        for res in ress:
            try:
                res = root(lambda log_t: np.log(res.T_fn(np.exp(log_t)) / T_eq),
                           (np.log(res.t[0]) + np.log(res.t[-1])) / 2
                )
                if res.success:
                    t_eq = np.exp(res.x[0])
                    t_eqs.append(t_eq)
                    ax2.axvline(t_eq, color="black", ls="--")

                    s = np.sign(transport_equation.rate(2 * T_eq)[alpha] - transport_equation.rate(T_eq)[alpha])
                    percent = (np.log10(t_eq) - np.log10(t_min_plot)) / (np.log10(t_max_plot) - np.log10(t_min_plot))
                    ax2.annotate("", (percent + s * arrow_percent_dx, arrow_height), (percent, arrow_height),
                                 xycoords="axes fraction", arrowprops=dict(arrowstyle="->"))

                    break
            except Exception as e:
                print(e)
        else:
            t_eqs.append(None)

    good_names = [n for n, t_eq in zip(transport_equation.process_names, t_eqs) if t_eq is not None]
    good_t_eqs = [t_eq for t_eq in t_eqs if t_eq is not None]
    ax2.set_xticks(good_t_eqs)
    ax2.set_xticklabels(good_names, rotation=50, fontsize=fs)

    # add title with optionally given parameters
    if title is None:
        title_string = ""
        sep = r"\,\mathrm{GeV},\,"
        if m_a is not None:
            title_string += "m_a = " + latex_exponential_notation(m_a, digits) + sep
        if Gamma_phi is not None:
            title_string += r" \Gamma_\phi = " + latex_exponential_notation(Gamma_phi, digits) + sep
        if H_inf is not None:
            title_string += r" H_{\mathrm{inf}} = " + latex_exponential_notation(H_inf, digits) + sep
        title_string = title_string[:-3]
        if title_string:
            fig.suptitle(f"${title_string}$", fontsize=fs + 1, y=1.0002)
    else:
        plt.title(title)

    ax1.tick_params(labelsize=fs)
    ax2.tick_params(labelsize=fs)
    ax_sec.tick_params(labelsize=fs)

    plt.tight_layout()
    # output of the figure
    if filename is not None:
        plt.savefig(make_plot_path(filename))
    plt.show()