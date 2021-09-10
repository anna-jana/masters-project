import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root
import model
import transport_equation
import clock_work_axion
from common.rh_neutrino import calc_Gamma_a_SU2
from common.util import latex_exponential_notation, make_plot_path
from common import util, constants


def plot(ress, filename=None, m_a=None, H_inf=None, Gamma_phi=None, digits=1, title=None, fs=15, source_scale=1e-6, asymmetry_limits=(1e-15, 1e-7)):
    # create figure and subplots
    fig, (ax2, ax1) = plt.subplots(2, 1, figsize=(10,10),
                                   gridspec_kw=dict(height_ratios=[1, 2]))
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

def find_obs(m_phi_range, mR_range, obs, sim_f_eff, actuall_f_eff, H_inf):
    A = np.log10(obs[:, :, 0] / constants.eta_B_observed)
    sim_f = np.array([clock_work_axion.calc_f(sim_f_eff, clock_work_axion.calc_eps(mR)) for mR in mR_range])[:, None]
    actuall_f = np.array([clock_work_axion.calc_f(actuall_f_eff, clock_work_axion.calc_eps(mR)) for mR in mR_range])[:, None]
    B = np.log10(obs[:, :, 1] / sim_f**2 * actuall_f**2 / constants.Omega_DM_h_sq)
    return A, B

def plot_clockwork_parameter_space(m_phi_range, mR_range, obs, sim_f_eff, actuall_f_eff, H_inf, do_log=True):
    plt.figure(figsize=(8,4))
    num_lines = 10
    fts = 15
    inline_label_fs = 10

    A, B = find_obs(m_phi_range, mR_range, obs, sim_f_eff, actuall_f_eff, H_inf)

    ############################ baryon asymmetry #########################
    level = np.sort(np.concatenate([[0], np.linspace(np.nanmin(A), np.nanmax(A), num_lines)]))
    if do_log:
        C1 = plt.contour(m_phi_range * 1e9, mR_range, A, levels=level, cmap="viridis")
    else:
        C1 = plt.contour(np.log10(m_phi_range * 1e9), mR_range, A, levels=level, cmap="viridis")

    cbar1 = plt.colorbar(pad=0.08)
    cbar1.set_label(r"$\log_{10} ( \eta_B / \eta_B^{\mathrm{obs}})$", fontsize=fts)
    C1.collections[np.where(C1.levels == 0)[0][0]].set_color("red")
    C1.collections[np.where(C1.levels == 0)[0][0]].set_linewidths(3)
    #plt.gca().clabel(C1, inline=mφ [eV]True, fontsize=inline_label_fs)

    ################################ relic density #############################
    if do_log:
        level = np.sort(np.concatenate([[0], np.linspace(np.nanmin(B), np.nanmax(B), num_lines)]))
    else:
        level = np.unique(np.concatenate([[0], np.arange(np.ceil(np.nanmin(B)), np.floor(np.nanmax(B)), 1)]))

    if do_log:
        C2 = plt.contour(m_phi_range * 1e9, mR_range, B, levels=level, cmap="plasma", linestyles="--")
    else:
        C2 = plt.contour(np.log10(m_phi_range * 1e9), mR_range, B, levels=level, cmap="plasma", linestyles="--")

    cbar2 = plt.colorbar()
    cbar2.set_label(r"$\log_{10} ( \Omega_a / \Omega_a^{\mathrm{obs}})$", fontsize=fts)
    C2.collections[np.where(C2.levels == 0)[0][0]].set_linewidths(3)
    C2.collections[np.where(C2.levels == 0)[0][0]].set_color("red")
    #plt.gca().clabel(C2, inline=True, fontsize=inline_label_fs)

    # constrains
    if do_log:
        # interference with inflation
        plt.fill_between(m_phi_range * 1e9, [clock_work_axion.get_max_mR(m_phi, H_inf, 1.0) for m_phi in m_phi_range], [mR_range[-1]]*len(m_phi_range),
                         color="blue", alpha=0.5)
        # decay
        plt.fill_between(m_phi_range * 1e9, [clock_work_axion.get_min_mR(m_phi, actuall_f_eff) for m_phi in m_phi_range], [mR_range[0]]*len(m_phi_range),
                         color="green", alpha=0.5)
    else:
        plt.fill_between(np.log10(m_phi_range * 1e9), [clock_work_axion.get_max_mR(m_phi, H_inf, 1.0) for m_phi in m_phi_range],
                         [mR_range[-1]]*len(m_phi_range), color="blue", alpha=0.5)
        # decay
        plt.fill_between(np.log10(m_phi_range * 1e9), [clock_work_axion.get_min_mR(m_phi, actuall_f_eff) for m_phi in m_phi_range],
                         [mR_range[0]]*len(m_phi_range), color="green", alpha=0.5)

    if do_log:
        plt.xlim(m_phi_range[0] * 1e9, m_phi_range[-1] * 1e9)
    else:
        plt.xlim(np.log10(m_phi_range[0] * 1e9), np.log10(m_phi_range[-1] * 1e9))

    plt.ylim(mR_range[0], mR_range[-1])
    if do_log:
        plt.xlabel(r"$m_\phi$ / eV", fontsize=fts)
    else:
        plt.xlabel(r"$\log_{10}(m_\phi$ / eV)", fontsize=fts)

    plt.ylabel("mR", fontsize=fts)
    plt.xticks(fontsize=fts)
    plt.yticks(fontsize=fts)
    if do_log:
        plt.xscale("log")
    plt.title(f"$f_\\mathrm{{eff}} = {util.latex_exponential_notation(actuall_f_eff, 2)}$ GeV", fontsize=fts)
    plt.tight_layout()


