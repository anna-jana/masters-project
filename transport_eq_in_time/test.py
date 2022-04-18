import time, importlib, itertools
import numpy as np, matplotlib.pyplot as plt
import decay_process, axion_motion, transport_equation
from scipy.optimize import root
decay_process = importlib.reload(decay_process)
axion_motion = importlib.reload(axion_motion)
transport_equation = importlib.reload(transport_equation)

def test(H_inf, Gamma_inf, m_a, f_a, tmax_axion_time=10.0, axion_decay_time=10.0, recalc_time=False):
    axion_parameter = (m_a,)
    axion_model = axion_motion.single_axion_field
    source_vector = transport_equation.source_vector_weak_sphaleron
    # this is a bit useless but I keep to make it work like the general case
    energy_scale = axion_model.find_dynamical_scale(*axion_parameter)
    conv_factor = Gamma_inf / energy_scale
    tmax_inf_time = tmax_axion_time * conv_factor

    # initial conditions
    rho_R_init = 0.0
    rho_inf_init = 3*decay_process.M_pl**2*H_inf**2
    axion_init = (1.0, 0.0)
    red_chem_pots_init = np.zeros(transport_equation.N)

    # reheating process
    scale = decay_process.find_scale(Gamma_inf)
    sol_rh = decay_process.solve(tmax_inf_time, 0.0, rho_inf_init, scale, Gamma_inf, debug=not recalc_time)
    T_and_H_fn, T_and_H_and_T_dot_fn = decay_process.to_temperature_and_hubble_fns(sol_rh, rho_inf_init, scale, Gamma_inf, debug=not recalc_time)

    # calc required integration time
    if recalc_time:
        T_eq_general = 1e12
        Tmax = (
            0.8 * decay_process.g_star**(-1/4) * rho_inf_init**(1/8)
            * (Gamma_inf * decay_process.M_pl / (8*np.pi))**(1/4)
        )
        if T_and_H_fn(tmax_inf_time)[0] > T_eq_general:
            H = np.sqrt(np.pi**2 / 30 * decay_process.g_star / (3*decay_process.M_pl**2)) * T_eq_general**2
            H /= Gamma_inf
            t_eq = (1/H - 1/H_inf) / 2 + decay_process.t0
        elif Tmax < T_eq_general:
            t_eq = np.nan
        else:
            goal_fn = lambda log_t: np.log(T_and_H_fn(np.exp(log_t))[0] / T_eq_general)
            sol = root(goal_fn, np.log(decay_process.t0 + tmax_inf_time  / 2), method="lm")
            t_eq = np.exp(sol.x[0] if sol.success else np.nan)
        t_eq /= conv_factor # we need the time in inf units
        t_axion = 1.0 + 2*np.pi*10 # integrate 10 axion oscillations
        t_RH = 1.0 / conv_factor # we want to reach reheating
        tmax_axion_time = max(t_eq, t_axion, t_RH)
        tmax_inf_time = tmax_axion_time * conv_factor
        sol_rh = decay_process.solve(tmax_inf_time, 0.0, rho_inf_init, scale, Gamma_inf, debug=True)
        T_and_H_fn, T_and_H_and_T_dot_fn = decay_process.to_temperature_and_hubble_fns(sol_rh, rho_inf_init, scale, Gamma_inf, debug=True)

    # evolution of the axion field
    sol_axion = axion_model.solve(axion_init, axion_parameter, tmax_axion_time, T_and_H_fn, Gamma_inf, debug=True)
    axion_source = axion_model.get_source(sol_axion, conv_factor)

    # transport eq. for standard model charges
    sol_transp_eq = transport_equation.solve(tmax_inf_time, red_chem_pots_init, T_and_H_and_T_dot_fn,
            axion_source, source_vector, Gamma_inf, conv_factor, debug=True)

    # dilution factor from axion decay
    rho_end_axion = axion_model.get_energy(sol_axion.y[:, -1], energy_scale, f_a, Gamma_inf)
    rho_end_rad = decay_process.find_end_rad_energy(sol_rh, scale)
    Gamma_axion = axion_model.get_decay_constant(f_a, *axion_parameter)
    axion_decay_scale = decay_process.find_scale(Gamma_axion)
    sol_axion_decay = decay_process.solve(axion_decay_time, rho_end_rad, rho_end_axion, axion_decay_scale, Gamma_axion, debug=True)
    T_and_H_fn_axion, _ = decay_process.to_temperature_and_hubble_fns(sol_axion_decay, rho_end_axion, axion_decay_scale, Gamma_axion, debug=True)
    t = np.exp(np.linspace(sol_axion_decay.t[0], sol_axion_decay.t[-1], 400))
    f = decay_process.find_dilution_factor(sol_axion_decay, T_and_H_fn_axion, t)
    plt.figure()
    plt.axhline(f[0], color="black", ls="--", label="initial")
    plt.axhline(f[-1], color="black", ls="-", label="final")
    plt.loglog(t, f, label="evolution")
    plt.legend()
    plt.xlabel(r"$t \cdot \Gamma$")
    plt.ylabel(r"dilution factor $(T(t_0) a(t_0) / T(t) a(t))^3$")

    # debug plots for the different terms in the transport eq. for B - L
    ts = np.geomspace(decay_process.t0, decay_process.t0 + tmax_inf_time, 400)
    red_chem_pots = sol_transp_eq(np.log(ts))
    red_chem_B_minus_L = transport_equation.calc_B_minus_L(red_chem_pots)
    T, H, T_dot = T_and_H_and_T_dot_fn(ts)
    E1 = sum(transport_equation.charge_vector_B_minus_L[i] * red_chem_pots[i, :] for i in range(transport_equation.N))
    Rs = [transport_equation.calc_rate_vector(x) for x in T]
    E2 = np.array([sum(R[alpha] * transport_equation.charge_vector[alpha, j] * red_chem_pots[j, k] # TODO: write as matrix expression
                   for alpha in range(transport_equation.N_alpha) for j in range(transport_equation.N))
        for k, R in enumerate(Rs)])
    E = - E1 * E2
    S1 = axion_source(ts) / T
    S2 = np.array([sum(source_vector[alpha] * R[alpha] for alpha in range(transport_equation.N_alpha)) for R in Rs])
    S3 = [np.sum(transport_equation.charge_vector_B_minus_L)] * len(S2)
    S = S1 * S2 * S3
    D = - 3 * red_chem_B_minus_L * (T_dot / T + H)

    plt.figure()
    fn = np.abs
    plt.plot(ts, fn(E + S + D), label="$\mathrm{d} (\mu_{B - L} / T) / \mathrm{d} t$", color="black", lw=2)
    plt.plot(ts, fn(E1), label="$E_1$", ls="--", color="tab:blue")
    plt.plot(ts, fn(E2), label="$E_2$", ls=":", color="tab:blue")
    plt.plot(ts, fn(E ), label="$E$", ls="-", color="tab:blue")
    plt.plot(ts, fn(S1), label="$S_1$", ls="--", color="tab:green")
    plt.plot(ts, fn(S2), label="$S_2$", ls=":", color="tab:green")
    plt.plot(ts, fn(S3), label="$S_3$", ls="-.", color="tab:green")
    plt.plot(ts, fn(S ), label="$S$", ls="-", color="tab:green")
    plt.plot(ts, fn(D ), label="$D$", ls="-", color="tab:orange")
    plt.legend(ncol=3, framealpha=1) # , loc="upper center", bbox_to_anchor=(0.5, -0.2))
    plt.xscale("log")
    plt.yscale("log")
    plt.ylim(1e-15, plt.ylim()[1])
    plt.xlabel(r"$t \cdot \Gamma_\mathrm{inf}$")

    plt.show()
