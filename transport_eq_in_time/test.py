import time, importlib, itertools
import numpy as np, matplotlib.pyplot as plt, matplotlib.colors as mcolors
import decay_process, axion_motion, transport_equation
decay_process = importlib.reload(decay_process)
axion_motion = importlib.reload(axion_motion)
transport_equation = importlib.reload(transport_equation)

def test(H_inf, Gamma_inf, m_a, f_a, tmax_axion_time=10.0, axion_decay_time=10.0, debug=True):
    axion_parameter = (m_a,)
    axion_model = axion_motion.single_axion_field

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
    start_time = time.time()
    sol_rh = decay_process.solve_decay_eqs(tmax_inf_time, 0.0, rho_inf_init, Gamma_inf, debug=debug)
    T_and_H_fn, T_and_H_and_T_dot_fn = decay_process.to_temperature_and_hubble_fns(sol_rh, rho_inf_init, Gamma_inf, debug=debug)
    decay_time = time.time()
    print("decay done, took:", decay_time - start_time, "seconds")

    # evolution of the axion field
    sol_axion = axion_model.solve(axion_init, axion_parameter, tmax_axion_time, T_and_H_fn, Gamma_inf, debug=debug)
    axion_source = axion_model.get_source(sol_axion, conv_factor)
    axion_time = time.time()
    print("axion done, took:", axion_time - decay_time, "seconds")

    # transport eq. for standard model charges
    sol_transp_eq = transport_equation.solve(tmax_inf_time, red_chem_pots_init, T_and_H_and_T_dot_fn,
            axion_source, transport_equation.source_vector_weak_sphaleron, Gamma_inf, conv_factor, debug=debug)
    trans_time = time.time()
    print("trans done, took:", trans_time - axion_time, "seconds")

    # dilution factor from axion decay
    rho_end_axion = axion_model.get_energy(sol_axion.y[:, -1], energy_scale, f_a, Gamma_inf)
    rho_end_rad = decay_process.find_end_rad_energy(sol_rh, rho_inf_init)
    Gamma_axion = axion_model.get_decay_constant(f_a, *axion_parameter)
    sol_axion_decay = decay_process.solve_decay_eqs(axion_decay_time, rho_end_rad, rho_end_axion, Gamma_axion, debug=debug)
    T_and_H_fn_axion, _ = decay_process.to_temperature_and_hubble_fns(sol_axion_decay, rho_end_axion, Gamma_axion, debug=debug)
    T_fn = lambda t: T_and_H_fn_axion(t)[0]
    f = decay_process.find_dilution_factor(sol_axion_decay, T_fn, debug=debug)
    decay_time = time.time()
    print("axion decay done, took:", decay_time - trans_time, "seconds")

    if debug: plt.show()

def compute_asymmetry(H_inf, Gamma_inf, m_a, f_a,
        start_tmax_axion_time=10.0, step_tmax_axion_time=2*2*np.pi,
        start_axion_decay_time=10.0, step_tmax_axion_decay_time=1.0,
        debug=False, convergence_rtol=1e-3, nsamples=100):
    # TODO: these need to be arguments to this function
    axion_parameter = (m_a,)
    axion_model = axion_motion.single_axion_field
    axion_init = (1.0, 0.0)

    # this is a bit useless but I keep to make it work like the general case
    energy_scale = axion_model.find_dynamical_scale(*axion_parameter)
    conv_factor = Gamma_inf / energy_scale
    rho_R_init = 0.0
    rho_inf_init = 3 * decay_process.M_pl**2 * H_inf**2
    red_chem_pots_init = np.zeros(transport_equation.N)
    tmax_axion_time = start_tmax_axion_time # initial time to integrate
    step = 1

    axion_sols = []
    red_chem_pot_sols = []
    background_sols = []

    while True:
        tmax_inf_time = tmax_axion_time * conv_factor
        if debug: print("step =", step)
        step += 1
        # reheating process
        sol_rh = decay_process.solve_decay_eqs(tmax_inf_time, rho_R_init, rho_inf_init, Gamma_inf)
        T_and_H_fn, T_and_H_and_T_dot_fn = decay_process.to_temperature_and_hubble_fns(sol_rh, rho_inf_init, Gamma_inf)
        if debug:
            background_sols.append(T_and_H_and_T_dot_fn)

        # evolution of the axion field
        sol_axion = axion_model.solve(axion_init, axion_parameter, tmax_axion_time, T_and_H_fn, Gamma_inf)
        if debug:
            axion_sols.append(sol_axion)
        axion_source = axion_model.get_source(sol_axion, conv_factor)

        # transport eq. for standard model charges
        sol_transp_eq = transport_equation.solve(tmax_inf_time, red_chem_pots_init, T_and_H_and_T_dot_fn,
                axion_source, transport_equation.source_vector_weak_sphaleron, Gamma_inf, conv_factor)
        if debug:
            red_chem_pot_sols.append(sol_transp_eq)

        # check for convergence of the B - L charge
        log_ts_inf = np.linspace(np.log(decay_process.t0 + tmax_inf_time - conv_factor * step_tmax_axion_time),
                np.log(decay_process.t0 + tmax_inf_time), nsamples)
        red_chem_pots = sol_transp_eq(log_ts_inf) # returned function converts internal unit
        B_minus_L_red_chem = transport_equation.calc_B_minus_L(red_chem_pots)
        a, b = np.max(B_minus_L_red_chem), np.min(B_minus_L_red_chem)
        if debug:
            print("B-L range:", b, a)
        delta = np.abs((a - b) / np.mean(B_minus_L_red_chem))
        if debug:
            print("delta =", delta, "convergence_rtol =", convergence_rtol)
        if delta < convergence_rtol: break

        # TODO: check units
        # set initial conditions to our end state and continue
        rho_R_init = decay_process.find_end_rad_energy(sol_rh, rho_inf_init) # [GeV^4]
        rho_inf_init = decay_process.find_end_inf_energy(sol_rh, rho_inf_init) # [GeV^4]
        axion_init = sol_axion.y[:, -1] # (1, energy_scale_axion (independent of initial condition and starting time))
        red_chem_pots_init = red_chem_pots[:, -1] # 1 (unit has been removed)
        tmax_axion_time = step_tmax_axion_time # continue to integrate with the step time

    if debug:
        # background cosmology
        plt.figure()
        tend = 0
        for i, (axion_sol, T_and_H_and_T_dot_fn) in enumerate(zip(axion_sols, background_sols)):
            t_inf_max = conv_factor * axion_sol.t[-1]
            ts_inf = np.linspace(decay_process.t0, decay_process.t0 + t_inf_max, 500)
            T, H, T_dot = T_and_H_and_T_dot_fn(ts_inf)
            plt.loglog(tend + ts_inf, H, label="numerical solution for reheating" if i == 0 else None)
            tend += conv_factor * axion_sol.t[-1]
        H0 = background_sols[0](decay_process.t0)[1]
        ts_inf = np.geomspace(decay_process.t0, tend, 500)
        plt.loglog(ts_inf, 1.0 / (2*((ts_inf - decay_process.t0) / Gamma_inf) + 1/H0),
                label="analytical radiation domination", color="black", ls="--")
        plt.legend()
        plt.xlabel(r"$t \cdot \Gamma_\mathrm{inf}$")
        plt.ylabel(r"$H / \Gamma_\mathrm{inf}$")

        # axion plot
        plt.figure()
        tend = 0
        for axion_sol in axion_sols:
            plt.axvline(1.0, color="black", ls="--")
            plt.axhline(0.0, color="black", ls="-")
            ts_ax = np.linspace(0.0, axion_sol.t[-1], 500)
            ts = tend + ts_ax
            tend += axion_sol.t[-1]
            plt.plot(ts, axion_sol.sol(ts_ax)[0, :])
        plt.xscale("log")
        plt.xlabel(r"$t \cdot m_a(T_\mathrm{osc})$")
        plt.ylabel(r"$\theta$")

        # transport eq. plot
        plt.figure()
        tend = 0
        for j, (axion_sol, red_chem_pot_sol, ls) in enumerate(zip(axion_sols, red_chem_pot_sols, itertools.cycle(("-", "--")))):
            t_inf_max = conv_factor * axion_sol.t[-1]
            ts_inf = np.linspace(decay_process.t0, decay_process.t0 + t_inf_max, 500)
            red_chem_pots = red_chem_pot_sol(np.log(ts_inf))
            for i, (name, color) in enumerate(zip(transport_equation.charge_names, mcolors.TABLEAU_COLORS)):
                plt.plot(ts_inf + tend, np.abs(red_chem_pots[i, :]), ls=ls, color=color, label=name if j == 0 else None)
            plt.plot(ts_inf + tend, np.abs(transport_equation.calc_B_minus_L(red_chem_pots)), label="B - L" if j == 0 else None, color="black", lw=2, ls=ls)
            tend += conv_factor * axion_sol.t[-1]
        plt.xscale("log")
        plt.yscale("log")
        plt.ylim(1e-14, plt.ylim()[1])
        plt.xlabel(r"$t \cdot \Gamma_\mathrm{inf}$")
        plt.ylabel(r"$|\mu_i / T|$")
        plt.legend(ncol=3, framealpha=1)

        plt.show()

    if axion_model.does_decay:
        raise NotImplementedError()
        # dilution factor from axion decay
        while True:
            rho_end_axion = axion_model.get_energy(sol_axion.y[:, -1], energy_scale, f_a, Gamma_inf)
            rho_end_rad = decay_process.find_end_rad_energy(sol_rh, rho_inf_init)
            Gamma_axion = axion_model.get_decay_constant(f_a, *axion_parameter)
            sol_axion_decay = decay_process.solve_decay_eqs(axion_decay_time, rho_end_rad, rho_end_axion, Gamma_axion, debug=debug)
            T_and_H_fn_axion, _ = decay_process.to_temperature_and_hubble_fns(sol_axion_decay, rho_end_axion, Gamma_axion, debug=debug)
            T_fn = lambda t: T_and_H_fn_axion(t)[0]
            f = decay_process.find_dilution_factor(sol_axion_decay, T_fn, debug=debug)


    if debug: plt.show()

