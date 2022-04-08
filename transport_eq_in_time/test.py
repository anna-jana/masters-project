import time, importlib
import numpy as np, matplotlib.pyplot as plt
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
    sol_rh = decay_process.solve_decay_eqs(tmax_inf_time, 0.0, rho0, Gamma_inf, debug=debug)
    T_and_H_fn, T_and_H_and_T_dot_fn = decay_process.to_temperature_and_hubble_fns(sol_rh, rho0, Gamma_inf, debug=debug)
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
    rho_end_rad = decay_process.find_end_rad_energy(sol_rh, rho0)
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
        debug=False, convergence_rtol=1e-6, nsamples=100):
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

    while True:
        tmax_inf_time = tmax_axion_time * conv_factor
        if debug: print("step =", step)
        step += 1
        # reheating process
        sol_rh = decay_process.solve_decay_eqs(tmax_inf_time, rho_R_init, rho_inf_init, Gamma_inf, debug=debug)
        T_and_H_fn, T_and_H_and_T_dot_fn = decay_process.to_temperature_and_hubble_fns(sol_rh, rho_inf_init, Gamma_inf, debug=debug)

        # evolution of the axion field
        sol_axion = axion_model.solve(axion_init, axion_parameter, tmax_axion_time, T_and_H_fn, Gamma_inf, debug=debug)
        axion_source = axion_model.get_source(sol_axion, conv_factor)

        # transport eq. for standard model charges
        sol_transp_eq = transport_equation.solve(tmax_inf_time, red_chem_pots_init, T_and_H_and_T_dot_fn,
                axion_source, transport_equation.source_vector_weak_sphaleron, Gamma_inf, conv_factor, debug=debug)

        # check for convergence of the B - L charge
        log_ts_inf = np.linspace(np.log(decay_process.t0 + tmax_inf_time - conv_factor * step_tmax_axion_time),
                np.log(decay_process.t0 + tmax_inf_time), nsamples)
        red_chem_pots = sol_transp_eq.sol(log_ts_inf)
        B_minus_L_red_chem = transport_equation.calc_B_minus_L(red_chem_pots)
        delta = (np.max(B_minus_L_red_chem) - np.min(B_minus_L_red_chem)) / np.mean(B_minus_L_red_chem)
        if debug:
            print("delta =", delta, "convergence_rtol =", convergence_rtol)
        if delta < convergence_rtol: break

        # TODO: check units
        # set initial conditions to our end state and continue
        rho_R_init = decay_process.find_end_rad_energy(sol_rh, rho_inf_init) # [GeV^4]
        rho_inf_init = decay_process.find_end_inf_energy(sol_rh, rho_inf_init) # [GeV^4]
        axion_init = sol_axion.y[:, -1] # (1, Gamma_inf)
        red_chem_pots_init = sol_transp_eq.y[:, -1] # 1 / unit
        tmax_axion_time = step_tmax_axion_time # continue to integrate with the step time

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

