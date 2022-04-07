import time, importlib
import numpy as np, matplotlib.pyplot as plt
import decay_process, axion_motion, transport_equation
decay_process = importlib.reload(decay_process)
axion_motion = importlib.reload(axion_motion)
transport_equation = importlib.reload(transport_equation)

def test(H_inf, Gamma_inf, m_a, f_a, tmax_axion_time=10.0, axion_decay_time=10.0, debug=True):
    axion_parameter = (m_a,)
    rho0 = 3*decay_process.M_pl**2*H_inf**2
    # this is a bit useless but I keep to make it work like the general case
    energy_scale = axion_motion.single_axion_field.find_dynamical_scale(*axion_parameter)
    conv_factor = Gamma_inf / energy_scale
    tmax_inf_time = tmax_axion_time * conv_factor

    rho_R_init = 0.0
    rho_a_init = rho0
    axion_init = (1.0, 0.0)
    red_chem_pots_init = np.zeros(transport_equation.N)

    # reheating process
    start_time = time.time()
    sol_rh = decay_process.solve_decay_eqs(tmax_inf_time, 0.0, rho0, Gamma_inf, debug=debug)
    T_and_H_fn, T_and_H_and_T_dot_fn = decay_process.to_temperature_and_hubble_fns(sol_rh, rho0, Gamma_inf, debug=debug)
    decay_time = time.time()
    print("decay done, took:", decay_time - start_time, "seconds")

    # evolution of the axion field
    sol_axion = axion_motion.single_axion_field.solve(axion_init, axion_parameter, tmax_axion_time, T_and_H_fn, Gamma_inf, debug=debug)
    axion_source = axion_motion.single_axion_field.get_source(sol_axion, conv_factor)
    axion_time = time.time()
    print("axion done, took:", axion_time - decay_time, "seconds")

    # transport eq. for standard model charges
    sol_transp_eq = transport_equation.solve(tmax_inf_time, red_chem_pots_init, T_and_H_and_T_dot_fn,
            axion_source, transport_equation.source_vector_weak_sphaleron, Gamma_inf, conv_factor, debug=debug)
    trans_time = time.time()
    print("trans done, took:", trans_time - axion_time, "seconds")

    # dilution factor from axion decay
    rho_end_axion = axion_motion.single_axion_field.get_energy(sol_axion.y[:, -1], energy_scale, f_a, Gamma_inf)
    rho_end_rad = decay_process.find_end_rad_energy(sol_rh, rho0)
    Gamma_axion = axion_motion.single_axion_field.get_decay_constant(f_a, *axion_parameter)
    sol_axion_decay = decay_process.solve_decay_eqs(axion_decay_time, rho_end_rad, rho_end_axion, Gamma_axion, debug=debug)
    T_and_H_fn_axion, _ = decay_process.to_temperature_and_hubble_fns(sol_axion_decay, rho_end_axion, Gamma_axion, debug=debug)
    T_fn = lambda t: T_and_H_fn_axion(t)[0]
    f = decay_process.find_dilution_factor(sol_axion_decay, T_fn, debug=debug)
    decay_time = time.time()
    print("axion decay done, took:", decay_time - trans_time, "seconds")

    plt.show()


