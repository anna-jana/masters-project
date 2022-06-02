import time, importlib, itertools, enum
import numpy as np, matplotlib.pyplot as plt
from scipy.optimize import root
import decay_process, axion_motion, transport_equation, plot_tools
decay_process, axion_motion, transport_equation, plot_tools = map(importlib.reload,
    (decay_process, axion_motion, transport_equation, plot_tools))
                                                                  
Status = enum.Enum("Status", "OK ASYM_CONVERGENCE_FAILURE RELIC_DENSITY_CONVERGENCE_FAILURE AXION_OSCILLATES_BEFORE_INFLATION INFLATON_DECAYS_DURING_INFLATION ISOCURVATURE_BOUNDS")

def calc_entropy_density(T, g_star):
    return 2*np.pi**2 / 45 * g_star * T**3

zeta3 = 1.20206
g_photon = 2
C_sph = 8 / 23
eta_B_observed = 6e-10 # from paper
g_star_0 = 43/11 # from paper
asym_const = - g_star_0 * C_sph * np.pi**2 / (6 * decay_process.g_star * zeta3 * g_photon)
T_CMB = 2.348654180597668e-13 # GeV
s_today = calc_entropy_density(T_CMB, g_star_0)
h = 0.673
rho_c = 3.667106289005098e-11 # [eV^4]
Omega_DM_h_sq = 0.11933

def abundance_to_relic_density(Y, m):
    n = s_today * Y
    rho = m * n
    Omega_h_sq = rho * 1e9**4 / rho_c * h**2
    return Omega_h_sq

def red_chem_pot_to_asymmetry(red_chem_pot_B_minus_L):
    return asym_const * red_chem_pot_B_minus_L

def compute_dilution_factor_from_axion_decay(axion_decay_time, rho_end_rad, rho_end_axion, axion_parameter, f_a, axion_model, debug):
    if not (np.isfinite(rho_end_rad) and np.isfinite(rho_end_axion)):
        return np.nan
    if debug:
        start_decay = time.time()
        print("initial (rad, axion):", rho_end_rad, rho_end_axion)
    # dilution factor from axion decay
    # we don't do converence check for this part right now

    Gamma_axion = axion_model.get_decay_constant(f_a, *axion_parameter) # [GeV]
    axion_scale = decay_process.find_scale(Gamma_axion) # GeV
        
    sol_axion_decay, T_and_H_fn_axion, _ = decay_process.solve(axion_decay_time, rho_end_rad, 
                                                               rho_end_axion, axion_scale, Gamma_axion)

    t = np.exp(sol_axion_decay.t[-1])
    f = decay_process.find_dilution_factor(sol_axion_decay, T_and_H_fn_axion, t)
    
    if debug:
        end_decay = time.time()
        print("axion decay took:", end_decay - start_decay, "seconds")
        plot_tools.plot_dilution_factor_time_evolution(sol_axion_decay, T_and_H_fn_axion)
        
    return f

def solve_system(H_inf, rho_R_init, rho_inf_init, axion_init, red_chem_pots_init,
                 tmax_axion_time, conv_factor, Gamma_inf, scale, 
                 axion_parameter, axion_model, source_vector_axion,
                 calc_init_time, debug):
    rh_start = time.time()
    tmax_inf_time = tmax_axion_time * conv_factor
    
    sol_rh, T_and_H_fn, T_and_H_and_T_dot_fn = decay_process.solve(tmax_inf_time, rho_R_init, rho_inf_init, scale, Gamma_inf)
    if calc_init_time:
        T_eq_general = 1e12 # [GeV]
        Tmax = (
            0.8 * decay_process.g_star**(-1/4) * rho_inf_init**(1/8)
            * (Gamma_inf * decay_process.M_pl / (8*np.pi))**(1/4)
        ) # [GeV]
        if debug:
            print(f"T_max = {Tmax:e}")
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
        t_axion = 2*np.pi*10 # integrate 10 axion oscillations [1/m_a]
        t_RH = decay_process.t0 # we want to reach reheating [1/Gamma_inf]
        tmax_inf_time = max(t_RH, t_eq) # DANGER: max is NOT commutativ if nan is involved, t_eq has to be in the second place!
        tmax_axion_time = tmax_inf_time / conv_factor
        if debug:
            print(f"t_eq = {t_eq}")
            print(f"tmax_inf_time = {tmax_inf_time}, tmax_axion_time = {tmax_axion_time}")
        sol_rh, T_and_H_fn, T_and_H_and_T_dot_fn = decay_process.solve(tmax_inf_time, rho_R_init, rho_inf_init, scale, Gamma_inf)
        if debug:
            print("calculcated initial integration time:")
            print("tmax_inf_time =", tmax_inf_time, "tmax_axion_time =", tmax_axion_time)
    if debug:
        rh_end = time.time()
        print("rh:", rh_end - rh_start)
               
    ############################### evolution of the axion field ##################################
    sol_axion = axion_model.solve(axion_init, axion_parameter, tmax_axion_time, T_and_H_fn, Gamma_inf)
    if debug:
        ax_end = time.time()
        print("axion:", ax_end - rh_end)
    axion_source = axion_model.get_source(sol_axion, conv_factor, *axion_parameter)

    ######################### transport eq. for standard model charges ##############################
    sol_transp_eq = transport_equation.solve(tmax_inf_time, red_chem_pots_init, T_and_H_and_T_dot_fn,
                axion_source, source_vector_axion, Gamma_inf, conv_factor)
    if debug:
        trans_end = time.time()
        print("transport eq.:", trans_end - ax_end)
        
    return sol_rh, T_and_H_fn, T_and_H_and_T_dot_fn, sol_axion, axion_source, sol_transp_eq, tmax_axion_time

def init_system(H_inf, Gamma_inf, axion_parameter, axion_model, tmax_axion_time):
    energy_scale = axion_model.find_dynamical_scale(*axion_parameter)
    conv_factor = Gamma_inf / energy_scale
    rho_R_init = 0.0
    rho_inf_init = 3 * decay_process.M_pl**2 * H_inf**2
    scale = decay_process.find_scale(Gamma_inf)
    tmax_inf_time = tmax_axion_time * conv_factor
    red_chem_pots_init = np.zeros(transport_equation.N)
    return energy_scale, conv_factor, rho_R_init, rho_inf_init, scale, tmax_inf_time, red_chem_pots_init  

def compute_observables(H_inf, Gamma_inf, axion_parameter, f_a, axion_model, axion_init,
        source_vector_axion=transport_equation.source_vector_weak_sphaleron,
        start_tmax_axion_time=10.0, step_tmax_axion_time=2*2*np.pi,
        asym_max_steps=None, relic_max_steps=None,
        axion_decay_time=10.0, debug=False,
        nosc_per_step=5, nsamples_per_osc=20,
        rtol_asym=1e-3, rtol_relic=1e-3,
        nsamples=100, calc_init_time=False, isocurvature_check=False):
    
    ############################### setup for asymmetry computation ########################
    status = Status.OK

    step = 1
    axion_sols = []
    red_chem_pot_sols = []
    background_sols = []
    
    tmax_axion_time = start_tmax_axion_time # initial time to integrate    
    energy_scale, conv_factor, rho_R_init, rho_inf_init, scale, tmax_inf_time, red_chem_pots_init = \
        init_system(H_inf, Gamma_inf, axion_parameter, axion_model, tmax_axion_time)
    if debug:
        print("conv factor:", conv_factor)
        
    ############################## check parameter consistency ####################
    if energy_scale > H_inf:
        return np.nan, np.nan, np.nan, np.nan, np.nan, Status.AXION_OSCILLATES_BEFORE_INFLATION.value
        #(invalidates the assumtion that the axion doesn't iterfere with inflation)
    if Gamma_inf > H_inf:
        return np.nan, np.nan, np.nan, np.nan, np.nan, Status.INFLATON_DECAYS_DURING_INFLATION.value    
    if isocurvature_check and H_inf/(2*np.pi)/f_a < 1e-5: # eq. 1 in 1412.2043
        return np.nan, np.nan, np.nan, np.nan, np.nan, Status.ISOCURVATURE_BOUNDS.value

    ################################## asymmmetry convergence loop ########################
    while True:
        if debug: print("step =", step)
        
        ################################ advance system ###################################
        sol_rh, T_and_H_fn, T_and_H_and_T_dot_fn, sol_axion, axion_source, sol_transp_eq, tmax_axion_time = solve_system(
                 H_inf, rho_R_init, rho_inf_init, axion_init, red_chem_pots_init,
                 tmax_axion_time, conv_factor, Gamma_inf, scale, axion_parameter, axion_model, source_vector_axion,
                 calc_init_time and step == 1, debug)
        tmax_inf_time = tmax_axion_time * conv_factor
        if debug:
            axion_sols.append(sol_axion)
            red_chem_pot_sols.append(sol_transp_eq)
            background_sols.append(T_and_H_and_T_dot_fn)
        
        ####################### check for convergence of the B - L charge ###########################
        log_ts_inf = np.linspace(np.log(decay_process.t0), np.log(decay_process.t0 + tmax_inf_time), nsamples)
        red_chem_pots = sol_transp_eq(log_ts_inf) # returned function converts internal unit
        B_minus_L_red_chem = transport_equation.calc_B_minus_L(red_chem_pots)
        if debug:
            print("B-L start .. end:", B_minus_L_red_chem[0], B_minus_L_red_chem[-1])
        
        # convergence by change within the last integration interval
        a, b = np.max(B_minus_L_red_chem), np.min(B_minus_L_red_chem)
        delta = np.abs((a - b) / np.mean(B_minus_L_red_chem))
        if debug:
            print("B-L range:", b, a)
            print("delta =", delta, "rtol_asym =", rtol_asym)
        if delta < rtol_asym:
            break
        if asym_max_steps is not None and step > asym_max_steps:
            status = Status.ASYM_CONVERGENCE_FAILURE
            break
        
        # set initial conditions to our end state and continue
        rho_R_init = decay_process.find_end_rad_energy(sol_rh, scale) # [GeV^4]
        rho_inf_init = decay_process.find_end_field_energy(sol_rh, rho_inf_init) # [GeV^4]
        axion_init = sol_axion.y[:, -1] # (1, energy_scale_axion (independent of initial condition and starting time))
        red_chem_pots_init = red_chem_pots[:, -1] # 1 (unit has been removed)
        tmax_axion_time = step_tmax_axion_time # continue to integrate with the step time
        step += 1
    
    ###################################### finish asymmetry #########################################
    if debug:
        plot_tools.plot_asymmetry_time_evolution(axion_model, conv_factor, Gamma_inf, axion_parameter, f_a, 
                                                 background_sols, axion_sols, red_chem_pot_sols)
    
    # use the last value of the reduced chemical potentials 
    eta_B = red_chem_pot_to_asymmetry(transport_equation.calc_B_minus_L(red_chem_pots[:, -1]))
    
    rho_end_rad = decay_process.find_end_rad_energy(sol_rh, scale) # [GeV^4]
    rho_end_axion = axion_model.get_energy(sol_axion.y[:, -1], f_a, *axion_parameter) # [GeV^4]

    ############################ entropy dilution from axion decay ################################
    if axion_model.does_decay:
        f = compute_dilution_factor_from_axion_decay(axion_decay_time, rho_end_rad, rho_end_axion, 
                                                     axion_parameter, f_a, axion_model, debug)
        Omega_h_sq = 0.0
    
    ################################## axion relic density ######################################
    elif axion_model.has_relic_density:
        f = 1.0
        if debug:
            relic_density_start = time.time()

        # initial
        H_osc = axion_model.find_H_osc(*axion_parameter)
        _, H = T_and_H_fn(np.exp(sol_rh.t[-1]))

        def H_to_t(H):
            return 1 / 2 * (1/H + 1/(H_inf / energy_scale))

        ################ evolve to oscillation start, if nessesary ####################
        if H > H_osc:
            t = H_to_t(H)
            t_osc = H_to_t(H_osc)
            t_advance_axion = t_osc - t
            t_advance_inf = t_advance_axion * conv_factor

            rho_R_init = decay_process.find_end_rad_energy(sol_rh, scale)
            rho_inf_init = decay_process.find_end_field_energy(sol_rh, rho_inf_init)

            if debug:
                print("advancing to oscillation")
                print("t_advance =", t_advance_axion)
                print(f"H = {H}, H_osc = {H_osc}")

            sol_rh, T_and_H_fn, _ = decay_process.solve(t_advance_inf, rho_R_init, rho_inf_init, scale, Gamma_inf)
            advance_sol_axion = sol_axion = axion_model.solve(sol_axion.y[:, -1], axion_parameter, 
                                                              t_advance_axion, T_and_H_fn, Gamma_inf)
        else:
            advance_sol_axion = None
            t_advance_inf = 0.0

        ####################### convergence of axion abundance ####################           
        last_Y_estimate = np.nan
        Y_samples = []
        step = 1
        
        while True:
            ######## advance background cosmology and axion field ###########
            current_T, _ = T_and_H_fn(np.exp(sol_rh.t[-1]))
            freq = axion_model.find_mass(current_T, *axion_parameter)
            step_time_axion = nosc_per_step * 2*np.pi / freq
            step_time_inf = step_time_axion * conv_factor
            rho_R_init = decay_process.find_end_rad_energy(sol_rh, scale)
            rho_inf_init = decay_process.find_end_field_energy(sol_rh, rho_inf_init)

            scale = rho_R_init
            sol_rh, T_and_H_fn, _ = decay_process.solve(step_time_inf, rho_R_init, rho_inf_init, scale, Gamma_inf)
            sol_axion = axion_model.solve(sol_axion.y[:, -1], axion_parameter, step_time_axion, T_and_H_fn, Gamma_inf)
            
            ############ convergence check ############
            ts_ax = np.linspace(0, step_time_axion, nsamples_per_osc * nosc_per_step)
            ts_inf = conv_factor * ts_ax + decay_process.t0
            T, H = T_and_H_fn(ts_inf)
            s = calc_entropy_density(T, decay_process.g_star) # [GeV^3]
            ys = sol_axion.sol(ts_ax)
            rho_over_f_sq = np.array([axion_model.get_energy(y, 1.0, *axion_parameter) for y in ys.T])
            m = axion_model.find_mass(T, *axion_parameter)
            n_over_f_sq = rho_over_f_sq / m # [GeV]
            Y = n_over_f_sq / s # [GeV^-2] bc its n/s/f^2
            is_min = (Y[2:] > Y[1:-1]) & (Y[:-2] > Y[1:-1])
            is_max = (Y[2:] < Y[1:-1]) & (Y[:-2] < Y[1:-1])
            Y_estimate = (np.sum(Y[1:-1][is_min]) + np.sum(Y[1:-1][is_max])) / (np.sum(is_min) + np.sum(is_max))
            delta = np.abs(Y_estimate - last_Y_estimate) / Y_estimate
            
            if debug:
                Y_samples.append((ts_ax, Y, is_min, is_max, Y_estimate))
                print(f"step = {step}: delta = {delta} vs {rtol_relic}")
            if delta < rtol_relic:
                break
            if relic_max_steps is not None and step > relic_max_steps:
                status = Status.RELIC_DENSITY_CONVERGENCE_FAILURE
                break
            last_Y_estimate = Y_estimate
            step += 1
        
        ######### finish relic density ###########
        m_today = axion_model.find_mass(T_CMB, *axion_parameter)
        Omega_h_sq = abundance_to_relic_density(Y_estimate, m_today) * f_a**2

        if debug:
            relic_density_end = time.time()
            print("relic density took:", relic_density_end - relic_density_start, "seconds")
            plot_tools.plot_relic_density_time_evolution(conv_factor, t_advance_inf, advance_sol_axion, Y_samples)
    else:
        Omega_h_sq = 0.0
        f = 1.0
    ###################################### final results ######################################
    return eta_B, f, rho_end_rad, rho_end_axion, Omega_h_sq, float(status.value)