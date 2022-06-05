import importlib, pickle, os, itertools, numpy as np, tqdm, tqdm.notebook
import axion_motion, analysis_tools, runner, observables, transport_equation, decay_process, analysis_tools
axion_motion, analysis_tools, runner, observables, transport_equation, decay_process, analysis_tools = \
    map(importlib.reload, (axion_motion, analysis_tools, runner, observables, transport_equation, decay_process, analysis_tools))

class RealignmentAxionField(axion_motion.SingleAxionField):
    def calc_pot_deriv(self, theta, T, m_a): return m_a**2 * theta
    def calc_pot(self, T, m_a): return 0.5 * m_a**2 * theta**2
    def find_dynamical_scale(self, m_a): return m_a
    def get_energy(self, y, f_a, m_a):
        theta, theta_dot = y
        energy_scale = self.find_dynamical_scale(m_a)
        return 0.5 * f_a**2 * (theta_dot * energy_scale)**2 + 0.5 * m_a**2 * f_a**2 * theta**2
    def calc_source(self, y, conv_factor, __m_a): return y[1] / conv_factor
    does_decay = True
    has_relic_density = False
    g_2 = 0.652 # [1] also from wikipedia
    alpha = g_2**2 / (4 * np.pi) # eq. from paper
    Gamma_a_const = alpha**2 / (64 * np.pi**3)
    # from paper a to SU(2) gauge bosons
    def get_decay_constant(self, f_a, m_a): return self.Gamma_a_const * m_a**3 / f_a**2
    def find_H_osc(self, m_a): return 1 / 3
    def find_mass(self, T, m_a): return 1.0

realignment_axion_field = RealignmentAxionField()

def recompute_dilution(data, f_a, notebook=False):
    progress = tqdm.notebook.tqdm if notebook else tqdm.tqdm
    m_a, Gamma_inf = data["m_a"], data["Gamma_inf"]
    rho_end_rad = data["rho_end_rad"][0, :, :, 0, 0]
    rho_end_axion = data["rho_end_axion"][0, :, :, 0, 0]     
    f_a_used = data["f_a"][0]
    dilution = np.zeros(rho_end_axion.shape)
    for i, j in progress(list(itertools.product(range(len(Gamma_inf)), range(len(m_a))))):
        dilution[i, j] = observables.compute_dilution_factor_from_axion_decay(10.0, 
                rho_end_rad[i, j], rho_end_axion[i, j] / f_a_used**2 * f_a**2, 
                (m_a[j],), f_a, realignment_axion_field, False)  
    return dilution

def compute_correct_curves(version):
    correct_alp_curves_filename = os.path.join(runner.datadir, f"generic_alp_correct_curves{version}.pkl")
    data = runner.load_data("generic_alp", version)
    eta = data["eta"][0, :, :, 0, 0]
    m_a = data["m_a"]
    Gamma_inf = data["Gamma_inf"]
    rho_end_rad = data["rho_end_rad"][0, :, :, 0, 0]
    rho_end_axion = data["rho_end_axion"][0, :, :, 0, 0]     
    f_a_used = data["f_a"][0]

    f_a_list = np.geomspace(1e12, 1e15, 10)
    correct_asym_curves = []

    for f_a in tqdm.tqdm(f_a_list, position=0):
        eta_B = eta * recompute_dilution(data, f_a)
        levels = find_level(np.log10(m_a), np.log10(Gamma_inf), np.log10(np.abs(eta_B) / eta_B_observed))
        correct_asym_curves.append([(10**xs, 10**ys) for xs, ys in levels])

    with open(correct_alp_curves_filename, "wb") as fhandle:
        pickle.dump((f_a_list, correct_asym_curves), fhandle)
        
def compute_example_trajectories(f_a, H_inf, interesting_points, notebook=False, output_filename=None):
    interesting_solutions = []

    for m_a, Gamma_inf in (tqdm.notebook.tqdm if notebook else tqdm.tqdm)(interesting_points):
        background_sols, axion_sols, red_chem_pot_sols = \
            observables.compute_observables(H_inf, Gamma_inf, (m_a,), f_a, realignment_axion_field, 
                                (1, 0), calc_init_time=True, return_evolution=True)
        ts = np.array([], dtype="d")
        sources = np.empty_like(ts)
        rates = np.empty_like(ts)
        red_chem_potss = np.empty((transport_equation.N, 0))
        tstart = 0
        conv_factor = Gamma_inf / m_a

        for T_and_H_and_T_dot_fn, axion_sol, red_chem_pot_sol in zip(background_sols, axion_sols, red_chem_pot_sols):
            tmax_axion = axion_sol.t[-1]
            tmax_inf = tmax_axion * conv_factor
            tinfs = np.geomspace(decay_process.t0, decay_process.t0 + tmax_inf, 300)
            taxs = (tinfs - decay_process.t0) / conv_factor
            plot_ts = tstart + tinfs
            tstart += tmax_inf 

            Ts, Hs, Tdots = T_and_H_and_T_dot_fn(tinfs)
            theta_dots = axion_sol.sol(taxs)[1, :]
            red_chem_pots = red_chem_pot_sol(np.log(tinfs))

            gammas = [transport_equation.calc_rate_vector(T) for T in Ts]
            source = theta_dots * m_a / Ts # m_a for unit conversion
            rate = - np.array([gamma @ transport_equation.charge_vector @ transport_equation.charge_vector_B_minus_L 
                                            for gamma in gammas]) / Hs
            ts = np.hstack([ts, plot_ts])
            sources = np.hstack([sources, source])
            rates = np.hstack([rates, rate])
            red_chem_potss = np.hstack([red_chem_potss, red_chem_pots])

        interesting_solutions.append((conv_factor, ts, sources, rates, red_chem_potss))
    
    if output_filename is not None:
        with open(output_filename, "bw") as fh:
            pickle.dump(interesting_solutions, fh)
    
    return interesting_solutions