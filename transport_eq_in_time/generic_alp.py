import importlib, pickle, os, itertools, numpy as np, tqdm, tqdm.notebook
import axion_motion, analysis_tools, runner, observables
axion_motion, analysis_tools, runner, observables = map(importlib.reload, (axion_motion, analysis_tools, runner, observables))

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

H_inf_index = 0
f_a_index = 0

def recompute_dilution(data, f_a, notebook=False):
    progress = tqdm.notebook.tqdm if notebook else tqdm.tqdm
    m_a, Gamma_inf = data["m_a"], data["Gamma_inf"]
    rho_end_rad = data["rho_end_rad"][H_inf_index, :, :, f_a_index]
    rho_end_axion = data["rho_end_axion"][H_inf_index, :, :, f_a_index]     
    f_a_used = data["f_a"][f_a_index]
    dilution = np.zeros(rho_end_axion.shape)
    for i, j in progress(list(itertools.product(range(len(Gamma_inf)), range(len(m_a))))):
        dilution[i, j] = observables.compute_dilution_factor_from_axion_decay(10.0, 
                rho_end_rad[i, j], rho_end_axion[i, j] / f_a_used**2 * f_a**2, 
                (m_a[j],), f_a, realignment_axion_field, False)  
    return dilution

def compute_correct_curves(version):
    correct_alp_curves_filename = os.path.join(runner.datadir, f"generic_alp_correct_curves{version}.pkl")
    data = runner.load_data("generic_alp", version)
    eta = data["eta"][H_inf_index, :, :, f_a_index]
    m_a = data["m_a"]
    Gamma_inf = data["Gamma_inf"]
    H_inf = data["H_inf"]
    rho_end_rad = data["rho_end_rad"][H_inf_index, :, :, f_a_index]
    rho_end_axion = data["rho_end_axion"][H_inf_index, :, :, f_a_index]     
    f_a_used = data["f_a"][f_a_index]

    f_a_list = np.geomspace(1e12, 1e15, 10)
    correct_asym_curves = []

    for f_a in tqdm.tqdm(f_a_list, position=0):
        eta_B = eta * recompute_dilution(data, f_a)
        levels = find_level(np.log10(m_a), np.log10(Gamma_inf), np.log10(np.abs(eta_B) / eta_B_observed))
        correct_asym_curves.append([(10**xs, 10**ys) for xs, ys in levels])

    with open(correct_alp_curves_filename, "wb") as fhandle:
        pickle.dump((f_a_list, correct_asym_curves), fhandle)