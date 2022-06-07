import numpy as np

# constants
M_pl = 2.435e18 # reduced Planck mass [GeV] from wikipedia (https://en.wikipedia.org/wiki/Planck_mass)
m_pl = 1.220910e19 # Planck mass [GeV] from wikipedia (https://en.wikipedia.org/wiki/Planck_mass)
g_star = 427/4 # during reheating from paper
N_f = 3 # [1] fermion generations
# TODO: renormalization group running?
g_1 = 0.357 # [1] from wikipedia (https://en.wikipedia.org/wiki/Mathematical_formulation_of_the_Standard_Model#Free_parameters)
g_2 = 0.652 # [1] also from wikipedia
g_3 = 1.221 # [1] also from wikipedia
alpha_1 = g_1**2 / (4 * np.pi)
alpha_2 = alpha = g_2**2 / (4 * np.pi) # eq. from paper
alpha_3 = g_3**2 / (4 * np.pi)
M_Z = 91.187 # [GeV]
c_shaleron = 28/79 # from paper
g_star_0 = 43/11 # from paper
eta_B_observed = 6e-10 # from paper
L_to_B_final_factor = c_shaleron * g_star_0 / g_star # formula from paper (*)

# all from wikipedia:
m_tau = 1.77686
m_top = 172.76
m_bottom = 4.18
higgs_vev = 246

# from the genric couplings paper
C_sph = 8 / 23

convergence_epsilon = 1e-2

Omega_DM_h_sq = 0.11933
h = 0.673
rho_c = 3.667106289005098e-11 # [eV^4]
T_CMB = 2.348654180597668e-13 # GeV

log_min_decay_time = 26 # [seconds]
