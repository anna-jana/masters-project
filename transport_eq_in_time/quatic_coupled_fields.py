import numpy as np, matplotlib.pyplot as plt
import axion_motion
from importlib import reload; axion_motion = reload(axion_motion)
from decay_process import M_pl, g_star
from observables import rho_c, h, g_star_0, T_CMB, Omega_DM_h_sq

class QuaticCoupledFields(axion_motion.AxionField):
    def rhs(self, t, y, T_and_H_fn, energy_scale, axion_parameter):
        m_1, m_2, g, f_1, f_2 = axion_parameter
        _, H = T_and_H_fn(t)
        theta1, theta2, theta_dot1, theta_dot2 = y
        theta_dotdot1 = (
                - 3 * H / energy_scale * theta_dot1
               - theta1 * theta2**2 * f_2**2 / g / energy_scale**2
       )
        theta_dotdot2 = (
            - 3 * H / energy_scale * theta_dot2
            - theta2 * theta1**2 * f_1**2 / energy_scale**2
        )
        return theta_dot1, theta_dot2, theta_dotdot1, theta_dotdot2

    def find_dynamical_scale(self, *params):
        return max(np.sqrt(m_1**2 + g * f_2**2), np.sqrt(m_2**2 + g * f_1**2))

    def find_H_osc(self, *params):
        return 1 / 3

    def find_mass(self, T, m_1, m_2, g, f_1, f_2):
        raise NotImplementedError
         # return m_2 / self.find_dynamical_scale(self, m_1, m_2, g, f_1, f_2) # the second axion has a relic density

    def calc_source(self, y, conv_factor, Q, Lambda):
        return y[2] / conv_factor # the first axion is coupled to the standard model

    def calc_V(self, thetas, m_1, m_2, g, f_1, f_2):
        theta1, theta2 = thetas
        return (
            0.5 * m_1**2 * f_1**2 * theta1**2 +
            0.5 * m_2**2 * f_2**2 * theta2**2 +
            g * f_1**2 * f_2**2 * theta1**2 * theta2**2
        )

    def get_energy(self, y, m_1, m_2, g, f_1, f_2):
        theta1, theta2, theta_dot1, theta_dot2 = y
        energy_scale = self.find_dynamical_scale(Q, Lambda)
        return (
            0.5 * f_1**2 * theta_dot1**2 +
            0.5 * f_2**2 * theta_dot2**2 +
            self.calc_V((theta1, theta1), m_1, m_2, g, f_1, f_2)
        )

    does_decay = False # True # -> its not feaseble to simulate this
    has_relic_density = False # True # -> -> its not feaseble to simulate this    

def relic_density_quatic_coupled_two_fields(initial_ratio, q, m_1, m_2, g, f_1, f_2):
    g_2 = 0.652 # [1] also from wikipedia
    alpha = g_2**2 / (4 * np.pi) # eq. from paper
    kappa = alpha**2 / (64 * np.pi**3)
    H_to_a = lambda H: 1 / np.sqrt(2 * H)
    m_eff_max = np.sqrt(2 * g) * np.maximum(f_1, f_2)
    H_osc = m_eff_max / 3 
    decay_rate = kappa * m_1**3 / f_1**2
    H_decay = decay_rate
    a_osc = H_to_a(H_osc)
    a_decay = H_to_a(H_decay)
    rho_decay = g * f_1**2 * f_2**2 * initial_ratio**2 * (a_osc / a_decay)**q
    T_osc2 = (10 * M_pl**2 * m_2**2 / (np.pi**2 * g_star))**(1/4)
    rho_today = g_star_0 / g_star * T_CMB**3 / T_osc2**3 * rho_decay
    Omega_h_sq = rho_today * (1e9)**4 / rho_c * h**2
    return Omega_h_sq 

quatic_coupled_fields = QuaticCoupledFields()

