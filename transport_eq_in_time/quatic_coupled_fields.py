import numpy as np, matplotlib.pyplot as plt
import axion_motion
from importlib import reload; axion_motion = reload(axion_motion)

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
        return self.find_dynamical_scale(*params)

    def find_mass(self, T, m_1, m_2, g, f_1, f_2):
        return m_2 # the second axion has a relic density

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

    does_decay = True
    has_relic_density = True

    Gamma_a_const = alpha**2 / (64 * np.pi**3)
    def get_decay_constant(self, _f_1, m_1, m_2, g, f_1, f_2):
        return self.Gamma_a_const * m_1**3 / f_1**2

quatic_coupled_fields = QuaticCoupledFields()

