import numpy as np
import axion_motion

class KineticAxionField(axion_motion.SingleAxionField):
    def calc_pot(self, theta, __T, m_a):        return m_a**2 * (1 - np.cos(theta))
    def calc_pot_deriv(self, theta, __T, m_a):  return m_a**2 * np.sin(theta)
    def find_dynamical_scale(self, m_a):        return m_a
    def find_H_osc(self, m_a):                  return m_a / 3
    def find_mass(self, m_a):                   return m_a
    def calc_source(self, y, conv_factor, m_a): return y[1] / conv_factor


kinetic_axion_field = KineticAxionField()

