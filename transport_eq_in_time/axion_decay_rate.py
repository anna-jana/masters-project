import numpy as np
import transport_equation

def get_axion_decay_rate(source_vector, f_a, m_a):
    alpha_eff_sq = 0.0
    alpha_eff_sq += source_vector[0] * transport_equation.alpha_2**2
    alpha_eff_sq += - source_vector[-1] / 2 * transport_equation.alpha_2**2
    alpha_eff_sq += source_vector[1] * transport_equation.alpha_3**2
    Gamma_a_const = alpha_eff_sq / (64 * np.pi**3)
    return Gamma_a_const * m_a**3 / f_a**2

