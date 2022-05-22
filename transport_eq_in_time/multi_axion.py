import numpy as np, matplotlib.pyplot as plt
import axion_motion
from importlib import reload; axion_motion = reload(axion_motion)

class MultiAxionField(axion_motion.AxionField):
    def rhs(self, t, y, T_and_H_fn, energy_scale, axion_parameter):
        Q, Lambda = axion_parameter
        _, H = T_and_H_fn(t)
        N = y.size // 2
        thetas, theta_dots = y[:N], y[N:]
        force = (Lambda * np.sin(Q @ thetas)) @ Q
        theta_dotdots = - 3 * H * theta_dots - force
        return np.hstack([theta_dots, theta_dotdots])

    def find_dynamical_scale(self, Q, Lambda):
        return np.sqrt(np.max(Lambda))

    def find_H_osc(self, Q, Lambda):
        return self.find_mass(Q, Lambda) / 3

    def find_mass(self, T, Q, Lambda):
        N = len(Lambda)
        return np.sqrt(Lambda[N - 1])

    def calc_source(self, y, conv_factor, Q, Lambda):
        N = len(Lambda)
        return y[N] / conv_factor

    def calc_V(self, thetas, Q, Lambda):
        return np.dot(Lambda, 1 - np.cos(Q @ thetas))

    def get_energy(self, y, f_a, Q, Lambda):
        N = len(Lambda)
        thetas, theta_dots = y[:N], y[N:]
        energy_scale = self.find_dynamical_scale(Q, Lambda)
        return f_a**2 * (0.5 * energy_scale**2 * np.sum(theta_dots**2) + calc_V(thetas, Q, Lambda))

    does_decay = True
    has_relic_density = True

    def get_decay_constant(self): raise NotImplementedError()


multi_axion_field = MultiAxionField()

