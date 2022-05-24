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
        theta_dotdots = - 3 * H / energy_scale * theta_dots - force / energy_scale**2
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
        return y[N] / conv_factor # the first axion is coupled to the standard model

    def calc_V(self, thetas, Q, Lambda):
        return np.dot(Lambda, 1 - np.cos(Q @ thetas))

    def get_energy(self, y, f_a, Q, Lambda):
        N = len(Lambda)
        thetas, theta_dots = y[:N], y[N:]
        energy_scale = self.find_dynamical_scale(Q, Lambda)
        return f_a**2 * (0.5 * energy_scale**2 * np.sum(theta_dots**2) + self.calc_V(thetas, Q, Lambda))
    
    def calc_mass_state(self, u, Q, Lambda):
        N = len(Lambda)
        return np.vstack([Q @ u[:N, :], Q @ u[N:, :]])

    does_decay = True
    has_relic_density = True

    def get_decay_constant(self): raise NotImplementedError()


multi_axion_field = MultiAxionField()

canonical_names = [r"$\theta_1$", r"$\theta_2$", r"$\dot{\theta}_1$", r"$\dot{\theta}_2$"]
mass_states_names = [r"$\phi_1$", r"$\phi_2$", r"$\dot{\phi}_1$", r"$\dot{\phi}_2$"]