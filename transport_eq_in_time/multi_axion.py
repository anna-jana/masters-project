from importlib import reload
import numpy as np, matplotlib.pyplot as plt
import axion_motion
axion_motion = reload(axion_motion)

class MultiAxionField(axion_motion.AxionField):
    def rhs(self, t, y, T_and_H_fn, energy_scale, axion_parameter): 
        Q, Lambda = axion_parameter
        _, H = T_and_H_fn(t)
        N = y.size // 2
        thetas, theta_dots = y[:N], y[N:]
        force = (Lambda * np.sin(Q @ thetas)) @ Q
        theta_dotdots = - 3 * H * theta_dots - force
        return np.hstack([theta_dots, theta_dotdots])
        
    def find_dynamical_scale(self, *axion_parameter): raise NotImplementedError()
    def find_H_osc(self, *axion_parameter): raise NotImplementedError()
    def find_mass(self, T, *axion_parameter): raise NotImplementedError()

    def calc_source(self, y, conv_factor, *axion_parameter):
        
    
    def calc_V(self, thetas, Q, Lambda):
        return np.dot(Lambda, 1 - np.cos(Q @ thetas))

    def get_energy(self, y, f_a, Gamma_inf, Q, Lambda):
        N = len(Lambda)
        thetas, theta_dots = y[:N], y[N:]
        return 0.5 * np.sum(theta_dots**2) + calc_V(thetas, Q, Lambda)

    does_decay = NotImplemented
    has_relic_density = NotImplemented
    def get_decay_constant(self): raise NotImplementedError()


multi_axion_field = MultiAxionField()
    