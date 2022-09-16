import numpy as np, matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import decay_process

# abstract class for axion fields
class AxionField:
    def rhs(self, t, y, T_and_H_fn, energy_scale, axion_parameter): raise NotImplementedError()
    def find_dynamical_scale(self, *axion_parameter): raise NotImplementedError()
    def find_H_osc(self, *axion_parameter): raise NotImplementedError()
    def find_mass(self, T, *axion_parameter): raise NotImplementedError()

    def solve(self, axion_init, axion_parameter, tmax_axion_time, T_and_H_fn, Gamma_inf,
              rtol=1e-10, method="Radau"):
        energy_scale = self.find_dynamical_scale(*axion_parameter)
        conv_factor = Gamma_inf / energy_scale
        sol = solve_ivp(self.rhs, (0.0, tmax_axion_time), axion_init,
                args=(lambda t: T_and_H_fn(conv_factor * t + decay_process.t0), energy_scale, axion_parameter),
                dense_output=True, rtol=rtol, method=method)
        assert sol.success
        return sol

    def calc_source(self, y, conv_factor, *axion_parameter): raise NotImplementedError()
    def get_source(self, sol, conv_factor, *axion_parameter):
        def source(t_inf):
            y = sol.sol((t_inf - decay_process.t0) / conv_factor)
            return self.calc_source(y, conv_factor, *axion_parameter)
        return source

    def get_energy(self, y, f_a, *axion_parameter): raise NotImplementedError()

    does_decay = NotImplemented
    has_relic_density = NotImplemented

class SingleAxionField(AxionField):
    def calc_pot_deriv(self, theta, T, m_a): raise NotImplementedError()
    def calc_pot(self, theta, T, m_a): raise NotImplementedError()
    # temperature is in GeV
    # axion time scale: 1 / m_a(T_osc)
    def rhs(self, t, y, T_and_H_fn, energy_scale, axion_parameter):
        theta, theta_dot = y
        T, H = T_and_H_fn(t) # T_and_H_fn has been transformed to take axion time
        return (theta_dot, - 3*H/energy_scale*theta_dot - self.calc_pot_deriv(theta, T, *axion_parameter)/energy_scale**2)



