import numpy as np, matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import decay_process

# abstract class for axion fields
class AxionField:
    def rhs(self, t, y, T_and_H_fn, energy_scale, axion_parameter): raise NotImplementedError()
    def find_dynamical_scale(self, *axion_parameter): raise NotImplementedError()
    def find_H_osc(self, *axion_parameter): raise NotImplementedError()
    def find_osc_period(self, *axion_parameter): raise NotImplementedError()

    def solve(self, axion_init, axion_parameter, tmax_axion_time, T_and_H_fn, Gamma_inf,
              debug=False, rtol=1e-10, method="Radau"):
        energy_scale = self.find_dynamical_scale(*axion_parameter)
        conv_factor = Gamma_inf / energy_scale
        sol = solve_ivp(self.rhs, (0.0, tmax_axion_time), axion_init,
                args=(lambda t: T_and_H_fn(conv_factor * t + decay_process.t0), energy_scale, axion_parameter),
                dense_output=True, rtol=rtol, method=method)
        assert sol.success
        if debug:
            plt.figure()
            plt.axvline(1.0, color="black", ls="--")
            plt.axhline(0.0, color="black", ls="-")
            N = 400
            ts = np.geomspace(sol.t[1], sol.t[-1], N)
            plt.plot(ts, sol.sol(ts)[0,:])
            plt.xscale("log")
            plt.xlabel(r"$t \cdot m_a(T_\mathrm{osc})$")
            plt.ylabel(r"$\theta$")
            plt.figure()
            plt.loglog(ts, [self.get_energy(sol.sol(t), 1.0, Gamma_inf, *axion_parameter) for t in ts])
            plt.xlabel("t * m_a")
            plt.ylabel("~ energy density")
        return sol

    def calc_source(self, y, conv_factor, *axion_parameter): raise NotImplementedError()
    def get_source(self, sol, conv_factor, *axion_parameter):
        def source(t_inf):
            y = sol.sol((t_inf - decay_process.t0) / conv_factor)
            return self.calc_source(y, conv_factor, *axion_parameter)
        return source

    def get_energy(self, y, f_a, Gamma_inf, *axion_parameter): raise NotImplementedError()

    does_decay = NotImplemented
    has_relic_density = NotImplemented
    def get_decay_constant(self): raise NotImplementedError()
    def find_relic_density(self): raise NotImplementedError()

class SingleAxionField(AxionField):
    def calc_pot_deriv(self, theta, T, m_a): raise NotImplementedError()
    def calc_pot(self, theta, T, m_a): raise NotImplementedError()
    # temperature is in GeV
    # axion time scale: 1 / m_a(T_osc)
    def rhs(self, t, y, T_and_H_fn, energy_scale, axion_parameter):
        theta, theta_dot = y
        T, H = T_and_H_fn(t) # T_and_H_fn has been transformed to take axion time
        return (theta_dot, - 3*H/energy_scale*theta_dot - self.calc_pot_deriv(theta, T, *axion_parameter)/energy_scale**2)

class RealignmentAxionField(SingleAxionField):
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

realignment_axion_field = RealignmentAxionField()

