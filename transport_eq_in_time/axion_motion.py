import importlib
import numpy as np, matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import root
import decay_process
decay_process = importlib.reload(decay_process)

# temperature is in GeV
# axion time scale: 1 / m_a(T_osc)
def make_single_field_rhs(calc_pot_deriv):
    def rhs(t, y, T_and_H_fn, m_a_osc, axion_parameter):
        theta, theta_dot = y
        T, H = T_and_H_fn(t) #T_and_H_fn has been transformed to take axion time
        return (theta_dot, - 3*H/m_a_osc*theta_dot - calc_pot_deriv(theta, T, *axion_parameter) / m_a_osc**2)
    return rhs

K = np.pi**2/30 * decay_process.g_star

def calc_T_osc(calc_axion_mass, axion_parameter, N=2):
    m_a_0 = calc_axion_mass(0, *axion_parameter)
    # we assume raditation domination, since this is only an initial guess it will not invalidate the final result if its
    # not perfectly correct
    T_osc_initial_guess = (3 * decay_process.M_pl**2 * (m_a_0/N)**2 / K)**(1/4)
    goal_fn = lambda T: np.log(calc_axion_mass(T, *axion_parameter) / (N * np.sqrt(K)*T**2/(np.sqrt(3)*decay_process.M_pl)))
    sol = root(goal_fn, T_osc_initial_guess)
    assert sol.success
    return sol.x[0]

def calc_axion_timescale(calc_axion_mass, axion_parameter, Gamma_phi):
    T_osc = calc_T_osc(calc_axion_mass, axion_parameter)
    m_a_osc = calc_axion_mass(T_osc, *axion_parameter)
    conv_factor = Gamma_phi / m_a_osc
    return m_a_osc, conv_factor

def solve(rhs, axion_init, calc_axion_mass, axion_parameter, tmax_axion_time, T_and_H_fn, Gamma_phi, debug=False):
    m_a_osc, conv_factor = calc_axion_timescale(calc_axion_mass, axion_parameter, Gamma_phi)
    sol = solve_ivp(rhs, (0.0, tmax_axion_time), axion_init,
            args=(lambda t: T_and_H_fn(conv_factor * t + decay_process.t0), m_a_osc, axion_parameter),
            dense_output=True, rtol=1e-6, method="LSODA")
    assert sol.success
    if debug:
        plt.figure()
        plt.axvline(1.0, color="black", ls="--")
        plt.axhline(0.0, color="black", ls="-")
        t = np.linspace(0.0, sol.t[-1], 400)
        plt.plot(t, sol.sol(t)[0,:])
        plt.xscale("log")
        plt.xlabel(r"$t \cdot m_a(T_\mathrm{osc})$")
        plt.ylabel(r"$\theta$")
    return sol

def get_axion_source_single_field(sol, conv_factor):
    def source(t_inf):
        theta, theta_dot = sol.sol((t_inf - decay_process.t0) / conv_factor)
        return theta_dot / conv_factor
    return source
