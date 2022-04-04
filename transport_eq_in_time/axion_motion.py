import numpy as np, matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import root
import decay_process

# temperature is in GeV
# axion time scale: 1 / m_a(T_osc)
def make_single_field_rhs(calc_pot_deriv):
    def rhs(t, y, H_fn, T_fn, m_a_osc, axion_parameter):
        theta, theta_dot = y
        H, T = H_fn(t), T_fn(t)
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

def solve(rhs, axion_init, calc_axion_mass, axion_parameter, tmax_axion_time, H_fn, T_fn, Gamma_phi, debug=False):
    m_a_osc, conv_factor = calc_axion_timescale(calc_axion_mass, axion_parameter, Gamma_phi)
    sol = solve_ivp(rhs, (0.0, tmax_axion_time), axion_init,
            args=(lambda t: H_fn(conv_factor * t + decay_process.t0), lambda t: T_fn(conv_factor * t + decay_process.t0),
                m_a_osc, axion_parameter),
            dense_output=True, rtol=1e-6, method="BDF")
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

def test(H_inf, Gamma_phi, m_a, tmax_axion_time=10.0):
    axion_parameter = (m_a,)
    calc_axion_mass = lambda T, m_a: m_a
    rho0 = 3*decay_process.M_pl**2*H_inf**2
    m_a_osc, conv_factor = calc_axion_timescale(calc_axion_mass, axion_parameter, Gamma_phi)
    sol_rh = decay_process.solve_decay_eqs(tmax_axion_time * conv_factor, 0.0, rho0, Gamma_phi)
    T_fn, H_fn = decay_process.to_temperature_and_hubble_fns(sol_rh, rho0, Gamma_phi, debug=True)
    sol_axion = solve(make_single_field_rhs(lambda theta, T, m_a: m_a**2*theta),
            (1.0, 0.0), calc_axion_mass, axion_parameter, tmax_axion_time, H_fn, T_fn, Gamma_phi, debug=True)
    plt.show()

