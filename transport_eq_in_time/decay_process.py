import numpy as np, matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

t0 = 1.0
M_pl = 2.435e18 # reduced Planck mass [GeV] from wikipedia (https://en.wikipedia.org/wiki/Planck_mass)
g_star = 427/4 # during reheating from paper
# energy density unit: init_rho_field, a0 = 1.0, t0 = 0.0, time unit: 1 / decay_const
# trange is in units of 1 / decay_const, init_rho_field, init_rho_rad are in GeV^4, decay_const in GeV

def decay_rhs(log_t, u, C):
    a, y = np.exp(u)
    t = np.exp(log_t)
    y_dot = (a - 1) * np.exp(-(t - t0))
    a_dot = C / a * np.sqrt(y + y_dot)
    return (t / a * a_dot, t / y * y_dot)

def solve_decay_eqs(trange, init_rho_rad, init_rho_field, decay_const, debug=False):
    init_y = (init_rho_rad + init_rho_field) / init_rho_field
    C = np.sqrt(init_rho_field) / (np.sqrt(3) * M_pl * decay_const)
    if debug:
        print("C =", C)
    tspan = (np.log(t0), np.log(t0 + trange))
    sol = solve_ivp(decay_rhs, tspan, (0.0, np.log(init_y)), args=(C,), dense_output=True, method="BDF")
    assert sol.success
    if debug:
        log_t = np.linspace(*tspan, 400)
        t_prime = np.exp(log_t)
        a, y = np.exp(sol.sol(log_t))
        x = np.exp(- (t_prime - t0))
        rho_a_prime = a**(-3) * x
        rho_R_prime = a**(-4) * (y - x)
        plt.figure()
        plt.subplot(2,1,1)
        plt.axvline(t0 + 1.0, color="black", ls="--", label="decay time")
        plt.loglog(t_prime, rho_a_prime, label=r"field")
        plt.loglog(t_prime, rho_R_prime, label=r"radiation")
        plt.xlabel(r"$t \cdot \Gamma_a$")
        plt.ylabel(r"$\rho / \rho_a(t_0)$")
        plt.legend(framealpha=1.0)
        plt.subplot(2,1,2)
        plt.loglog(t_prime, a)
        plt.xlabel(r"$t \cdot \Gamma_a$")
        plt.ylabel("a")
        plt.tight_layout()
    return sol

def to_temperature_and_hubble_fns(sol, rho0, decay_const, debug=False):
    T_const = (rho0 / np.pi**2 / 30 * g_star)**(1/4)
    def T_fn(t_prime):
        a, y = np.exp(sol.sol(np.log(t_prime)))
        x = np.exp(- (t_prime - t0))
        rho_R_prime = a**(-4) * (y - x)
        return T_const * rho_R_prime**(1/4)
    H_const = np.sqrt(rho0) / (np.sqrt(3) * M_pl)
    def H_fn(t_prime):
        a, y = np.exp(sol.sol(np.log(t_prime)))
        x = np.exp(- (t_prime - t0))
        rho_a_prime = a**(-3) * x
        rho_R_prime = a**(-4) * (y - x)
        return H_const * np.sqrt(rho_R_prime + rho_a_prime)
    if debug:
        t = np.exp(np.linspace(sol.t[0], sol.t[-1], 400))
        plt.figure()
        plt.subplot(2,1,1)
        reheating_const = (45*M_pl**2/g_star)**(1/4)
        T_RH = reheating_const * np.sqrt(decay_const)
        plt.loglog(t, T_fn(t))
        plt.ylabel("T / GeV")
        plt.subplot(2,1,2)
        plt.loglog(t, H_fn(t), label="numerical")
        plt.loglog(t, 1.0 / (2*((t - t0) / decay_const) + 1/H_fn(t0)), label="rad. dom.")
        plt.ylabel("H / GeV")
        plt.xlabel(r"$t \cdot \Gamma$")
        plt.tight_layout()
    return T_fn, H_fn

def find_dilution_factor(sol, T_fn, debug=False):
    if debug:
        logt = np.linspace(sol.t[0], sol.t[-1], 400)
        a = np.exp(sol.sol(logt)[0, :])
        t = np.exp(logt)
        T = T_fn(t)
        dilution_factor = (T[0] * a[0] / (T * a))**3
        plt.figure()
        plt.axhline(dilution_factor[0], color="black", ls="--")
        plt.axhline(dilution_factor[-1], color="black", ls="--")
        plt.loglog(t, dilution_factor)
        plt.xlabel(r"$t \cdot \Gamma$")
        plt.ylabel(r"dilution factor $(T(t_0) a(t_0) / T(t) a(t))^3$")
        return dilution_factor[-1]
    else:
        T_s = T_fn(np.exp(sol.t[0]))
        a_s = np.exp(sol.sol(sol.t[0])[0])
        T_ad = T_fn(np.exp(sol.t[-1]))
        a_ad = np.exp(sol.sol(sol.t[-1])[0])
        return ((T_s * a_s) / (T_ad * a_ad))**3

def test(H0, Gamma_phi, time=10.0):
    rho0 = 3*M_pl**2*H0**2
    sol = solve_decay_eqs(time, 0.0, rho0, Gamma_phi, debug=True)
    T_fn, H_fn = to_temperature_and_hubble_fns(sol, rho0, Gamma_phi, debug=True)
    # find_dilution_factor(sol, T_fn, debug=True)
    plt.show()
