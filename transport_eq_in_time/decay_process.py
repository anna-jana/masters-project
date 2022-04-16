import numpy as np, matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import root
t0 = 1.0
M_pl = 2.435e18 # reduced Planck mass [GeV] from wikipedia (https://en.wikipedia.org/wiki/Planck_mass)
g_star = 427/4 # during reheating from paper

def rhs(log_t, u, C, rho0):
    t = np.exp(log_t)
    rho_rad, a = u
    rho_field = rho0 * a**(-3) * np.exp(-(t - t0))
    H = np.sqrt(rho_field + rho_rad) * C
    rho_rad_dot = - 4 * H * rho_rad + rho_field
    a_dot = a * H
    return t * rho_rad_dot, t * a_dot

def find_scale(Gamma):
    # what is the energy density at T_decay?
    return 3 / 2 * M_pl**2 * Gamma**2


def solve(tmax, rho_rad_init, rho_field_init, scale, Gamma, debug=False):
    C = np.sqrt(scale) / (np.sqrt(3)*M_pl*Gamma)
    rho0 = rho_field_init / scale
    sol = solve_ivp(rhs, (np.log(t0), np.log(t0 + tmax)), (rho_rad_init / scale, 1.0),
            args=(C, rho0), rtol=1e-6, method="LSODA", dense_output=True)
    assert sol.success
    if debug:
        log_t = np.linspace(sol.t[0], sol.t[-1], 400)
        t_prime = np.exp(log_t)
        rho_rad, a = sol.sol(log_t)
        rho_field = rho0 * a**(-3) * np.exp(-(t_prime - t0))

        plt.figure()
        plt.subplot(2,1,1)
        plt.axvline(t0 + 1.0, color="black", ls="--", label="decay time")
        plt.loglog(t_prime, rho_field, label=r"field")
        plt.loglog(t_prime, rho_rad, label=r"radiation")
        plt.xlabel(r"$t \cdot \Gamma$")
        plt.ylabel(r"$\rho / \rho_phi(t_0)$")
        plt.ylim(1e-15, plt.ylim()[1])
        plt.legend(framealpha=1.0)

        plt.subplot(2,1,2)
        plt.loglog(t_prime, a, label="numerical rh")
        plt.xlabel(r"$t \cdot \Gamma$")
        plt.ylabel("a")
        plt.tight_layout()
    return sol

def find_end_rad_energy(sol, scale):
    rho_rad, a = sol.y[:, -1]
    return scale * rho_rad

def find_end_field_energy(sol, rho_field_init):
    rho_rad, a = sol.y[:, -1]
    return rho_field_init * a**(-3) * np.exp(- (np.exp(sol.t[-1]) - t0))

def T_to_t(T, T_and_H_fn, t_end):
    goal_fn = lambda t: T_and_H_fn(t)[0] / T - 1
    sol = root(goal_fn, t_end)
    return sol.x[0] if sol.success else np.nan

def to_temperature_and_hubble_fns(sol, rho_field_init, scale, Gamma, debug=False):
    T_const = (scale)**(1/4) / (np.pi**2 / 30 * g_star)**(1/4)
    H_const = np.sqrt(scale) / (np.sqrt(3) * M_pl)
    T_dot_const = scale / (np.pi**2/30*g_star * 4)

    def _helper(t_prime):
        rho_rad, a = sol.sol(np.log(t_prime))
        rho_field = (rho_field_init / scale) * a**(-3) * np.exp(-(t_prime - t0))
        T = T_const * rho_rad**(1/4)
        H = H_const * np.sqrt(rho_rad + rho_field)
        return rho_field, rho_rad, T, H

    def T_and_H_fn(t_prime):
        _, _, T, H = _helper(t_prime)
        return T, H

    def T_and_H_and_T_dot_fn(t_prime):
        rho_field, rho_rad, T, H = _helper(t_prime)
        T_dot = np.where(T == 0, np.inf, T_dot_const * (Gamma * rho_field - 4*H*rho_rad) / T**3)
        return T, H, T_dot

    if debug:
        t = np.exp(np.linspace(sol.t[0], sol.t[-1], 400))
        plt.figure()
        plt.subplot(2,1,1)
        T_RH = (45*M_pl**2/g_star)**(1/4) * np.sqrt(Gamma)
        T, H = T_and_H_fn(t)
        plt.loglog(t, T)
        plt.ylabel("T / GeV")
        plt.subplot(2,1,2)
        plt.loglog(t, H, label="reheating, numerical")
        H0 = T_and_H_fn(t0)[1]
        plt.loglog(t, 1.0 / (2*((t - t0) / Gamma) + 1/H0), label="radiation domination, analytical")
        plt.ylabel("H / GeV")
        plt.xlabel(r"$t \cdot \Gamma$")
        plt.legend()
        plt.tight_layout()

    return T_and_H_fn, T_and_H_and_T_dot_fn

def find_dilution_factor(sol, T_and_H_fn, t):
    T_s, _ = T_and_H_fn(np.exp(sol.t[0]))
    a_s = sol.sol(sol.t[0])[1]
    T_ad, _ = T_and_H_fn(t)
    a_ad = sol.sol(np.log(t))[1]
    return ((T_s * a_s) / (T_ad * a_ad))**3

