#!/usr/bin/env python
# coding: utf-8

import sys
if ".." not in sys.path: sys.path.append("..")
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as si
from common import constraints

def rhs(t, y, m_a, m_chi, g, g_a, g_chi):
    H = 1 / (2*t)
    a, a_dot, chi, chi_dot = y
    return (
        a_dot,
        - 3 * H * a_dot - m_a**2 * a - g * a * chi**2 - g_a * a**3,
        chi_dot,
        - 3 * H * chi_dot - m_chi**2 * chi - g * chi * a**2 - g_chi * chi**3,
    )

def solve(f_a, chi0, g, g_a, g_chi, span, m_a, m_chi):
    H_inf = max(constraints.calc_H_inf_max(f_a), constraints.calc_H_inf_max(chi0))
    t0 = 1 / H_inf
    steps = np.geomspace(t0, span * t0, 1000)
    steps[0] = t0
    steps[-1] = span * t0
    return si.solve_ivp(rhs, (t0, span * t0), (f_a, 0.0, chi0, 0.0),
                        args=(m_a, m_chi, g, g_a, g_chi), t_eval=steps)


f_a = 1e10
chi0 = 1e9
g = 1e-3
g_a = 0
g_chi = 0
m_chi = 1e-2
m_a = 1e-2
span = 1e2
sol_pert = solve(f_a, chi0 + 1e5, g, g_a, g_chi, span, m_a, m_chi)
sol = solve(f_a, chi0, g, g_a, g_chi, span, m_a, m_chi)

plt.figure()
plt.semilogx(sol.t, sol.y[0] / f_a, ls="-", color="tab:blue", label="a / a0")
plt.semilogx(sol.t, sol.y[2] / chi0, ls="-", color="tab:orange", label="chi / chi0")
plt.semilogx(sol_pert.t, sol_pert.y[0] / f_a, ls="--", color="tab:blue", label="a / a0 perturbed")
plt.semilogx(sol_pert.t, sol_pert.y[2] / chi0, ls="--", color="tab:orange", label="chi / chi0 perturbed")
plt.legend()
plt.xlabel("t * GeV")

a_range = np.linspace(-0.8 * f_a, 1.1 * f_a, 100)
chi_range = np.linspace(-6 * chi0, 9.5 * chi0, 100)
aa, cc = np.meshgrid(a_range, chi_range)
V = 0.5 * m_a**2 * aa**2 + 0.5 * m_chi**2 * cc**2 + g * aa**2 * cc**2
plt.figure()
plt.contourf(a_range / f_a, chi_range / chi0, np.log10(V), cmap="summer")
plt.colorbar().set_label("log_10 (V / GeV^4)")
plt.plot(sol.y[0] / f_a, sol.y[2] / chi0, label="unpert.")
plt.plot(sol_pert.y[0] / f_a, sol.y[2] / chi0, label="pert.")
plt.legend()
plt.xlabel("a / a0")
plt.ylabel("chi / chi0")

# fit the total energy power law
chi_energy = 0.5 * sol.y[3]**2 + 0.5 * m_chi**2 * sol.y[2]**2 + g * sol.y[0]**2 * sol.y[2]**2 / 2
a_energy = 0.5 * sol.y[1]**2 + 0.5 * m_a**2 * sol.y[0]**2 + g * sol.y[0]**2 * sol.y[2]**2 / 2
total = chi_energy + a_energy
p, b = np.polyfit(np.log(sol.t), np.log(total), 1)
plt.figure()
plt.loglog(sol.t, total)
plt.loglog(sol.t, np.exp(p*np.log(sol.t) + b))
plt.xlabel("t * GeV")
plt.ylabel("total energy density")
plt.title(f"rho ~ t^{p:.2} ~ a^{2*p:.2}, w = {-2/3*p - 1:.2}")

# case in which the chion field is frozen and the axion oscillates
f_a = 1e9
chi0 = 1e10
m_chi = 1e-2
m_a = 1e-2

g = 0 # 1e-3
g_a = 1e-3
g_chi = 0

span = 1e3
sol = solve(f_a, chi0, g, g_a, g_chi, span, m_a, m_chi)
a = sol.y[0]; chi = sol.y[2]
plt.figure()
plt.semilogx(sol.t, a, label="axion")
plt.semilogx(sol.t, chi, label="chion")
plt.xlabel("t")
plt.legend()

a_range = np.linspace(np.min(a), np.max(a), 100)
chi_range = np.linspace(np.min(chi), np.max(chi), 100)
aa, cc = np.meshgrid(a_range, chi_range)
V = 0.5 * m_a**2 * aa**2 + 0.5 * m_chi**2 * cc**2 + g * aa**2 * cc**2 + g_a * aa**4 + g_chi * cc**4
plt.figure()
plt.contourf(a_range / f_a, chi_range / chi0, np.log10(V), cmap="summer")
plt.colorbar().set_label("log_10 (V / GeV^4)")
plt.plot(sol.y[0] / f_a, sol.y[2] / chi0)
plt.xlabel("a / a0")
plt.ylabel("chi / chi0");
plt.show()


