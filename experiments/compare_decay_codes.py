# new: 1.19 s ± 25.7 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
# old: 1.19 s ± 6.61 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

import old_decay_process, new_decay_process
old_decay_process = importlib.reload(old_decay_process)
new_decay_process = importlib.reload(new_decay_process)

H0 = 1e10
rho_inf = 3*old_decay_process.M_pl**2*H0**2
Gamma = 1e8
tmax = 10.0
sol_old = old_decay_process.solve(tmax, 0.0, rho_inf, Gamma)
sol_new = new_decay_process.solve(tmax, 0.0, rho_inf, Gamma)
old_fn, _ = old_decay_process.to_temperature_and_hubble_fns(sol_old, rho_inf, Gamma)
new_fn, _ = new_decay_process.to_temperature_and_hubble_fns(sol_new, rho_inf, Gamma)
ts = np.geomspace(decay_process.t0, tmax, 400)
T_old, H_old = old_fn(ts)
T_new, H_new = new_fn(ts)
plt.figure()
plt.plot(ts, H_old, label="old")
plt.plot(ts, H_new, label="new")
plt.xlabel("t * Gamma")
plt.ylabel("H")
plt.xscale("log")
plt.yscale("log")
plt.figure()
plt.plot(ts, T_old, label="old")
plt.plot(ts, T_new, label="new")
plt.xlabel("t * Gamma")
plt.ylabel("T")
plt.xscale("log")
plt.yscale("log")

m_a = 1e7
ax_sol = axion_motion.single_axion_field.solve((1.0, 0.0), (m_a,), 10.0, old_fn, Gamma)

f_a = 1e12
rho_a = axion_motion.single_axion_field.get_energy(ax_sol.y[:, -1], f_a, Gamma, m_a)
rho_rad_end = decay_process.find_end_rad_energy(sol_old, rho_inf)
Gamma_a = axion_motion.single_axion_field.get_decay_constant(f_a, m_a)

tmax = 100.0
sol_old = old_decay_process.solve(tmax, rho_rad_end, rho_a, Gamma)
sol_new = new_decay_process.solve(tmax, rho_rad_end, rho_a, Gamma)
old_fn, _ = old_decay_process.to_temperature_and_hubble_fns(sol_old, rho_a, Gamma)
new_fn, _ = new_decay_process.to_temperature_and_hubble_fns(sol_new, rho_a, Gamma)
ts = np.geomspace(decay_process.t0, tmax, 400)
T_old, H_old = old_fn(ts)
T_new, H_new = new_fn(ts)
plt.figure()
plt.plot(ts, H_old, label="old")
plt.plot(ts, H_new, label="new")
plt.xlabel("t * Gamma")
plt.ylabel("H")
plt.xscale("log")
plt.yscale("log")
plt.figure()
plt.plot(ts, T_old, label="old")
plt.plot(ts, T_new, label="new")
plt.xlabel("t * Gamma")
plt.ylabel("T")
plt.xscale("log")
plt.yscale("log")
old_decay_process.find_dilution_factor(sol_old, old_fn, debug=True)
new_decay_process.find_dilution_factor(sol_new, new_fn, debug=True)

plt.show()
