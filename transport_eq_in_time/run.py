import time, os, logging, importlib, pickle, functools, itertools, operator
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import axion_motion, observables, clockwork_axion, generic_alp, transport_equation, util

############################ general code ##########################

def run_task(task):
    n, xs, f, nsteps, nres = task
    logging.info(f"starting step %i of %i {xs}", n, nsteps)
    start = time.time()
    try:
        x = f(*xs)
    except Exception as e:
        x = [np.nan]*nres
        logging.error(f"step {n} raised an exception: {e}")
    assert len(x) == nres
    end = time.time()
    logging.info(f"{n}th result: {x} (took {end - start} seconds)")
    return x

added_stderr_logger = False

if not os.path.exists(util.datadir):
    os.mkdir(util.datadir)

def run(name, f, argnames, xss, nres):
    start_time = time.time()
    i = 1
    while True:
        outputfile = os.path.join(util.datadir, f"{name}{i}.pkl")
        if not os.path.exists(outputfile):
            break
        i += 1
    logfile = os.path.join(util.datadir, "log_file")

    logging.basicConfig(filename=logfile, level=logging.DEBUG)
    # make the log messages appear both in the file and on stderr but only once!
    global added_stderr_logger
    if not added_stderr_logger:
        logging.getLogger().addHandler(logging.StreamHandler())
        added_stderr_logger = True

    shape = tuple(map(len, xss))
    nsteps = functools.reduce(operator.mul, shape)
    input_data = itertools.product(*xss)
    tasks = zip(itertools.count(), input_data, itertools.repeat(f),
            itertools.repeat(nsteps), itertools.repeat(nres))

    logging.info(f"starting computations for {name} ...")
    with ProcessPoolExecutor() as pool:
        data_list = list(pool.map(run_task, tasks))
    logging.info("... computations done")

    logging.info("storing data in file %s", outputfile)
    data = dict(output=np.reshape(data_list, shape + (nres,)))
    data.update(dict(args=argnames))
    data.update({name : xs for name, xs in zip(argnames, xss)})
    util.save_pkl(data, outputfile)
    logging.info("storing output data done")

    end_time = time.time()
    logging.info(f"total time elapsed: {end_time - start_time}")
    logging.info("Terminating program.")

######################### realignment ########################
def f_generic_alp(H_inf, Gamma_inf, m_a, f_a, nsource_vector):
    source_vector = transport_equation.source_vectors[nsource_vector]
    model = observables.Model.make(H_inf, Gamma_inf,
            generic_al.realignment_axion_field, (m_a,), source_vector, f_a)
    asym_config = observables.AsymmetrySolverConfig(calc_init_time=True)
    status, state, eta_B = observables.compute_asymmetry(model, asym_config)
    f = observables.compute_dilution_factor_from_axion_decay(model, state, asym_config)
    rho_end_axion = model.axion_model.get_energy(state.axion, model.f_a, model.axion_parameter)
    return eta_B, f, status.rho_rad, rho_end_axion, float(status.value)

f_a = 4 * 1e15
H_inf_max = f_a*2*np.pi*1e-5 / 10

def run_generic_alp(nsource_vector=0, m_a_min=1e6, Gamma_inf_min=1e6):
    N = 30
    Gamma_inf_list = np.geomspace(Gamma_inf_min, H_inf_max, N)
    m_a_list = np.geomspace(m_a_min, H_inf_max, N)
    run("generic_alp", f_generic_alp, ["H_inf", "Gamma_inf", "m_a", "f_a", "nsource_vector"],
        [[H_inf_max], Gamma_inf_list, m_a_list, [f_a], [nsource_vector]])

############################ clockwork ##########################
def f_clockwork(H_inf, Gamma_inf, mR, m_phi, nsource_vector):
    source_vector = transport_equation.source_vectors[nsource_vector]
    eps = clockwork_axion.calc_eps(mR)
    f = clockwork_axion.calc_f(clockwork_axion.default_f_eff, eps)
    M = m_phi / eps
    theta_i = 3*np.pi/4 # as in the paper = sqrt(average value of theta^^2 over 0..2pi)
    ax0 = (clockwork_axion.theta_to_phi_over_f(theta_i, eps), 0.0)
    model = observables.Model.make(H_inf, Gamma_inf, clockwork_axion.clockwork_axion_field,
            (eps, M), source_vector, f)
    status, state, eta_B = observables.compute_asymmetry(model,
            observables.AsymmetrySolverConfig(calc_init_time=True, isocurvature_check=False))
    if status != observables.Status.OK:
        return eta_B, np.nan, float(status.values)
    status, Omega_h_sq = observables.compute_relic_density_from_state(
            mode, state, observables.RelicDensitySolverConfig())
    return eta_B, Omega_h_sq, float(status.value)

def run_cw_mR_vs_mphi(nsource_vector=0):
    Gamma_inf_list = np.array([1e-4, 1e-2, 1]) * H_inf_max
    N = 55
    m_phi_list = np.geomspace(1e-6, 1e8, N) * 1e-9 # [GeV]
    mR_list = np.linspace(1, 15, N)
    run("clockwork_mR_vs_mphi", f_clockwork, ["H_inf", "Gamma_inf", "mR", "m_phi", "nsource_vector"],
        [[H_inf_max], Gamma_inf_list, mR_list, m_phi_list, [0, 1, 2]])

def run_cw_Gammainf_vs_mphi():
    mR_list = [8, 11, 14]
    N = 35
    m_phi_list = np.geomspace(1e-6, 1e6, N) * 1e-9 # [GeV]
    Gamma_inf_list = np.geomspace(1e-5 * H_inf_max, H_inf_max, N)
    run("clockwork_Gammainf_vs_mphi", f_clockwork, ["H_inf", "Gamma_inf", "mR", "m_phi", "nsource_vector"],
        [[H_inf_max], Gamma_inf_list, mR_list, m_phi_list, [0, 1, 2]])


################################## main ###################################
if __name__ == "__main__":
    run_generic_alp(0)
    run_generic_alp(1, m_a_min=1e5)
    run_generic_alp(2)
    for n in range(3):
        generic_alp.compute_correct_curves(n + 1)
    generic_alp.recompute_all_dilutions()
    generic_alp.compute_all_example_trajectories()
    run_cw_mR_vs_mphi(n)
    run_cw_Gammainf_vs_mphi()
    clockwork_axion.compute_all_example_trajectories()
