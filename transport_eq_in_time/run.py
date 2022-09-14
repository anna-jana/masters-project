import time, os, logging, importlib, pickle, functools, itertools, operator
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import h5py
import axion_motion, observables, clockwork_axion, generic_alp, transport_equation, util

############################ general code ##########################
nres = 6

def run_task(task):
    n, xs, f, nsteps = task
    logging.info(f"starting step %i of %i {xs}", n, nsteps)
    start = time.time()
    try:
        x = f(*xs)
    except Exception as e:
        x = [np.nan]*nres
        logging.error(f"step {n} raised an exception: {e}")
    end = time.time()
    logging.info(f"{n}th result: {x} (took {end - start} seconds)")
    return x

added_stderr_logger = False

def run(name, f, argnames, xss):
    start_time = time.time()
    if not os.path.exists(util.datadir):
        os.mkdir(util.datadir)
    i = 1
    while True:
        outputfile = os.path.join(util.datadir, f"{name}{i}.hdf5")
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

    tasks = zip(itertools.count(), input_data, itertools.repeat(f), itertools.repeat(nsteps))

    logging.info(f"starting computations for {name} ...")
    with ProcessPoolExecutor() as pool:
        data_list = list(pool.map(run_task, tasks))
    logging.info("... computations done")

    data = np.reshape(data_list, shape + (nres,))

    logging.info("storing data in hdf5 file %s", outputfile)
    with h5py.File(outputfile, "w") as fh:
        # save output
        eta = fh.create_dataset("eta", shape, dtype="d")
        dilution = fh.create_dataset("dilution", shape, dtype="d")
        rho_end_rad = fh.create_dataset("rho_end_rad", shape, dtype="d")
        rho_end_axion = fh.create_dataset("rho_end_axion", shape, dtype="d")
        Omega_h_sq = fh.create_dataset("Omega_h_sq", shape, dtype="d")
        status = fh.create_dataset("status", shape, dtype="i")
        eta[...] = data[..., 0]
        dilution[...] = data[..., 1]
        rho_end_rad[...] = data[..., 2]
        rho_end_axion[...] = data[..., 3]
        Omega_h_sq[...] = data[..., 4]
        status[...] = data[..., 5].astype("int")

        # save input
        for argname, xs in zip(argnames, xss):
            ds = fh.create_dataset(argname, (len(xs),), dtype="d")
            ds[...] = xs
    logging.info("storing output data done")

    end_time = time.time()
    logging.info(f"total time elapsed: {end_time - start_time}")
    logging.info("Terminating program.")

######################### realignment ########################
def f_generic_alp(H_inf, Gamma_inf, m_a, f_a, nsource_vector):
    source_vector = transport_equation.source_vectors[nsource_vector]
    return observables.compute_observables(H_inf, Gamma_inf, (m_a,), f_a,
                generic_alp.realignment_axion_field, (1.0, 0.0),
                calc_init_time=True, source_vector_axion=source_vector)

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
    return observables.compute_observables(H_inf, Gamma_inf, (eps, M), f,
                        clockwork_axion.clockwork_axion_field, ax0,
                        calc_init_time=True, isocurvature_check=False, 
                            source_vector_axion=source_vector)

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

###################################### main ################################
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
