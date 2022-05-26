import time, os, logging
import functools, itertools, operator
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import h5py
import axion_motion, observables, clockwork_axion

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
    i = 1
    while True:
        logfile = f"{name}{i}.log"
        if not os.path.exists(logfile):
            break
        i += 1
    outputfile = f"{name}{i}.hdf5"

    logging.basicConfig(filename=logfile, level=logging.DEBUG)
    # make the log messages appear both in the file and on stderr but only once!
    global added_stderr_logger
    if not added_stderr_logger:
        logging.getLogger().addHandler(logging.StreamHandler()) 
        added_stderr_logger = False

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
def f_realignment(H_inf, Gamma_inf, m_a, f_a):
    return observables.compute_observables(H_inf, Gamma_inf, (m_a,), f_a,
                axion_motion.realignment_axion_field, (1.0, 0.0), calc_init_time=True)

def run_realignment():
    f_a = 4 * 1e15
    N = 30
    H_inf_max = f_a*2*np.pi*1e-5
    Gamma_inf_list = np.geomspace(1e6, H_inf_max, N)
    m_a_list = np.geomspace(1e6, H_inf_max, N)
    run("realignment", f_realignment, ["H_inf", "Gamma_inf", "m_a", "f_a"],
        [[H_inf_max], Gamma_inf_list, m_a_list, [f_a]])


############################ clockwork ##########################
def f_clockwork(H_inf, Gamma_inf, mR, m_phi):
    eps = clockwork_axion.calc_eps(mR)
    f_eff = 1e12 # arbitary value since only Omega depends on f_eff and it is ~ f^2
    f = clockwork_axion.calc_f(f_eff, eps)
    M = m_phi / eps
    theta_i = 3*np.pi/4 # as in the paper = sqrt(average value of theta^^2 over 0..2pi)
    ax0 = (clockwork_axion.theta_to_phi_over_f(theta_i, eps), 0.0)
    return observables.compute_observables(H_inf, Gamma_inf, (eps, M), f,
                        clockwork_axion.clockwork_axion_field, ax0,
                        calc_init_time=True, isocurvature_check=False)

def run_cw_mR_vs_mphi():
    H_inf = Gamma_inf = 1e8
    N = 50
    m_phi_list = np.geomspace(1e-6, 1e6, N + 1) * 1e-9 # [GeV]
    mR_list = np.linspace(1, 15, N)
    run("clockwork_mR_vs_mphi", f_clockwork, ["H_inf", "Gamma_inf", "mR", "m_phi"],
        [[H_inf], [Gamma_inf], mR_list, m_phi_list])

def run_cw_Gammainf_vs_mphi():
    H_inf = 1e10
    mR_list = [10]
    N = 30
    m_phi_list = np.geomspace(1e-6, 1e2, N) * 1e-9 # [GeV]
    Gamma_inf_list = np.geomspace(1e-5 * H_inf, H_inf, N)
    run("clockwork_Gammainf_vs_mphi", f_clockwork, ["H_inf", "Gamma_inf", "mR", "m_phi"],
        [[H_inf], Gamma_inf_list, mR_list, m_phi_list])


###################################### loading data ########################################
def load_data(name, version):
    filename = f"{name}{version}.hdf5"
    with h5py.File(filename, "r") as fh:
        data = {key : fh[key][...] for key in fh}
    return data
