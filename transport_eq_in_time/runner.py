import time, os, logging
import functools, itertools, operator
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import h5py
import axion_motion, observables, clockwork_axion

############################ general code ##########################
def run_task(task):
    n, xs, args, kwargs, f, nsteps = task
    logging.info(f"starting step %i of %i {xs}", n, nsteps)
    start = time.time()
    try:
        x = f(*xs, *args, **kwargs)
    except Exception as e:
        nres = 4
        x = [np.nan]*nres
        logging.error(f"step {n} raised an exception: {e}")
    end = time.time()
    logging.info(f"{n}th result: {x} (took {end - start} seconds)")
    return x

def run(name, f, argnames, xss, args, kwargs):
    start_time = time.time()
    i = 1
    while True:
        logfile = f"{name}{i}.log"
        if not os.path.exists(logfile):
            break
        i += 1
    outputfile = f"{name}{i}.hdf5"

    logging.basicConfig(filename=logfile, level=logging.DEBUG)
    logging.getLogger().addHandler(logging.StreamHandler()) # make the log messages appear both in the file and on stderr

    shape = tuple(map(len, xss))
    nsteps = functools.reduce(operator.mul, shape)
    nres = 4

    input_data = itertools.product(*xss)

    tasks = zip(itertools.count(), input_data, itertools.repeat(args),
                itertools.repeat(kwargs), itertools.repeat(f), itertools.repeat(nsteps))

    logging.info(f"starting computations for {name} ...")
    with ProcessPoolExecutor() as pool:
        data_list = list(pool.map(run_task, tasks))
    logging.info("... computations done")

    data = np.reshape(data_list, shape + (nres,))

    logging.info("storing data in hdf5 file %s", outputfile)
    with h5py.File(outputfile, "w") as fh:
        # save output
        eta = fh.create_dataset("eta", shape, dtype="f")
        dilution = fh.create_dataset("dilution", shape, dtype="f")
        Omega_h_sq = fh.create_dataset("Omega_h_sq", shape, dtype="f")
        status = fh.create_dataset("status", shape, dtype="i")
        eta[...] = data[..., 0]
        dilution[...] = data[..., 1]
        Omega_h_sq[...] = data[..., 2]
        status[...] = data[..., 3].astype("int")
        # save input
        for argname, xs in zip(argnames, xss):
            ds = fh.create_dataset(argname, xs.shape, dtype="f")
            ds[...] = xs
    logging.info("storing output data done")
    
    end_time = time.time()
    logging.info(f"total time elapsed: {end_time - start_time}")
    logging.info("Terminating program.")

######################### realignment ########################
def f_realignment(Gamma_inf, m_a, f_a):
    H_inf_max = f_a*2*np.pi*1e-5
    return observables.compute_observables(H_inf_max, Gamma_inf, (m_a,), f_a,  
                                           axion_motion.realignment_axion_field, (1.0, 0.0), calc_init_time=True)

def run_realignment(N=15):
    f_a_list = 4 * 10**np.linspace(10, 15, 4)
    Gamma_inf_list = np.geomspace(1e6, 1e10, N)
    m_a_list = np.geomspace(1e6, 1e10, N + 1)
    run("realignment", f_realignment, ["Gamma_inf", "m_a", "f_a"], [Gamma_inf_list, m_a_list, f_a_list], [], dict())


############################ clockwork ##########################
def f_clockwork(H_inf_over_Gamma_inf, Gamma_inf, mR, m_phi):
    H_inf = Gamma_inf * H_inf_over_Gamma_inf
    eps = clockwork_axion.calc_eps(mR)
    f_eff = 1e12 # arbitary value since only Omega depends on f_eff and it is ~ f^2
    f = clockwork_axion.calc_f(f_eff, eps)
    M = m_phi / eps
    theta_i = 3*np.pi/4 # as in the paper = sqrt(average value of theta^^2 over 0..2pi)
    ax0 = (clockwork_axion.theta_to_phi_over_f(theta_i, eps), 0.0)
    return observables.compute_observables(H_inf, Gamma_inf, (eps, M), f, 
                                           clockwork_axion.clockwork_axion_field, ax0, 
                                           calc_init_time=True, isocurvature_check=False)

def run_clockwork(N=20):
    m_phi_list = np.geomspace(1e-6, 1e6, N + 1) * 1e-9 # [GeV]
    mR_list = np.linspace(1, 15, N)
    Gamma_inf_list = np.geomspace(1e6, 1e10, N - 1)
    H_inf_over_Gamma_inf_list = np.geomspace(1, 1e4, 2)
    run("clockwork", f_clockwork, ["H_inf_over_Gamma_inf", "Gamma_inf", "mR", "m_phi"], [H_inf_over_Gamma_inf_list, Gamma_inf_list, mR_list, m_phi_list], [], dict())

###################################### loading data ########################################
def load_data(name, version):
    filename = f"{name}{version}.hdf5"
    with h5py.File(filename, "r") as fh:
        data = {key : fh[key][...] for key in fh}
    return data
