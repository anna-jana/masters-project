import time, os, logging
import functools, itertools, operator
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import h5py
import axion_motion, observables, clockwork_axion

############################ general code ##########################
def run_task(task):
    n, xs, args, kwargs, f, nsteps = task
    logging.info("starting step %i of %i", n, nsteps)
    start = time.time()
    try:
        x = f(*xs, *args, **kwargs)
    except Exception as e:
        nres = 4
        x = [np.nan]*nres
        logging.error(f"step {n} raised an exception: {e}")
    end = time.time()
    logging.info("%i took %f seconds", n, end - start)
    logging.info(f"{n}th result: {x}")
    return x

def run(name, f, xss, args, kwargs):
    i = 1
    while True:
        logfile = f"{name}_run{i}.log"
        if not os.path.exists(logfile):
            break
        i += 1
    outputfile = f"{name}_data{i}.hdf5"
    
    logging.basicConfig(filename=logfile, level=logging.DEBUG)
    
    shape = tuple(map(len, xss))
    nsteps = functools.reduce(operator.mul, shape)
    nres = 4
    neutral = [np.nan]*nres
        
    logging.info(f"starting computations for {name} ...")
    with ProcessPoolExecutor() as pool:
        input_data = itertools.product(*xss)
        tasks = zip(itertools.count(), input_data, itertools.repeat(args), 
                    itertools.repeat(kwargs), itertools.repeat(f), itertools.repeat(nsteps))
        data_list = list(pool.map(run_task, tasks))
    logging.info("... computations done")
    
    data = np.reshape(data_list, shape + (nres,))
    
    logging.info("storing data in hdf5 file %s", outputfile)
    with h5py.File(outputfile, "w") as fh:
        eta = fh.create_dataset("eta", shape, dtype="f")
        dilution = fh.create_dataset("dilution", shape, dtype="f")
        Omega_h_sq = fh.create_dataset("Omega_h_sq", shape, dtype="f")
        status = fh.create_dataset("status", shape, dtype="i")
        eta[...] = data[..., 0]
        dilution[...] = data[..., 1]
        Omega_h_sq[...] = data[..., 2]
        status[...] = data[..., 3].astype("int")
    logging.info("storing output data done")
    
    logging.info("Terminating program.")

######################### realignment ########################
def f_realignment(H_inf, Gamma_inf, m_a, f_a):
    return H_inf, Gamma_inf, m_a, 1.0
    return compute_observables(H_inf, Gamma_inf, (m_a,), f_a, 
                        axion_motion.realignment_axion_field, 
                        (1.0, 0.0),
                        calc_init_time=True)
    
def run_realignment(N=20):
    H_inf = []
    f_a = 4 * 10**np.arange(10, 15+1)
    Gamma_inf_list = np.geomspace(1e6, 1e10, N)
    m_a_list = np.geomspace(1e6, 1e10, N + 1)
    run("realignment", f_realignment, [H_inf_list, Gamma_inf_list, m_a_list, f_a_list], [], dict())

############################ clockwork ##########################
def f_clockwork(H_inf, Gamma_inf, m_a, f_a):
    return compute_observables(H_inf, Gamma_inf, (eps, M), f, clockwork_axion_field, ax0, 
                                calc_init_time=True)

def run_clockwork():
    m_phi_list = np.geomspace(1e-6, 1e6, N + 1) * 1e-9 # [GeV]
    mR_list = np.linspace(0, 15, N)
    sim_f_eff = f_eff = 1e13
    H_inf = 1e8
    Gamma_inf = H_inf
        run("realignment", f_realignment, [H_inf_list, Gamma_inf_list, m_a_list, f_a_list], [], dict())

############################### test ##########################
# testing the run code
def f_test(a, b, arg1, arg2, option1=1, option2=2):
    assert (arg1, arg2, option1, option2) == ("arg1", "arg2", "option1", "option2"), \
            (arg1, arg2, option1, option2)
    return a*b, a+b, min(a,b), 1  
  
def run_test():
    a_list = np.linspace(1, 10, 10)
    b_list = np.geomspace(100, 1000, 11)
    run("test", f_test, [a_list, b_list], ["arg1", "arg2"], dict(option1="option1", option2="option2"))
    
    
