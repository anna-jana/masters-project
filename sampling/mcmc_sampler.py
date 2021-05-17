import sys
import os
import time
import numpy as np
import emcee
from schwimmbad import MPIPool
from collections import namedtuple
from func_timeout import func_timeout, FunctionTimedOut # func-timeout package

sys.path.append("../")
from two_field_sbg.model import compute_B_asymmetry
from common import constraints

os.environ["OMP_NUM_THREADS"] = "1"

output_file = sys.argv[1]
restart = False
# TODO: tweak
nwalkers = 100
nsteps = 1000
# ndim = 8 # fix m_chi for now
ndim = 7

m_chi_fixed = 1e-2

Parameters = namedtuple("Parameters", ["log10_m_a", "log10_f_a", "log10_Gamma_phi", "log10_H_inf", "log10_chi0",
    # "log10_m_chi",
    "log10_g", "theta0"])
ConvertedParameters = namedtuple("ConvertedParameters", ["m_a", "f_a", "Gamma_phi", "H_inf", "chi0",
    # "m_chi",
    "g", "theta0"])

def convert_parameters(parameters):
    return ConvertedParameters(m_a=10**parameters.log10_m_a, f_a=10**parameters.log10_f_a, Gamma_phi=10**parameters.log10_Gamma_phi,
            H_inf=10**parameters.log10_H_inf, chi0=10**parameters.log10_chi0,
            # m_chi=10**parameters.log10_m_chi,
            g=10**parameters.log10_g, theta0=parameters.theta0)

# all priors

# the likelihood
eta_B_obs = 6e-10
eta_B_err = 1e-10 # TODO: fix

def gaussian_ln_prop(val, mean, std):
    return - (val - mean)**2 / (2* std**2)

timeout = 60 * 2

def ln_likelihood(parameters):
    p = Parameters(*parameters)
    cp = convert_parameters(p)
    try:
        eta_B_final = func_timeout(timeout, compute_B_asymmetry,
                kwargs=dict(
                    m_a=cp.m_a, f_a=cp.f_a,
                    Gamma_phi=cp.Gamma_phi, H_inf=cp.H_inf,
                    chi0=cp.chi0,
                    m_chi=m_chi_fixed, # cp.m_chi,
                    g=cp.g,
                    bg_kwargs=dict(theta0=cp.theta0)))
    except FunctionTimedOut:
        return - np.inf # ignore points, where the code takes more than timeout seconds to run
    return gaussian_ln_prop(eta_B_final, eta_B_obs, eta_B_err)

log10_f_a_max = 16
log10_Gamma_phi_min = 1
log10_H_inf_min, log10_H_inf_max = 5, 12
log10_chi0_min, log10_chi0_max = 8, 12
# log10_m_chi_min =  # we ignore m_chi for now
log10_g_min, log10_g_max = -4, -2
theta0_min, theta0_max = 0, 3

def ln_prior(parameters):
    print("running:", parameters)
    p = Parameters(*parameters)
    in_bounds = (
        # range of m_a fixed by over parameters
                               p.log10_f_a       <= log10_f_a_max       and # ok
        log10_Gamma_phi_min <= p.log10_Gamma_phi                        and # ok
        log10_H_inf_min     <= p.log10_H_inf     <= log10_H_inf_max     and # ok
        log10_chi0_min      <= p.log10_chi0      <= log10_chi0_max      and # ok
        # log10_m_chi_min     <= p.log10_m_chi                            and # ok
        log10_g_min         <= p.log10_g         <= log10_g_max         and # ok
        theta0_min          <= p.theta0          <= theta0_max              # ok
    )
    apriori_constrains_okay = (
        cp.Gamma_phi < cp.H_inf and
        cp.m_a < cp.H_inf and
        cp.m_chi < cp.H_inf and
        cp.H_inf < constraints.calc_H_inf_max(cp.f_a) and
        m_a > constraints.minimal_axion_mass_from_decay(cp.f_a)
    )
    if in_bounds and apriori_constrains_okay:
        return 0.0
    else:
        return - np.inf

def make_initial():
    log10_H_inf = np.random.uniform(log10_H_inf_min, log10_H_inf_max)
    log10_f_a = np.random.uniform(np.log10(constraints.calc_f_a_min(10**log10_H_inf)),
                                  log10_f_a_max)

    return tuple(Parameters(
        log10_m_a=np.random.uniform(
            np.log10(constraints.minimal_axion_mass_from_decay(10**log10_f_a)),
            log10_H_inf),
        theta0=np.random.uniform(theta0_min, theta0_max),
        log10_f_a=log10_f_a,

        log10_H_inf=log10_H_inf,
        log10_Gamma_phi=np.random.uniform(log10_Gamma_phi_min, log10_H_inf),

        log10_chi0=np.random.uniform(log10_chi0_min, log10_chi0_max),
        # log10_m_chi=np.random.uniform(log10_m_chi_min, log10_H_inf),

        log10_g=np.random.uniform(log10_g_min, log10_g_max),
    ))

def ln_prob(parameters):
    return ln_likelihood(parameters) + ln_prior(parameters)

# create MPI pool to run things on
with MPIPool() as pool:
    if not pool.is_master():
        pool.wait()
        sys.exit(0)

    backend = emcee.backends.HDFBackend(output_file)

    if not restart:
        # create initial configurations for each walker
        backend.reset(nwalkers, ndim)
        initial = np.array([make_initial() for i in range(nwalkers)])

    # create sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, ln_prob, backend=backend, pool=pool)

    # run sampler
    if restart:
        sampler.run_mcmc(None, nsteps, progress=False)
    else:
        sampler.run_mcmc(initial, nsteps, progress=False)
