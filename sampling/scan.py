from concurrent.futures import ProcessPoolExecutor
import sys
import time
sys.path.append("..")

import numpy as np
from func_timeout import func_timeout, FunctionTimedOut # func-timeout package

from common import util, constraints
from two_field_sbg import model

runtime = 24 * 60 * 60
timeout = 2 * 60
num_workers = 20

if len(sys.argv) > 2:
    H_inf = float(sys.argv[2])
else:
    H_inf = 1e9
m_chi = 1e-2

f_a_min = constraints.calc_f_a_min(H_inf)
m_a = constraints.minimal_axion_mass_from_decay(f_a_min)

theta_i_range = np.geomspace(1e-3, 1, 11)
chi0_range = np.logspace(-3, 2, 11) * H_inf
g_range = np.logspace(-5, -3, 3)
Gamma_phi_range = np.logspace(-5, 1, 4) * H_inf

inputs = []
for theta_i in theta_i_range:
    a0 = f_a_min * theta_i
    for g in g_range:
        for chi0 in chi0_range:
            for Gamma_phi in Gamma_phi_range:
                params = (a0, g, chi0, Gamma_phi)
                inputs.append(params)

def do(params):
    (a0, g, chi0, Gamma_phi) = params
    ans = model.compute_B_asymmetry(
            m_a, a0, Gamma_phi,
            H_inf, chi0, m_chi, g=g,
            bg_kwargs=dict(theta0=1),
            do_decay=False)
    return ans

def do_with_timeout(n):
    try:
        start = time.time()
        ans = func_timeout(timeout, do, args=(n,), kwargs={})
        end = time.time()
        return ans, (end - start) / timeout
    except FunctionTimedOut:
        return np.nan, 1.0

if len(sys.argv) > 1:
    output_filename = sys.argv[1]
else:
    output_filename = "scan.pkl"

with ProcessPoolExecutor(max_workers=num_workers) as pool:
    outputs = list(pool.map(do_with_timeout, inputs))
util.save_data(output_filename, H_inf, inputs, outputs)
