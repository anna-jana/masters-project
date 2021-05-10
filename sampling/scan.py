from concurrent.futures import ProcessPoolExecutor
import sys
sys.path.append("..")

import numpy as np
from func_timeout import func_timeout, FunctionTimedOut # func-timeout package

from common import util, constraints
from two_field_sbg import model

runtime = 24 * 60 * 60
timeout = 2 * 60 # 5 minutes
num_workers = 20

H_inf = 1e9
m_chi = 1e-2

theta_i = 1.0
f_a_min = constraints.calc_f_a_min(H_inf)
a0_range = np.geomspace(1e-3, 1e3, 12) * f_a_min
g_range = np.logspace(-5, -2, 3)
chi0_range = np.logspace(-4, 1, 10) * H_inf
Gamma_phi_range = np.logspace(-5, 1, 6) * H_inf

inputs = []
for a0 in a0_range:
    m_a = constraints.minimal_axion_mass_from_decay(a0 / theta_i)
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
            bg_kwargs=dict(theta0=theta_i))

def do_with_timeout(n):
    try:
        ans = func_timeout(timeout, do, args=(n,), kwargs={})
        return ans
    except FunctionTimedOut:
        return np.nan

output_filename = "scan.pkl"

if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        outputs = list(pool.map(do_with_timeout, inputs))
    util.save_data(output_filename, inputs, outputs)
