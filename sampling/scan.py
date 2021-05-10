from concurrent.futures import ProcessPoolExecutor
import sys
import time
import argparse
sys.path.append("..")

import numpy as np
from func_timeout import func_timeout, FunctionTimedOut # func-timeout package

from common import util, constraints
from two_field_sbg import model

runtime = 24 * 60 * 60
timeout = 2 * 60
num_workers = 20

m_chi = 1e-2

parser = argparse.ArgumentParser()

parser.add_argument("--H_inf", dest="H_inf", type=float, default=1e9)

parser.add_argument("--min-theta_i", dest="min_theta_i", type=float, default=1e-3)
parser.add_argument("--max-theta_i", dest="max_theta_i", type=float, default=1)
parser.add_argument("--num-theta_i", dest="num_theta_i", type=int, default=11)

parser.add_argument("--min-chi0", dest="min_chi0", type=float, default=1e-3)
parser.add_argument("--max-chi0", dest="max_chi0", type=float, default=1e2)
parser.add_argument("--num-chi0", dest="num_chi0", type=int, default=11)

parser.add_argument("--min-g", dest="min_g", type=float, default=1e-5)
parser.add_argument("--max-g", dest="max_g", type=float, default=1e-3)
parser.add_argument("--num-g", dest="num_g", type=int, default=3)

parser.add_argument("--min-Gamma_phi", dest="min_Gamma_phi", type=float, default=1e-5)
parser.add_argument("--max-Gamma_phi", dest="max_Gamma_phi", type=float, default=1)
parser.add_argument("--num-Gamma_phi", dest="num_Gamma_phi", type=int, default=4)

parser.add_argument("--output-filename", dest="output_filename", default="scan.pkl")

args = parser.parse_args()

H_inf = args.H_inf
f_a_min = constraints.calc_f_a_min(H_inf)
m_a = constraints.minimal_axion_mass_from_decay(f_a_min)

theta_i_range = np.geomspace(args.min_theta_i, args.max_theta_i, args.num_theta_i)
chi0_range = np.geomspace(args.min_chi0, args.max_chi0, args.num_chi0) * H_inf
g_range = np.geomspace(args.min_g, args.max_g, args.num_g)
Gamma_phi_range = np.geomspace(args.min_Gamma_phi, args.max_Gamma_phi, args.num_Gamma_phi) * H_inf

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

with ProcessPoolExecutor(max_workers=num_workers) as pool:
    outputs = list(pool.map(do_with_timeout, inputs))
util.save_data(output_filename, H_inf, inputs, outputs)
