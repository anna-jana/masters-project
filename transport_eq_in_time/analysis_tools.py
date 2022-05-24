import numpy as np, matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import root, curve_fit
from skimage.measure import find_contours
from scipy.fft import rfft, rfftfreq

def make_interp(vals):
    return interp1d(np.arange(len(vals)), vals)

def find_level(xrange, yrange, fvals, level=0.0):
    xs_interp = make_interp(xrange)
    ys_interp = make_interp(yrange)
    for c in find_contours(fvals, level):
        rows, cols = c.T
        xs = xs_interp(cols)
        ys = mR_interp(rows)
        yield xs, ys

def find_curve_intersection(xs1, ys1, xs2, ys2):
    assert len(xs1) == len(ys1)
    assert len(xs2) == len(ys2)
    xs1_interp = make_interp(xs1)
    ys1_interp = make_interp(ys1)
    xs2_interp = make_interp(xs2)
    ys2_interp = make_interp(ys2)
    def goal(I):
        i, j = I # index into curve one and index into curve two
        # the two curves should intersection i.e. both variables from each curve are equal
        try:
            return (xs1_interp(i) / xs2_interp(j) - 1, ys1_interp(i) / ys2_interp(j) - 1)
        except ValueError:
            return np.nan, np.nan # evaluation outside of the curves is undefined
    initial_guess = len(xs1) / 2, len(ys1) / 2
    sol = root(goal, initial_guess, method="lm")
    if sol.success:
        i, j = sol.x
        return xs1_interp(i), ys1_interp(j)
    else:
        raise ValueError(sol.message)

def analyse_power_spectrum(name, ts, var, ys, axion_model, axion_parameter, skip_percent=1/100):
    tspan = ts[-1] - ts[0]
    skip = tspan * skip_percent
    s = int(np.ceil(skip / (ts[1] - ts[0])))
    fake_f_a = 1
    rho = np.array([ axion_model.get_energy(x, fake_f_a, *axion_parameter) for x in ys.T ])
    def fit_fn(log_t, alpha, log_beta):
        return alpha * log_t + log_beta
    p, cov = curve_fit(fit_fn, np.log(ts[s:]), np.log(rho[s:]), p0=(1, 1))
    alpha, log_beta = p
    signal = var[s:] / ts[s:]**(alpha/2)
    ts = ts[s:] 
    dt = ts[1] - ts[0]
    freq = rfftfreq(len(signal), dt)
    ft = rfft(signal)
    pow_spec = np.abs(ft)
    is_local_max = (pow_spec[:-2] < pow_spec[1:-1]) & (pow_spec[2:] < pow_spec[1:-1])
    is_larger_than_mean = pow_spec[1:-1] > np.mean(pow_spec)
    is_peak = is_local_max & is_larger_than_mean
    peaks = freq[1:-1][is_peak]
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(ts, signal)
    plt.xlabel("time")
    plt.ylabel("signal")
    plt.subplot(1,2,2)
    plt.plot(peaks, pow_spec[1:-1][is_peak], "or")
    plt.plot(freq, pow_spec)
    plt.xscale("linear")
    plt.yscale("log")
    plt.xlabel("freq")
    plt.ylabel("power")
    plt.suptitle(f"time domain power spectrum of {name}")
    plt.tight_layout()
    return peaks