import numpy as np

# parameter: mR the radius of the compactification times a mass scale m

def calc_Z(theta, mR):
    return 1 / (np.cosh(np.pi * mR) - np.cos(theta))

def coth(x):
    return 1 / np.tanh(x)

def calc_dZ_dtheta(theta, mR):
    return - np.sin(theta) / (coth(np.pi * mR) - np.cos(theta))**2

# parameter: M^2 = \Lambda^4 / f^2 !!!! not the axion mass

def calc_dV_dtheta(theta, M):
    return M**2 * np.sin(theta)

def clock_work_rhs(log_t, y, _T_fn, H_fn, axion_parameter):
    # v = \dot{\theta}
    theta, v = y
    mR, M = axion_parameter
    t = np.exp(log_t)
    H = H_fn(t)
    Z = calc_Z(theta, mR)
    v_dot = (
            - 1 / (2*Z) * calc_dZ_dtheta(theta, mR) * v**2
            - 3 * H * v
            - calc_dV_dtheta(theta, M) / Z
    )
    return v * t, v_dot * t

def clock_work_mass(_T, mR, M):
    return np.exp(- np.pi * mR) * M

