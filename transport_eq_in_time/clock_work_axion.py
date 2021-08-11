import numpy as np

# parameter: mR the radius of the compactification times a mass scale m

def calc_Z(theta, mR):
    return 1 / (np.cosh(np.pi * mR) - np.cos(theta))

def calc_dZdtheta(theta, mR):
    return - np.sin(theta) / (np.sinh(np.pi * mR) - np.cos(theta))**2

# parameter: M_sq = \Lambda^4 / f^2 !!!! not the axion mass

def calc_d_pot_d_theta(theta, M):
    return M**2 * np.sin(theta)

def clock_work_rhs(log_t, y, _T_fn, H_fn, axion_parameter):
    # v = \dot{\theta}
    theta, v = y
    mR, M = axion_parameter
    t = np.exp(log_t)
    H = H_fn(t)
    Z = calc_Z(theta, mR)
    dZdtheta = calc_dZdtheta(theta, mR)
    v_dot = (
            - 1 / (2*Z) * dZdtheta * v**2
            - 3*H*v
            - calc_d_pot_d_theta(theta, M) / Z
    )
    return v * t, v_dot * t

def clock_work_mass(_T, mR, M):
    return np.exp(- np.pi * mR) * M

