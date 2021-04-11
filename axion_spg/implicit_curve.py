import numpy as np

# ## Search for the Observed Asymmetry and Compute Possible Parameter Volume
def grad(f, p, h=1e-3, **kwargs):
    x, y = p
    return np.array([
        (f((x + h, y)) - f((x - h, y))) / (2*h),
        (f((x, y + h)) - f((x, y - h))) / (2*h),
    ])

def valid(p, x_bounds, y_bounds, only_upper_bounds=True, **kwargs):
    min_x, max_x = x_bounds
    min_y, max_y = y_bounds
    x, y = p
    if only_upper_bounds:
        return x <= max_x and y <= max_y
    else:
        return min_x <= x <= max_x and min_y <= y <= max_y

def find_root(f, p, x_bounds, y_bounds, eps=1e-3, max_steps=100, **kwargs):
    for step in range(max_steps):
        if not valid(p, x_bounds, y_bounds, only_upper_bounds=kwargs["only_upper_bounds"] if "only_upper_bounds" in kwargs else True):
            raise ValueError("run out of domain")
        grad_f = grad(f, p, **kwargs)
        new_p = p - f(p) / np.dot(grad_f, grad_f) * grad_f
        if np.linalg.norm(p - new_p) < eps:
            return new_p
        p = new_p
    raise ValueError("exceeded maximal steps")

def trace_curve(f, p, x_bounds, y_bounds, direction, step_length=1.0, debug=False, **kwargs):
    points = []
    while True:
        grad_f = grad(f, p, **kwargs)
        t = np.array([-grad_f[1], grad_f[0]])
        next_p = p + direction * step_length * t
        try:
            next_p = find_root(f, next_p, x_bounds, y_bounds, **kwargs)
        except ValueError:
            print("INFO: stopped because of convergence failure of newton iteration")
            break
        if valid(next_p, x_bounds, y_bounds, **kwargs):
            p = next_p
            points.append(p)
            if debug:
                print("added point", p)
        else:
            break
    return points

def find_implicit_curve(f, x_bounds, y_bounds, p, **kwargs):
    try:
        p = find_root(f, p, x_bounds, y_bounds, **kwargs)
    except ValueError:
        print("INFO: falied to find initial point")
        return np.array([]), np.array([])
    points = list(reversed(trace_curve(f, p, x_bounds, y_bounds, 1, **kwargs)))
    points.append(p)
    points += trace_curve(f, p, x_bounds, y_bounds, -1, **kwargs)
    return np.array([p[0] for p in points]), np.array([p[1] for p in points])


