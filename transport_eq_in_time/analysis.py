from scipy.interpolate import interp1d
from scipy.optimize import root
from skimage.measure import find_contours

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

def find_level_intersection(xs1, ys1, xs2, ys2):
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
