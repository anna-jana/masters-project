
def calc_H_inf_max(f_a):
    return 6e11 * (f_a / 1e15) # paper

def calc_f_a_min(H_inf):
    return 1e15 / 6e11 * H_inf

def minimal_axion_mass_from_decay(f_a):
    return 8e4 * (f_a / 1e15)**(2/3)
