import os.path
import shutil
import pickle
from concurrent.futures import ProcessPoolExecutor
import numpy as np

base = os.path.join(os.path.dirname(__file__), "..")

input_data_path = os.path.join(base, "input_data")
output_data_path = os.path.join(base, "output_data")
plot_path = os.path.join(base, "plots")

if not os.path.exists(input_data_path):
    os.mkdir(input_data_path)

if not os.path.exists(output_data_path):
    os.mkdir(output_data_path)

if not os.path.exists(plot_path):
    os.mkdir(plot_path)

def make_plot_path(filename):
    return os.path.join(plot_path, filename)

def make_input_data_path(filename):
    return os.path.join(input_data_path, filename)

def make_output_data_path(filename):
    return os.path.join(output_data_path, filename)

def parallel_map(func, xs):
    with ProcessPoolExecutor() as pool:
        res = pool.map(func)
    return list(res)

def count_oscillations(theta):
    s = np.sign(theta)
    roots = np.sum(s[:-1] != s[1:])
    return roots // 2

def find_local_maxima(theta):
    return np.where((theta[1:-1] > theta[:-2]) & (theta[1:-1] > theta[2:]))[0]

def save_data(filename, *data, use_default_path=True):
    if use_default_path:
        filepath = make_output_data_path(filename)
    else:
        filepath = filename
    with open(filepath, "wb") as fp:
        pickle.dump(data, fp)

def load_data(filename, use_default_path=True):
    if use_default_path:
        filepath = make_output_data_path(filename)
    else:
        filepath = filename
    with open(filepath, "rb") as fp:
        data = pickle.load(fp)
    return data

