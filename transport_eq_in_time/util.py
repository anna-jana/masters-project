import numpy as np, h5py
import pickle, os

datadir = "data"

def load_pkl(filename):
    with open(filename, "rb") as fh:
        data = pickle.load(fh)
    return data

def save_pkl(data, filename):
    with open(filename, "wb") as fh:
        pickle.dump(data, fh
        )
