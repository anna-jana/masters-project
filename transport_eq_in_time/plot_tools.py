import numpy as np, matplotlib.pyplot as plt, matplotlib.colors as mcolors, matplotlib as mpl
import os

plotpath = "plots"
if not os.path.exists(plotpath):
    os.mkdir(plotpath)

mpl.rcParams["font.size"] = 15

def latex_exponential_notation(value, digits=1):
    exponent = int(np.floor(np.log10(np.abs(value))))
    prefix = value / 10**exponent
    rounded_prefix = np.round(np.abs(prefix) * 10**digits) / 10.0**digits
    format_string_prefix = r"%." + str(digits) + "f"
    rounded_prefix_string = format_string_prefix % rounded_prefix
    while rounded_prefix_string and rounded_prefix_string[-1] == "0":
        rounded_prefix_string = rounded_prefix_string[:-1]
    if rounded_prefix_string and rounded_prefix_string[-1] == ".":
        rounded_prefix_string = rounded_prefix_string[:-1]
        if rounded_prefix_string and rounded_prefix_string[-1] == "1":
            rounded_prefix_string = ""
    if rounded_prefix_string:
        latex_string = rounded_prefix_string + r"\cdot 10^{%i}" % exponent
    else:
        latex_string = "10^{%i}" % exponent
    if value < 0:
        latex_string = "-" + latex_string
    return latex_string
