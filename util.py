import os.path

input_data_path = "input_data"
output_data_path = "output_data"
plot_path = "plots"

def make_plot_path(filename):
    return os.path.join(plot_path, filename)

def make_input_data_path(filename):
    return os.path.join(input_data_path, filename)

def make_output_data_path(filename):
    return os.path.join(output_data_path, filename)
