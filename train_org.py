# Import the graphscope module

import graphscope

graphscope.set_option(show_log=True)  # enable logging


# Load the obgn_mag dataset as a graph

from graphscope.dataset import load_ogbn_mag

graph = load_ogbn_mag()