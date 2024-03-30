# CONFIGURATION FILE
# Two-dimensional near-surface seismic imaging with surface waves : alternative methodology for waveform inversion
# Carlos Andrés Pérez Solano

import numpy as np


name = "solano_config"


t_domain = {
    "dt" : 0.002,
    "t_min" : 0.0,
    "t_max" : 20.0,
}


f_domain = {
    "f_max" : 250
}


velocity_model = np.array([
   [10, 2500, 1200, 1.00],
   [10, 3000, 1500, 1.20],
   [30, 3500, 1800, 1.40],
])


train = {
    "train_speed" : 30,
    "wagons_nbr" : 1,
    "L_w" : 26.5,
    "d_a" : 2.5,
    "d_b1" : 19.0,
    "d_b2" : 7.5
}


rail_way = {
    "N" : 1
}


sensor_array = {
    "L" : 1000,
    "x_start" : -100
}


seismogram = {
    "filter" : "hanning"
}