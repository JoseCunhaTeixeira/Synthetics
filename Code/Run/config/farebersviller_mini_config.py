# CONFIGURATION FILE
# Farebersviller

import numpy as np



name = "farebersviller_mini_config"


t_domain = {
    "dt" : 0.002,
    "t_min" : 0.0,
    "t_max" : 20.0,
}


f_domain = {
    #-- kwargs
    "f_max" : 50.0,
}


s_domain = {
    "dx" : 0.1,
    "dy" : 0.1,
    "x_min" : 0.0,
    "x_max" : 150.0,
    "y_min" : 0.0,
    "y_max" : 5.0
}


# thickness, Vp, Vs, density
# m, m/s, m/s, g/cm3
velocity_model = np.array([
   [0.005, 1.900, 0.200, 1.00],
   [0.010, 2.000, 0.300, 1.00],
   [0.030, 2.500, 0.800, 1.20],
   [0.050, 3.000, 1.000, 1.40],
])
# velocity_model = None


train = {
    "train_speed" : 34,
    "wagons_nbr" : 8,
    "L_w" : 26.5,
    "d_a" : 2.5,
    "d_b1" : 19.0,
    "d_b2" : 7.5
}


rail_way = {
    #-- kwargs
    "d_sleeper" : 0.6,
    "L" : 146,
    'y' : 3.0,
    'x_start' : 2
}


sensor_array = {
    #-- kwargs
    "x_start" : 60,
    "y" : 2.0,
    "N" : 30,
    "d_sensor" : 1.0
}