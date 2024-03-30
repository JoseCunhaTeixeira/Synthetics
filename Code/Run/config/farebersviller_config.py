# CONFIGURATION FILE
# Farebersviller

from numpy import array



name = "farebersviller_config"


t_domain = {
    "dt" : 0.002,
    "t_min" : 0.0,
    "t_max" : 60.0,
}


f_domain = {
    #-- kwargs
    "f_max" : 50.0,
}


s_domain = {
    "dx" : 0.6,
    "dy" : 0.6,
    "x_min" : 0.0,
    "x_max" : 1308.0,
    "y_min" : 0.0,
    "y_max" : 10.2
}


# thickness, Vp, Vs, density
# km, km/s, km/s, g/cm3
# velocity_model = array([
#                     [0.005, 1.900, 0.200, 1.00],
#                     [0.005, 2.000, 0.300, 1.10],
#                     [0.001, 1.900, 0.200, 0.90],
#                     [0.001, 1.700, 0.150, 0.80],
#                     [0.001, 1.500, 0.100, 0.50],
#                     [0.001, 1.900, 0.150, 0.80],
#                     [0.001, 2.000, 0.300, 1.00],
#                     [0.050, 2.500, 0.900, 1.20],
#                     [0.100, 3.000, 1.000, 1.40],
#                     ])
velocity_model = None


train = {
    "train_speed" : 34.0,
    "wagons_nbr" : 8,
    "L_w" : 26.5,
    "d_a" : 2.5,
    "d_b1" : 19.0,
    "d_b2" : 7.5
}


rail_way = {
    #-- kwargs
    'x_start' : 3.0,
    'y' : 7.2,
    "d_sleeper" : 0.6,
    "L" : 1302.0,
}


sensor_array = {
    #-- kwargs
    "x_start" : 603.0,
    "y" : 4.2,
    "d_sensor" : 1.2,
    "L" : 102.0,
}