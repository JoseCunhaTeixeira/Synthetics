from pickle import load, dump
from os import getcwd, makedirs, path
from numpy import flipud

import sys
sys.path.append('./../Run/')
from tools import diag_print



_DATA_PATH = getcwd() + "/../../Data/"

input = open(_DATA_PATH + "Cavity_TS34/seismogram.pickle", "rb")
seismogram34 = load(input)
input.close()

input = open(_DATA_PATH + "Cavity_TS32/seismogram.pickle", "rb")
seismogram32 = load(input)
input.close()

input = open(_DATA_PATH + "Cavity_TS30/seismogram.pickle", "rb")
seismogram30 = load(input)
input.close()

input = open(_DATA_PATH + "Cavity_TS28/seismogram.pickle", "rb")
seismogram28 = load(input)
input.close()

input = open(_DATA_PATH + "Cavity_TS26/seismogram.pickle", "rb")
seismogram26 = load(input)
input.close()

seismogram = seismogram26
seismogram.data_array = seismogram26.data_array + seismogram28.data_array + seismogram30.data_array + seismogram32.data_array + seismogram34.data_array

_PATH = _DATA_PATH + "Stack_TS/"
if not path.exists(_PATH):
    makedirs(_PATH)
    diag_print("INFO", "Main", f"Data folder Stack_TS created")

output = open(_PATH + "seismogram.pickle", "wb")
dump(seismogram, output)
output.close()
diag_print("INFO", "Stack_velocities", "seismogram saved")