from os import getcwd
from pickle import load, dump
from processing import beamforming, slant_stack, slant_stack_sercel, MASW_tomography, FK_to_FV_curve

import sys
sys.path.append('./../Run/')
from tools import diag_print



_DATA_PATH = "/home/jteixeira/Documents/PhD_Monitoring/Work/Processing/Data/Synthetics/"
_FOLDER = "Stack_TS" + "/"
_PATH = _DATA_PATH + _FOLDER

diag_print("Info", "Process", f"Processing '{_FOLDER}' data")

input = open(_PATH + "seismogram.pickle", "rb")
seismogram = load(input)
input.close()



# PROCESSING --------------------------------------------------------------------------------------------------------------
# FK spectrum ----------
# (FK_spectrum_pre, FK_curve_pre, freqs, ks) = beamforming(seismogram, 0, 1.67, 0.01, period='pre')
# diag_print("Info", "Process", "Fk_spectrum_pre extracted")

# FV_curve_pre = FK_to_FV_curve(FK_curve_pre, ks)

# (FK_spectrum_post, FK_curve_post, freqs, ks) = beamforming(seismogram, 0, 1.67, 0.01, period='post')
# diag_print("Info", "Process", "Fk_spectrum_post extracted")

# FV_curve_post = FK_to_FV_curve(FK_curve_post, ks)

# output = open(_PATH + "FK_spectrum.pickle", "wb")
# dump((FK_spectrum_pre, FK_spectrum_post, FK_curve_pre, FK_curve_post, FV_curve_pre, FV_curve_post, freqs, ks), output)
# output.close()


# FV spectrum ----------
# (FV_spectrum_pre, FV_curve_pre, freqs, vs) = slant_stack_sercel(seismogram, 1, 1500, 1, period='pre')
(FV_spectrum_pre, FV_curve_pre, freqs, vs) = slant_stack(seismogram, 1, 1500, 1, period='pre')
diag_print("Info", "Process", "FV_spectrum_pre done")

# (FV_spectrum_post, FV_curve_post, freqs, vs) = slant_stack_sercel(seismogram, 1, 1500, 1, period='post')
(FV_spectrum_post, FV_curve_post, freqs, vs) = slant_stack(seismogram, 1, 1500, 1, period='post')
diag_print("Info", "Process", "FV_spectrum_post done")

output = open(_PATH + "FV_spectrum.pickle", "wb")
dump((FV_spectrum_pre, FV_spectrum_post, FV_curve_pre, FV_curve_post, freqs, vs), output)
output.close()


# MASW ----------
# antenna_size = 5
# (MASW_pre, freqs) = MASW_tomography(seismogram, antenna_size, period='pre')
# diag_print("Info", "Process", "MASW_pre done")

# (MASW_post, freqs) = MASW_tomography(seismogram, antenna_size, period='post')
# diag_print("Info", "Process", "MASW_post done")

# output = open(_PATH + "MASW.pickle", "wb")
# dump((MASW_pre, MASW_post, freqs, antenna_size), output)
# output.close()