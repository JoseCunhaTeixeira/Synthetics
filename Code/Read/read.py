from os import getcwd
from pickle import load
from display import plot, field_layout
from matplotlib.pyplot import imshow, title, xlabel, ylabel, show, colorbar, figure, savefig
from matplotlib.pyplot import plot as py_plot
from numpy import flipud
import numpy as np


import sys
sys.path.append('./../Run/')
from tools import diag_print



_DATA_PATH = "/home/jteixeira/Documents/PhD_Monitoring/Work/Processing/Data/Synthetics/"
_FOLDER = "2023-02-22_12h07" + "/"
_PATH = _DATA_PATH + _FOLDER

diag_print("Info", "Read", f"Reading '{_FOLDER}' data")

input = open(_PATH + "seismogram.pickle", "rb")
seismogram = load(input)
input.close()

t_domain = seismogram.t_domain
t_domain.print()

f_domain = seismogram.f_domain
f_domain.print()

s_domain = seismogram.s_domain
s_domain.print()

ground_model = seismogram.ground_model

train = seismogram.train
train.print()

rail_way = seismogram.rail_way
rail_way.print()

sensor_array = seismogram.sensor_array
sensor_array.print()

# input = open(_PATH + "FK_spectrum.pickle", "rb")
# (FK_spectrum_pre, FK_spectrum_post, FK_curve_pre, FK_curve_post, FV_curve_pre_beam, FV_curve_post_beam, freqs_FK, ks) = load(input)
# input.close()

# input = open(_PATH + "FV_spectrum.pickle", "rb")
# (FV_spectrum_pre, FV_spectrum_post, FV_curve_pre, FV_curve_post, freqs_FV, vs) = load(input)
# input.close()

# input = open(_PATH + "MASW.pickle", "rb")
# (MASW_pre, MASW_post, freqs_MASW, antenna_size) = load(input)
# input.close()



# DISPLAY -----------------------------------------------------------------------------------------------------------------
field_layout(rail_way, sensor_array)
plot(ground_model, 'dispersion')
plot(ground_model, 'model')
plot(train)
plot(rail_way)
plot(seismogram, 'wiggle')
plot(seismogram, 'image')

# FK spectrums
# extent = [ks[0], ks[-1], freqs_FK[0], freqs_FK[-1]]

# figure('f-k - pre')
# imshow(flipud(FK_spectrum_pre), cmap='Spectral_r', aspect='auto', extent=extent)
# py_plot(ks, FK_curve_pre, color='black', linewidth=0.5)
# xlabel('Wavenumbers k (1/m)')
# ylabel('Frequencies (Hz)')
# title('Dispersion spectrum f-k (Pre)')

# figure('f-v curve - pre')
# py_plot(FK_curve_post, FV_curve_pre_beam, '.-', color='black', linewidth=0.5, markersize=1)
# xlabel('Frequencies (Hz)')
# ylabel('Phase velocities (m/s)')
# title('Dispersion curve f-v (Pre)')

# figure('f-k - post')
# imshow(flipud(FK_spectrum_post), cmap='Spectral_r', aspect='auto', extent=extent)
# py_plot(ks, FK_curve_pre, color='black', linewidth=0.5)
# xlabel('Wavenumbers k (1/m)')
# ylabel('Frequencies (Hz)')
# title('Dispersion spectrum f-k (Post)')

# figure('f-v curve - post')
# py_plot(FK_curve_post, FV_curve_post_beam, '.-', color='black', linewidth=0.5, markersize=1)
# xlabel('Frequencies (Hz)')
# ylabel('Phase velocities (m/s)')
# title('Dispersion curve f-v (Post)')
# show()


# FV spectrums
# extent = [freqs_FV[0], freqs_FV[-1], vs[0], vs[-1]]

# figure('f-v - pre')
# imshow(flipud(FV_spectrum_pre), cmap='Spectral_r', aspect='auto', extent=extent)
# py_plot(freqs_FV, FV_curve_pre, color='black', linewidth=0.5)
# xlabel('Frequencies (Hz)')
# ylabel('Phase velocities (m/s)')
# title('Dispersion spectrum f-v (Pre)')

# figure('f-v - post')
# imshow(flipud(FV_spectrum_post), cmap='Spectral_r', aspect='auto', extent=extent)

# py_plot(freqs_FV, FV_curve_post, color='black', linewidth=0.5)
# xlabel('Frequencies (Hz)')
# ylabel('Phase velocities (m/s)')
# title('Dispersion spectrum f-v (Post)')
# show()


# MASW
# N_sensors = sensor_array.N
# extent = [round(antenna_size/2)+1, N_sensors-round(antenna_size/2), freqs_MASW[0], freqs_MASW[-1]]
# vmin = vs[0]
# vmax = vs[-1]

# imshow(flipud(MASW_pre), cmap='terrain', aspect='auto', extent=extent, vmin=vmin, vmax=vmax)
# xlabel('Sensor Number')
# ylabel('Frequency (Hz)')
# title('MASW Section (Pre)')
# cbar = colorbar()
# cbar.set_label('Phase velocity (m/s)')
# show()

# imshow(flipud(MASW_post), cmap='terrain', aspect='auto', extent=extent, vmin=vmin, vmax=vmax)
# xlabel('Sensor Number')
# ylabel('Frequency (Hz)')
# title('MASW Section (Post)')
# cbar = colorbar()
# cbar.set_label('Phase velocity (m/s)')
# show()

# disp_curves_array = (MASW_pre + MASW_post) / 2
# imshow(flipud(disp_curves_array), cmap='terrain', aspect='auto', extent=extent, vmin=vmin, vmax=vmax)
# xlabel('Sensor Number')
# ylabel('Frequency (Hz)')
# title('Stacked MASW Section (Arithmetic)')
# cbar = colorbar()
# cbar.set_label('Phase velocity (m/s)')
# show()

# disp_curves_array = 2 / (1/MASW_pre + 1/MASW_post)
# imshow(flipud(disp_curves_array), cmap='terrain', aspect='auto', extent=extent, vmin=vmin, vmax=vmax)
# xlabel('Sensor Number', fontsize=14)
# ylabel('Frequency (Hz)', fontsize=14)
# title('Stacked MASW Section (Harmonic)', fontsize=14)
# cbar = colorbar()
# cbar.set_label('Phase velocity (m/s)', fontsize=14)
# show()
# savefig("~/Downloads/MASW.png", transparent=True)