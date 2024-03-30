import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import sys
sys.path.append('./../Run/')
import waves, sources, sensors, models
from tools import verify_expected, diag_print



# ALL ---------------------------------------------------------------------------------------------------------------------
def plot_curve(*args, **kwargs):
    if len(args) == 2:
        x1, y1 = args[0], args[1]
        plt.plot(x1, y1, color="black", linewidth=0.5)
    elif len(args) == 4:
        x1, y1, x2, y2 = args[0], args[1], args[2], args[3]
        plt.plot(x1, y1, x2, y2, color="black", linewidth=0.5)
    if len(kwargs) > 0:
        verify_expected(kwargs, ("title", "xlabel", "ylabel", "xlim", "ylim"))
        if "title" in kwargs:
            plt.title(kwargs["title"])
        if "xlabel" in kwargs:
            plt.xlabel(kwargs["xlabel"])
        if "ylabel" in kwargs:
             plt.ylabel(kwargs["ylabel"])
        if "xlim" in kwargs:
            plt.xlim(kwargs["xlim"])
        if "ylim" in kwargs:
            plt.ylim(kwargs["ylim"])
    plt.show()


# FIELD -------------------------------------------------------------------------------------------------------------------
def field_layout(rail_way, sensor_array):
    x_r = []
    y_r = []
    x_s = []
    y_s = []
    for sleeper in rail_way.sleepers:
        x_r.append(sleeper.x)
        y_r.append(sleeper.y)
    for sensor in sensor_array.sensors:
        x_s.append(sensor.x)
        y_s.append(sensor.y)
    plt.scatter(x_r, y_r, color='black', marker='|', s=1000)
    plt.scatter(x_s, y_s, color='blue', marker='^')
    plt.gca().invert_yaxis()
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("Sleepers and sensors layout")
    plt.legend(['sleepers', 'sensors'], markerscale=0.3, fontsize='medium')
    plt.show()


# DISPERSION --------------------------------------------------------------------------------------------------------------
def plot_dispersion_curve(dispersion_curve):
    if dispersion_curve is None:
        diag_print("WARNING", "plot_dispersion_curve", "Dispersion curve is None")
    else :
        plot_curve(dispersion_curve.f_domain.tab, dispersion_curve.tab, title="Dispersion Curve f-v", xlabel="Frequency (Hz)", ylabel="Phase Velocity (m/s)")

def plot_velocity_model(dispersion_curve):
    if dispersion_curve is None:
        diag_print("WARNING", "Read", "Dispersion curve is None")
    else:
        vp = []
        vs =[]
        prof = 0
        vp.append(dispersion_curve.velocity_model[0][1]*1000)
        vs.append(dispersion_curve.velocity_model[0][2]*1000)
        for elt in dispersion_curve.velocity_model:
            for i in np.arange(elt[0]*1000):
                vp.append(elt[1]*1000)
                vs.append(elt[2]*1000)
            prof += elt[0]*1000
        plt.step(vp, np.arange(prof+1), where='post',  linewidth=0.5)
        plt.step(vs, np.arange(prof+1), where='post',  linewidth=0.5)
        plt.ylim(prof, 0)
        plt.xlabel("Velocity (m/s)")
        plt.ylabel("Depth (m)")
        plt.title("Velocity model")
        plt.legend(['Vp', 'Vs'])
        plt.show()


# SENSORS -----------------------------------------------------------------------------------------------------------------
def plot_wiggle(seismogram):
    ylim = (seismogram.t_domain.t_max, seismogram.t_domain.t_min)
    data_array = seismogram.data_array
    vmax = np.max(data_array)
    vmin = np.min(data_array)
    total_subplots = seismogram.sensor_array.N
    for (sensor, data), i in zip(seismogram.data_dict.items(), tqdm(range(seismogram.sensor_array.N), colour='blue', initial=1, desc='PLOTTING | Seismogram', leave=False)):
        index = i+1
        plt.subplot(1, total_subplots, index)
        plt.plot(data, seismogram.t_domain.tab, linewidth=0.5, color="black")
        plt.gca().get_xaxis().set_visible(False) if index == 1 else plt.axis("off") # Only first x axe visible
        plt.box(on=None) # Box contour invisible
        plt.ylabel("Time (s)") if index == 1 else None
        plt.ylim(ylim) # Limit and inversion of y axe
        plt.xlim(vmin, vmax)
        where_black = []; [where_black.append(True) if val > 0 else where_black.append(False) for val in data] # Array containing where amp is positive and negative
        plt.fill_betweenx(y=seismogram.t_domain.tab, x1=0, x2=data, where=where_black, interpolate=True, color="black") # Color  under amp when positive
    plt.gcf().suptitle('Generated multichannel record - Wiggle traces')
    plt.show()

def plot_img(seismogram):
    data_array = seismogram.data_array
    extent = [seismogram.sensor_array.sensors[0].x, seismogram.sensor_array.sensors[-1].x, seismogram.t_domain.t_max, seismogram.t_domain.t_min]
    # vmax = np.quantile(data_array, 0.90)
    # vmin = np.quantile(data_array, 0.10)
    plt.imshow(data_array, cmap='Spectral_r', extent=extent, aspect="auto")#, vmin=vmin, vmax=vmax)
    plt.gca().get_yaxis().set_visible(True)
    plt.box(on=None)
    plt.ylabel("Time (s)")
    plt.xlabel("Position (m)")
    plt.gcf().suptitle('Generated multichannel record - Seismic image')
    plt.show()


# SOURCES -----------------------------------------------------------------------------------------------------------------
def plot_train(train):
    t_w = train.L_w / train.speed
    T = t_w * train.wagons_nbr
    dt = train.t_domain.dt
    plot_curve(np.arange(0, T, dt), train.function, title="Train time function", xlabel="Time (s)", ylabel="Amplitude")

def plot_sleeper(sleeper):
    plt.plot(sleeper.t_domain.tab, sleeper.source, linewidth=0.5, color="black")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

def plot_rail_way(rail_way):
    for sleeper, index, i in zip(rail_way.sleepers, range(rail_way.N), tqdm(range(rail_way.N), colour='blue', initial=1, desc='PLOTTING  | Sleepers activation ', leave=False)):
        plt.subplot(rail_way.N, 1, index+1)
        plt.gca().get_yaxis().set_visible(False) if index == rail_way.N-1 else plt.axis("off") # Only first x axe visible
        plt.ylabel("Time (s)") if index == rail_way.N-1 else None
        plt.box(on=None) # Box contour invisible
        plt.gcf().suptitle("Sleepers activation functions")
        plot(sleeper)
    plt.show()


# WAVES -------------------------------------------------------------------------------------------------------------------
def plot_wavelet(wavelet):
    amp_fft = np.fft.fft(wavelet.amp)
    freq = np.fft.fftfreq(wavelet.t_domain.N, d=wavelet.t_domain.step)
    
    plt.subplot(2, 1, 1)
    plt.plot(wavelet.t_domain.tab, wavelet.amp,  color="black", linewidth=0.5)
    plt.title("Wavelet")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    
    plt.subplot(2, 1, 2)
    plt.plot(freq, np.abs(amp_fft),  color="black", linewidth=0.5)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.show()


# READ --------------------------------------------------------------------------------------------------------------------
def plot(*kargs):
    if len(kargs) == 1:
        if isinstance(kargs[0], sources.Train):
            plot_train(kargs[0])
        if isinstance(kargs[0], sources.Sleeper):
            plot_sleeper(kargs[0])
        if isinstance(kargs[0], sources.RailWay):
            plot_rail_way(kargs[0])
        if isinstance(kargs[0], waves.Wavelet):
            plot_wavelet(kargs[0])
    elif len(kargs) == 2:
        if isinstance(kargs[0], models.DispersionCurve) and kargs[1] == 'dispersion':
            plot_dispersion_curve(kargs[0])
        elif isinstance(kargs[0], models.DispersionCurve) and kargs[1] == 'model':
            plot_velocity_model(kargs[0])
        if isinstance(kargs[0], sensors.Seismogram) and kargs[1] == 'wiggle':
            plot_wiggle(kargs[0])
        elif isinstance(kargs[0], sensors.Seismogram) and kargs[1] == 'image':
            plot_img(kargs[0])

def plot_img_array(data_array, extent, cmap='Spectral_r', **kwargs):
    plt.imshow(data_array, cmap=cmap, extent=extent, aspect="auto")
    plt.gca().yaxis.set_visible(True)
    plt.box(on=None)
    if len(kwargs) > 0:
        verify_expected(kwargs, ("ylabel", "xlabel", "title"))
        if "ylabel" in kwargs:
            plt.ylabel(kwargs["ylabel"])
        if "xlabel" in kwargs:
            plt.xlabel(kwargs["xlabel"])
        if "title" in kwargs:
            plt.gcf().suptitle(kwargs["title"])
    plt.show()

def plot_wiggle_array(data_array, t_domain, sensor_array):
    ylim = (t_domain.max, t_domain.min)
    vmax = np.max(data_array)
    vmin = np.min(data_array)
    total_subplots = data_array.shape[1]
    for data, i in zip(np.transpose(data_array), tqdm(range(sensor_array.N), colour='blue', initial=1, desc='PLOTTING  |', leave=False)):
        index = i+1
        plt.subplot(1, total_subplots, index)
        plt.plot(data, t_domain.tab, linewidth=0.5, color="black")
        plt.gca().xaxis().set_visible(False) if index == 1 else plt.axis("off") # Only first x axe visible
        plt.box(on=None) # Box contour invisible
        plt.ylabel("Time (s)") if index == 1 else None
        plt.ylim(ylim) # Limit and inversion of y axe
        plt.xlim(vmin, vmax)
        where_black = []; [where_black.append(True) if val > 1/10 else where_black.append(False) for val in data] # Array containing where amp is positive and negative
        plt.fill_betweenx(y=t_domain.tab, x1=0, x2=data, where=where_black, interpolate=True, color="black") # Color  under amp when positive
    plt.gcf().suptitle('Generated multichannel record - Wiggle traces')
    plt.show()