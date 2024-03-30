import numpy as np
from matplotlib.pyplot import axes, imshow, plot, title, xlabel, ylabel, show, subplot, close, pause, colorbar
from numpy.core.fromnumeric import size

from numpy.fft import fft, rfft, rfftn, fftn
# from mkl_fft import rfft
from numpy.fft import fftfreq, rfftfreq
from numpy.lib.twodim_base import flipud

# from scipy.fftpack.basic import fft
# from scipy.fft import fftn
#from scipy.fftpack.helper import rfftfreq

from sigproc import makeFV, whiten, correl, rfft_min_ech, rfft2complex
from scipy.signal import lti_conversion, tukey, convolve
from tqdm import tqdm
from math import factorial
from tqdm import tqdm

import sys
sys.path.append('./../Run/')
from tools import distance


#----------------------------------------------------------------------------------------------------
def cut(data, sensor_array, rail_way, t_domain, train, period):
    sleepers = rail_way.sleepers
    closest_sleepers = []
    for sensor in sensor_array.sensors:
        distance_prec = distance((sensor.x, sensor.y), (sleepers[0].x, sleepers[0].y))
        closest_sleeper = sleepers[0]
        for sleeper in sleepers:
            distance_curr = distance((sensor.x, sensor.y), (sleeper.x, sleeper.y))
            if distance_curr < distance_prec :
                closest_sleeper = sleeper
                distance_prec = distance_curr
        closest_sleepers.append(closest_sleeper)
    data_cut = []
    phi_start = []
    phi_end = []
    dt = t_domain.dt
    N = t_domain.N
    t_min = t_domain.t_min
    t_max = t_domain.t_max
    speed = train.speed
    duration = t_domain.tab[len(train.function)] 
    for closest_sleeper in closest_sleepers :
        if period == 'pre':
            phi_s = t_min + 3
            # phi_e = closest_sleeper.x / train.speed - 3
            phi_e = closest_sleepers[0].x / speed - 6
        elif period == 'post':
            phi_s = closest_sleepers[-1].x / speed + duration + 6
            phi_e = t_max - 3
        phi_end.append(phi_e)
        phi_start.append(phi_s)
    for data, phi_s, phi_e in zip(np.transpose(data), phi_start, phi_end):
        temp = data[round(phi_s/dt) : round(phi_e/dt)]
        temp *= tukey(len(temp))
        temp = np.concatenate( (np.zeros(round(phi_s/dt)), temp) )
        data_cut.append(np.concatenate( (temp, np.zeros(N-len(temp))) ))
    data_cut = np.transpose(np.array(data_cut))
    return data_cut


#----------------------------------------------------------------------------------------------------
def interferometry(data):
    data_interf = []
    fictive_source = np.transpose(data)[0]
    for data_T in np.transpose(data):
        data_interf.append(correl(data_T, fictive_source))
    data_interf = np.transpose(np.array(data_interf))
    return data_interf


#----------------------------------------------------------------------------------------------------
def beamforming(seismogram, k_min, k_max, dk, period='pre'):
    data = np.copy(seismogram.data_array)
    t_domain = seismogram.t_domain
    f_domain = seismogram.f_domain
    sensor_array = seismogram.sensor_array
    sensors = sensor_array.sensors
    rail_way = seismogram.rail_way
    train = seismogram.train

    data = cut(data, sensor_array, rail_way, t_domain, train, period)

    data = interferometry(data)

    if period == 'pre':
        x_source = rail_way.sleepers[0].x
        y_source = rail_way.sleepers[0].y
    elif period == 'post':
        x_source = rail_way.sleepers[-1].x
        y_source = rail_way.sleepers[-1].y
    offsets = []
    for sensor in sensors:
        offsets.append(distance((x_source, y_source), (sensor.x, sensor.y)))
    offsets = np.array(offsets)

    # ---

    S = rfft(data, axis=0, n=data.shape[0]) # S[frequency, distance]
    S = S.T # S[distance, frequency]
    
    fs = rfftfreq(data.shape[0], t_domain.dt)

    # Remove f > fmax et f < 1 Hz
    fi_max = np.where(fs > f_domain.f_max)[0][0]
    fi_min = np.where(fs >= 1)[0][0]
    fs = fs[fi_min : fi_max]
    S = S[:, fi_min : fi_max]

    ks = np.arange(k_min, k_max, dk)
    
    e = np.empty((len(offsets), len(ks)), dtype=complex) # e[distance, k]
    for ki, k in enumerate(ks):
        for xi, x  in enumerate(offsets):
            e[xi, ki] = np.exp(-1j * k * x)
    e_H = np.matrix(e).H

    W = np.zeros((len(offsets), len(offsets))) # Shading weights matrix W[distance, distance]
    for xi, x in enumerate(offsets):
        W[xi, xi] = np.sqrt(x)

    Z = np.empty((len(ks), len(fs)), dtype=complex) # Beamformer output Z[k, frequency]
    tmp = np.dot(e_H, W)
    Z = np.dot(tmp, S)
    # Z_H = np.matrix(Z).H

    # P = np.empty((len(ks), len(ks)), dtype=complex) # Steered response power spectrum P[k, k]
    # P = dot(Z, Z_H)

    Z = abs(Z.T) # Z[frequency, k]

    for fi, f in enumerate(fs):
        Z[fi,:] = Z[fi,:] / np.max((Z[fi,:]))

    FK_curve = extract_curve(Z, fs, delta=500, start=50)

    return Z, FK_curve, fs, ks


#----------------------------------------------------------------------------------------------------
def slant_stack(seismogram, v_min, v_max, dv, period='pre'):
    data = np.copy(seismogram.data_array)
    t_domain = seismogram.t_domain
    f_domain = seismogram.f_domain
    sensor_array = seismogram.sensor_array
    sensors = sensor_array.sensors
    rail_way = seismogram.rail_way
    train = seismogram.train

    # extent = [sensors[0].x, sensors[-1].x, t_domain.t_max, t_domain.t_min]
    
    # data = whiten(data, perc=1)
    # plot_img_array(data, extent, title="Generated multichannel record - Whitened", xlabel="Offset (m)", ylabel="Time (s)")
    
    data = cut(data, sensor_array, rail_way, t_domain, train, period)
    # plot_img_array(data, extent, title="Generated multichannel record - Whitened and Cut", xlabel="Offset (m)", ylabel="Time (s)")
    
    data = interferometry(data)
    # plot_img_array(data, extent, title="Generated multichannel record - Whitened, Cut and Correlated", xlabel="Offset (m)", ylabel="Time (s)")

    if period == 'pre':
        x_source = rail_way.sleepers[0].x
        y_source = rail_way.sleepers[0].y
    elif period == 'post':
        x_source = rail_way.sleepers[-1].x
        y_source = rail_way.sleepers[-1].y
    offsets = []
    for sensor in sensors:
        offsets.append(distance((x_source, y_source), (sensor.x, sensor.y)))
    offsets = np.array(offsets)

    vs = np.arange(v_min, v_max, dv)

    Y = rfft(data, axis=(0), n=data.shape[0]) # data[time, distance] -> Y[frequency, distance]
    fs = rfftfreq(data.shape[0], t_domain.dt)

    # Remove f > fmax
    fi_max = np.where(fs > f_domain.f_max)[0][0]
    fs = fs[0:fi_max]
    Y = Y[0:fi_max, :]

    FV = np.empty((len(vs), len(fs))) # FV[velocity, frequency]
    for (fi, f), j in zip( enumerate(fs), tqdm(range(len(fs)), colour='cyan', initial=1, leave=False, desc="COMPUTING | Slant Stack") ):
        for vi, v in enumerate(vs):
            k = 2 * np.pi * f / v
            FV[vi, fi] = np.abs( np.dot( np.exp(1j * k * offsets), Y[fi, :] ) )

    # Remove f < 1Hz
    fi_1 = np.where(fs > 1)[0][0]
    fs = fs[fi_1:]
    FV =  FV[0:, fi_1:]

    for fi, f in enumerate(fs):
        FV[:, fi] = FV[:, fi] / np.max((FV[:, fi]))

    FV_curve = extract_curve(FV, vs, delta=5, start=30)
    
    return FV, FV_curve, fs, vs


#----------------------------------------------------------------------------------------------------
def slant_stack_sercel(seismogram, v_min, v_max, dv, period='pre'):
    data = np.copy(seismogram.data_array)
    t_domain = seismogram.t_domain
    f_domain = seismogram.f_domain
    sensor_array = seismogram.sensor_array
    sensors = sensor_array.sensors
    rail_way = seismogram.rail_way
    train = seismogram.train

    # extent = [sensors[0].x, sensors[-1].x, t_domain.t_max, t_domain.t_min]
    
    # data = whiten(data, perc=1)
    # plot_img_array(data, extent, title="Generated multichannel record - Whitened", xlabel="Offset (m)", ylabel="Time (s)")
    
    data = cut(data, sensor_array, rail_way, t_domain, train, period)
    # plot_img_array(data, extent, title="Generated multichannel record - Whitened and Cut", xlabel="Offset (m)", ylabel="Time (s)")
    
    data = interferometry(data)
    # plot_img_array(data, extent, title="Generated multichannel record - Whitened, Cut and Correlated", xlabel="Offset (m)", ylabel="Time (s)")

    if period == 'pre':
        x_source = rail_way.sleepers[0].x
        y_source = rail_way.sleepers[0].y
    elif period == 'post':
        x_source = rail_way.sleepers[-1].x
        y_source = rail_way.sleepers[-1].y
    offsets = []
    for sensor in sensors:
        offsets.append(distance((x_source, y_source), (sensor.x, sensor.y)))
    offsets = np.array(offsets)

    (FV_spectrum, vs, freqs) = makeFV(np.transpose(data), t_domain.dt, offsets, v_min, v_max, dv, f_domain.f_max)

    # Remove < 1Hz
    i = np.where(freqs == 1)[0][0]
    freqs = freqs[i : ]
    FV_spectrum =  FV_spectrum[0 : , i : ]

    for fi, f in enumerate(freqs):
        FV_spectrum[:,fi] = FV_spectrum[:,fi] / np.max((FV_spectrum[:,fi]))

    FV_curve = extract_curve(FV_spectrum, vs, delta=5, start=30)

    return FV_spectrum, FV_curve, freqs, vs


#----------------------------------------------------------------------------------------------------
def MASW_tomography(seismogram, antenna_size, period='pre'):
    t_domain = seismogram.t_domain
    f_domain = seismogram.f_domain
    rail_way = seismogram.rail_way
    train = seismogram.train
    sensor_array = seismogram.sensor_array
    N_sensors = sensor_array.N
    disp_curves_array = []
    freqs = None
    vs = None

    if period == 'pre':
        x_source = rail_way.sleepers[0].x
        y_source = rail_way.sleepers[0].y
    elif period == 'post':
        x_source = rail_way.sleepers[-1].x
        y_source = rail_way.sleepers[-1].y

    for i, k in zip(range(N_sensors-antenna_size), tqdm(range(N_sensors-antenna_size), colour='blue', initial=0, desc='RUNING    | MASW ', leave=False)):
        data = np.copy(seismogram.data_array[::, i : i+antenna_size-1])
        # print(np.mean(seismogram.data_array))
        sensors = sensor_array.sensors[i : i+antenna_size-1]

        # extent = [sensors[0].x, sensors[-1].x, t_domain.t_max, t_domain.t_min]
        
        # data = whiten(data, perc=1)
        # plot_img_array(data, extent, title="Generated multichannel record - Whitened", xlabel="Offset (m)", ylabel="Time (s)")
        
        data = cut(data, sensor_array, rail_way, t_domain, train, period)
        # plot_img_array(data, extent, title="Generated multichannel record - Whitened and Cut", xlabel="Offset (m)", ylabel="Time (s)")
        
        data = interferometry(data)
        # plot_img_array(data, extent, title="Generated multichannel record - Whitened, Cut and Correlated", xlabel="Offset (m)", ylabel="Time (s)")

        offsets = []
        for sensor in sensors:
            offsets.append(distance((x_source, y_source), (sensor.x, sensor.y)))
        offsets = np.array(offsets)

        (FV_spectrum, vs, freqs) = makeFV(np.transpose(data), t_domain.dt, offsets, 1, 1500, 1, f_domain.f_max)

        # Remove 0-1Hz
        idx = np.where(freqs == 1)[0][0]
        freqs = freqs[idx : ]
        FV_spectrum = FV_spectrum[0 : , idx : ]

        FV_curve = extract_curve(FV_spectrum, vs, delta=5, start=300)
        disp_curves_array.append(FV_curve)

        for j, f in enumerate(freqs):
            FV_spectrum[:,j] = FV_spectrum[:,j] / np.max((FV_spectrum[:,j]))

        # extent = [freqs[0], freqs[-1], vs[0], vs[-1]]
        # imshow(np.flipud(FV_spectrum), extent=extent, aspect='auto', cmap='Spectral_r' )
        # plot(freqs, FV_curve)
        # show()

    disp_curves_array = np.transpose(np.array(disp_curves_array))

    return disp_curves_array, freqs


#----------------------------------------------------------------------------------------------------
def extract_curve(arr, ax, delta, start):
    """
    Extracts f-v dispersion curve from f-v dispersion diagram by aiming maximums

    args :
        arr (2D numpy array) : dispersion diagram
        ax (1D numpy array) : velocity axis
        delta (float) : 
        start (float) : start value

    returns :
        curve (1D numpy array[velocity]) : f-v dispersion curve
    """
    curve = []
    for i, line in enumerate((np.flipud(np.array(arr).T))):
        if i >= start :
            b_inf = max_idx - delta
            b_sup = max_idx + delta
            if b_inf <= 0 :
                b_inf = 0
            if b_sup >= len(line):
                b_sup = len(line)
            line2 = line[b_inf : b_sup]
            max_idx = np.where(line2 == line2.max())[0][0] + b_inf
            max_val = ax[max_idx]
            curve.append(max_val)
        else :
            max_idx = np.where(line == line.max())[0][0]
            max_val = ax[max_idx]
            curve.append(max_val)
    curve = curve[::-1]
    window_size = len(curve) // 10
    if window_size % 2 == 0 :
        window_size += 1
    curve = savitzky_golay(curve, window_size, 3)
    return curve


#----------------------------------------------------------------------------------------------------
def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')


#----------------------------------------------------------------------------------------------------
def FK_to_FV_curve(FK_curve, ks):
    FV_curve = np.empty(ks.shape)
    for (ki, k), FK in zip(enumerate(ks[::-1]), FK_curve[::-1]):
        if k != 0 :
            FV_curve[ki] = 2 * np.pi * FK / k
        else :
            FV_curve[ki] = FV_curve[ki-1]
    FV_curve = FV_curve[::-1]
    window_size = len(FV_curve) // 10
    if window_size % 2 == 0 :
        window_size += 1
    return FV_curve
