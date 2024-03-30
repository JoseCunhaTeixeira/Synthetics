from numpy import cos, hanning, zeros, concatenate, arange
from math import pi, sqrt
from tools import distance
from models import DispersionCurve



class Wavelet:
    def __init__(self, f_domain, t_domain, sensor, sleeper, ground_model):
        self.f_domain = f_domain
        self.t_domain = t_domain
        self.sensor = sensor
        self.sleeper = sleeper
        if isinstance(ground_model, DispersionCurve) :
            self.amp = self.wave_sum_dispersion_curve(f_domain, t_domain, sensor, sleeper, ground_model)
        else :
            self.amp = self.wave_sum_model(f_domain, t_domain, sensor, sleeper, ground_model)

    def wave_sum_model(self, f_domain, t_domain, sensor, sleeper, ground_model):
        tab_f = f_domain.tab
        N_f = f_domain.N

        t_min = t_domain.t_min
        t_max = t_domain.t_max
        dt = t_domain.dt
        N = t_domain.N

        x_sensor = sensor.x
        y_sensor = sensor.y
        x_sleeper = sleeper.x
        y_sleeper = sleeper.y
        offset = distance([x_sleeper, y_sleeper], [x_sensor, y_sensor])
        
        sns_nbr = sensor.number
        slp_nbr = sleeper.number

        travel_times = ground_model.travel_times
        t_max2 = 0
        t_min2 = 1000000000
        for f_idx, f in enumerate(tab_f):
            if travel_times[f_idx][sns_nbr][slp_nbr] < t_min2:
                t_min2 = travel_times[f_idx][sns_nbr][slp_nbr]
            if travel_times[f_idx][sns_nbr][slp_nbr] > t_max2:
                t_max2 = travel_times[f_idx][sns_nbr][slp_nbr]

        delta = 4
        t_min2 = t_min2 - delta
        t_max2 = t_max2 + delta
        if t_min2 < t_min or t_min2 >= t_max :
            t_min2 = t_min
        if t_max2 > t_max or t_max2 <= t_min:
            t_max2 = t_max
        if t_min2 >= t_max2 :
            t_min2 = t_min
            t_max2 = t_max
        t2 = arange(t_min2, t_max2, dt)
        N2 = len(t2)
        
        wave_sum = zeros(N2)
        for (f_idx, f), appo in zip(enumerate(tab_f), hanning(N_f)):
            puls = 2*pi*f
            dissipation = 1/sqrt(offset)
            travel_time = travel_times[f_idx][sns_nbr][slp_nbr]
            wave_sum +=  dissipation * cos(puls * (t2 - travel_time)) * appo
        if t_min2 != t_min:
            wave_sum = concatenate( (zeros(round(t_min2/dt)), wave_sum) )
        if t_max2 != t_max:
            wave_sum = concatenate( (wave_sum, zeros(N - N2)) )
        return wave_sum

    def wave_sum_dispersion_curve(self, f_domain, t_domain, sensor, sleeper, dispersion_curve):
        tab_f = f_domain.tab
        N_f = f_domain.N
        tab_v = dispersion_curve.tab

        t_min = t_domain.t_min
        t_max = t_domain.t_max
        dt = t_domain.dt
        N = t_domain.N

        x_sensor = sensor.x
        y_sensor = sensor.y
        x_sleeper = sleeper.x
        y_sleeper = sleeper.y
        offset = distance([x_sleeper, y_sleeper], [x_sensor, y_sensor])

        delta = 4
        t_min2 = round(offset/max(tab_v)) - delta
        t_max2 = round(offset/min(tab_v)) + delta
        if t_min2 < t_min or t_min2 >= t_max :
            t_min2 = t_min
        if t_max2 > t_max or t_max2 <= t_min:
            t_max2 = t_max
        if t_min2 >= t_max2 :
            t_min2 = t_min
            t_max2 = t_max
        t2 = arange(t_min2, t_max2, dt)
        N2 = len(t2)
        
        wave_sum = zeros(N2)
        for f, v_phase, appo in zip(tab_f, tab_v, hanning(N_f)):
            k = 2*pi*f/v_phase
            puls = 2*pi*f
            dissipation = 1/sqrt(offset)
            wave_sum +=  dissipation * cos(k * offset - puls * t2) * appo
        if t_min2 != t_min:
            wave_sum = concatenate( (zeros(round(t_min2/dt)), wave_sum) )
        if t_max2 != t_max:
            wave_sum = concatenate( (wave_sum, zeros(N - N2)) )
        return wave_sum