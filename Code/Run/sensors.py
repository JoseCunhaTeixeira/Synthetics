from waves import Wavelet
from numpy import zeros, arange, array, transpose
from tools import diag_print, verify_expected
from models import DispersionCurve
from tqdm import tqdm
from scipy.signal import convolve



class Sensor:
    def __init__(self, x, y, number):
        self.x = x
        self.y = y
        self.number = number


class SensorArray:
    def __init__(self, **kwargs):
        self.d_sensor = 3.0
        self.N = 0
        self.L = 0.0
        self.sensors = []
        if len(kwargs) > 0:
            self.fill(**kwargs)

    def add_sensor(self, sensor):
        if isinstance(sensor, Sensor):
            self.sensors.append(sensor)
            self.N += 1
        else:
            diag_print("ERROR", "sensorArray add_to_lit", "Element added is not from class Sensor")
            raise SystemExit

    def fill(self, **kwargs):
        x_start = 0
        y = 0
        verify_expected(kwargs, ("N", "L", "d_sensor", "y", "x_start"))
        if "d_sensor" in kwargs:
            self.d_sensor = kwargs["d_sensor"]
        if "N" in kwargs and "L" not in kwargs:
            self.N = kwargs["N"]
            self.L = kwargs["N"] * self.d_sensor
        elif "L" in kwargs and "N" not in kwargs:
            self.L = kwargs["L"]
            self.N = round(self.L / self.d_sensor)
        elif "N" in kwargs and "L" in kwargs:
            diag_print("ERROR", "SensorArray - add_line", "Only L (Array line leght in meters) or N (number of sensors) is expected")
            raise SystemExit
        if "y" in kwargs:
            y = kwargs["y"] 
        if "x_start" in kwargs:
            x_start = kwargs["x_start"]
        for i in arange(0, self.N, 1):
            self.sensors.append(( Sensor(round(x_start + i*self.d_sensor, 1), y, i) ))
            # self.sensors.append(( Sensor(x_start, round(y + i*self.d_sensor, 1), i) ))

    def print(self):
        diag_print("INFO", "Sensor Array parameters", f"Sensors step = {self.d_sensor} m\n          | Number of sensors = {self.N}\n          | Sensor Array length = {self.L} m")


class Seismogram:
    def __init__(self, t_domain, f_domain, s_domain, train, rail_way, sensor_array, ground_model):
        self.t_domain = t_domain
        self.f_domain = f_domain
        self.s_domain = s_domain
        self.train = train
        self.rail_way = rail_way
        self.sensor_array = sensor_array
        if isinstance(ground_model, DispersionCurve):
            self.ground_model = ground_model    # DispersionCurve stocked
        else :
            self.ground_model = None    # EikonalModel too big to be stocked
        self.data_dict = self.generate(f_domain, t_domain, sensor_array, rail_way, train, ground_model)
        self.data_array = transpose(array(list(self.data_dict.values())))

    def generate(self, f_domain, t_domain, sensor_array, rail_way, train, ground_model):
        seismo = {}
        sensor_nbr = 1
        rail_way.generate(train)
        for sensor, i in zip( sensor_array.sensors, tqdm(range(sensor_array.N), colour='blue', initial=1, desc='COMPUTING | Seismogram', leave=False) ):
            sensor_str = sensor.number
            seismo[sensor_str] = zeros(t_domain.N)
            for sleeper, j in zip( rail_way.sleepers, tqdm(range(rail_way.N), colour='cyan', initial=1, desc='          | Sensor '+str(sensor_nbr)+"/"+str(sensor_array.N), leave=False) ):
                wavelet_amp = Wavelet(f_domain, t_domain, sensor, sleeper, ground_model).amp
                source = sleeper.source
                signal = convolve(source, wavelet_amp)[0:t_domain.N]
                seismo[sensor_str] += signal
            sensor_nbr += 1
        diag_print("INFO", "seismogram", "Seismogram fully computed")
        return seismo