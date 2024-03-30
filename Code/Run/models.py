from numpy import zeros, array
from tqdm import tqdm
import pykonal
from tools import diag_print
from disba import PhaseDispersion
import numpy as np



_VM_HEALTHY = array([
            [0.005, 1.900, 0.200, 1.00],
            [0.010, 2.000, 0.300, 1.10],
            [0.050, 2.500, 0.800, 1.20],
            [0.100, 3.000, 1.000, 1.40],
            ])
_VM_SICK = array([
            [0.005, 1.900, 0.200, 1.00],
            [0.005, 2.000, 0.300, 1.10],
            [0.001, 1.900, 0.200, 0.90],
            [0.001, 1.700, 0.150, 0.80],
            [0.001, 1.500, 0.100, 0.50],
            [0.001, 1.900, 0.150, 0.80],
            [0.001, 2.000, 0.300, 1.00],    
            [0.050, 2.500, 0.800, 1.20],
            [0.100, 3.000, 1.000, 1.40],
            ])

_XMIN = 651
_XMAX = 657
_YMIN = 2.4
_YMAX = 8.4


class DispersionCurve:
    def __init__(self, f_domain, velocity_model):
        self.f_domain = f_domain
        self.velocity_model = velocity_model
        # self.tab = self.generate(velocity_model)
        self.tab = np.ones((len(f_domain.tab)))*1000

    def generate(self, velocity_model):
        t = 1/self.f_domain.tab[::-1]
        pd = PhaseDispersion(*velocity_model.T)
        cpr = pd(t, mode=0, wave="rayleigh").velocity[::-1]*1000
        return cpr


class EikonalModel:
    def __init__(self, f_domain, s_domain, rail_way, sensor_array):
        self.f_domain = f_domain
        self.s_domain= s_domain
        self.rail_way = rail_way
        self.sensor_array = sensor_array
        self.velocity_models = self.generate_velocity_models(s_domain)
        self.dispersion_curves = self.generate_dispersion_curves(f_domain, s_domain)
        self.travel_times = self.generate_travel_times(f_domain, s_domain, rail_way, sensor_array)

    def generate_velocity_models(self, s_domain):
        velocity_models = {}
        for i in range(0, s_domain.Nx):
            for j in range(0, s_domain.Ny):
                if s_domain.x_list[i] >= _XMIN and s_domain.x_list[i] <= _XMAX and s_domain.y_list[j] >= _YMIN and s_domain.y_list[j] <= _YMAX:
                    velocity_models[i,j] = _VM_SICK
                else:
                    velocity_models[i,j] = _VM_HEALTHY
        return velocity_models

    def generate_dispersion_curves(self, f_domain, s_domain):
        dispersion_curves = {}
        # FOR NOW -----------------------------------------------------------------
        dispersion_curve_healthy = DispersionCurve(f_domain, _VM_HEALTHY).tab
        dispersion_curve_sick = DispersionCurve(f_domain, _VM_SICK).tab
        # -------------------------------------------------------------------------
        for i in range(0, s_domain.Nx):
            for j in range(0, s_domain.Ny):
                # dispersion_curves[i,j] = DispersionCurve(f_domain, self.velocity_models[i,j]).tab
                if s_domain.x_list[i] >= _XMIN and s_domain.x_list[i] <= _XMAX and s_domain.y_list[j] >= _YMIN and s_domain.y_list[j] <= _YMAX:
                    dispersion_curves[i,j] = dispersion_curve_sick
                else:
                    dispersion_curves[i,j] = dispersion_curve_healthy
        return dispersion_curves

    def generate_travel_times(self, f_domain, s_domain, rail_way, sensor_array):
        freqs = f_domain.tab
        Nx = s_domain.Nx
        Ny = s_domain.Ny
        x_list = s_domain.x_list
        y_list = s_domain.y_list
        x_min = s_domain.x_min
        y_min = s_domain.y_min
        dx = s_domain.dx
        dy = s_domain.dy
        sleepers = rail_way.sleepers
        sensors = sensor_array.sensors
        dispersion_curves =  self.dispersion_curves

        travel_times = []
        for f, k in zip( freqs, tqdm(range(len(freqs)), colour='blue', initial=1, desc='INIT      | Travel Times', leave=False) ):
            travel_times_f = []
            for sensor in sensors:
                travel_times_f_sensor = []
                for sleeper in sleepers:
                    travel_times_f_sensor.append(0.0)
                travel_times_f.append(travel_times_f_sensor)
            travel_times.append(travel_times_f)
            
        min_coords = x_min/1000, y_min/1000, 0
        node_intervals = dx/1000, dy/1000, 1
        npts = Nx, Ny, 1
        for (f_idx, f), k in zip( enumerate(freqs), tqdm(range(len(freqs)), colour='blue', initial=1, desc='COMPUTING | Travel Times', leave=False ) ):
            velocities_array = zeros((Nx, Ny))
            for i in range(0, Nx):
                for j in range(0, Ny):
                    velocities_array[i][j] = dispersion_curves[i,j][f_idx]/1000 
            values = velocities_array.reshape(npts)
            for sensor, l in zip(sensors, tqdm(range(len(sensors)), colour='cyan', initial=1, desc='          | Sensor', leave=False) ):
                sns_nbr = sensor.number
                src_idx = x_list.index(sensor.x), y_list.index(sensor.y), 0
                solver = pykonal.EikonalSolver(coord_sys="cartesian")
                solver.velocity.min_coords = min_coords
                solver.velocity.node_intervals = node_intervals
                solver.velocity.npts = npts
                solver.velocity.values = values
                solver.traveltime.values[src_idx] = 0
                solver.unknown[src_idx] = False
                solver.trial.push(*src_idx)
                solver.solve()
                for sleeper in sleepers:
                    slp_nbr = sleeper.number
                    slp_idx = x_list.index(sleeper.x), y_list.index(sleeper.y)
                    travel_time = solver.traveltime.values.reshape((Nx, Ny))[slp_idx]
                    travel_times[f_idx][sns_nbr][slp_nbr] = travel_time
                    del slp_nbr, slp_idx, travel_time
                del sns_nbr, src_idx, solver
            del velocities_array, values
        diag_print("INFO", "ground_model", "Travel times fully computed")
        return travel_times