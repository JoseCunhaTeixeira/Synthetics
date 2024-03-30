from domains import TimeDomain, FrequencyDomain, SpaceDomain
from models import EikonalModel, DispersionCurve
from sensors import SensorArray, Seismogram
from sources import Train, RailWay
from pickle import dump
from os import getcwd, makedirs, path
from time import time
from tools import save_info, diag_print
from datetime import datetime, timedelta
from config import farebersviller_config as cfg



if __name__ == "__main__":
    for train_speed in (34, 32, 30, 28, 26):

        # START ----------
        start_time = time()
        diag_print('INFO', 'Main', f"Loading '{cfg.name}' configuration : train speed : {train_speed}")
    


        # DOMAINS ----------
        dt = cfg.t_domain["dt"]
        t_min = cfg.t_domain["t_min"]
        t_max = cfg.t_domain["t_max"]
        t_domain = TimeDomain(dt, t_min, t_max)
        t_domain.print()

        f_domain = FrequencyDomain(t_domain, **cfg.f_domain)
        f_domain.print()

        dx = cfg.s_domain["dx"]
        dy = cfg.s_domain["dy"]
        x_min = cfg.s_domain["x_min"]
        x_max = cfg.s_domain["x_max"]
        y_min = cfg.s_domain["y_min"]
        y_max = cfg.s_domain["y_max"]
        s_domain = SpaceDomain(dx, dy, x_min, x_max, y_min, y_max)
        s_domain.print()



        # TRAIN ---------
        wagons_nbr = cfg.train["wagons_nbr"]
        L_w = cfg.train["L_w"]
        d_a = cfg.train["d_a"]
        d_b1 = cfg.train["d_b1"]
        d_b2 = cfg.train["d_b2"]
        train = Train(train_speed, wagons_nbr, L_w, d_a, d_b1, d_b2, t_domain)
        train.print()


    
        # RAILWAY ----------
        rail_way = RailWay(**cfg.rail_way)
        rail_way.print()



        # SENSORS ---------
        sensor_array = SensorArray(**cfg.sensor_array)
        sensor_array.print()



        # DISPERSION OR EIKONAL MODEL ----------
        if cfg.velocity_model is None :
            ground_model = EikonalModel(f_domain, s_domain, rail_way, sensor_array)
        else :
            ground_model = DispersionCurve(f_domain, cfg.velocity_model)

    

        # SEISMOGRAM ----------
        seismogram = Seismogram(t_domain, f_domain, s_domain, train, rail_way, sensor_array, ground_model)
    


        # DATA SAVING
        date = datetime.now().strftime("%Y-%m-%d_%Hh%M")
        _FOLDER = "Cavity_TS" + str(train_speed) + '/'
        _PATH = getcwd() + "/../../Data/" + _FOLDER
        if not path.exists(_PATH):
            makedirs(_PATH)
        diag_print("INFO", "Main", f"Data folder {_FOLDER} created")

        output = open(_PATH + "seismogram.pickle", "wb")
        dump(seismogram, output)
        output.close()
        diag_print("INFO", "Main", "seismogram saved")



        # END ----------
        exe_time = str(timedelta(seconds=time() - start_time))
        save_info(_PATH, date, t_domain, f_domain, s_domain, train, rail_way, sensor_array, exe_time)
        print(f"Execution time:  {exe_time}\n")

        del seismogram, ground_model