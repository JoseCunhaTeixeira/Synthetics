from numpy import mean, std, sqrt, pi, exp
from math import sqrt



CRED = "\033[91m"
CYEL = "\033[93m"
CGRE = "\033[92m"
BOLD = "\033[1m"
CEND = "\033[0m"


def diag_print(case, str1, str2):
    if case in ("Error", "error", "ERROR"):
        return print(BOLD + CRED + "ERROR     | " + str1 + "\n          | " + str2 + "\n" + CEND)
    elif case in ("Warning", "warning", "WARNING"):
        return print(CYEL + "WARNING   | " + str1 + "\n          | " + str2 + "\n" + CEND)
    elif case in ("Info", "info", "INFO"):
        return print(CGRE + "INFO      | " + str1 + "\n          | " + str2 + "\n" + CEND)


def verify_expected(kwargs, list):
    for key in kwargs:
        if key not in list:
            diag_print("ERROR", "", "Argument {} not expected".format(key))
            raise SystemExit


def gaussian(x):
    mean_val = mean(x)
    std_val = std(x)
    y_out = 1/(std_val * sqrt(2 * pi)) * exp( - (x - mean_val)**2 / (2 * std_val**2))
    return y_out


def distance(pos1, pos2):
    if len(pos1) != len(pos2) or len(pos1) != 2:
        diag_print("ERROR", "distance", "Size of both elements is not (2,)")
    x1 = pos1[0]
    y1 = pos1[1]
    x2 = pos2[0]
    y2 = pos2[1]
    return sqrt( (x2-x1)**2 + (y2-y1)**2 )


def save_info(path, date, t_domain, f_domain, s_domain, train, rail_way, sensor_array, exe_time):
    f = open(path + "parameters.txt", "w")
    f.write(date)
    f.write(f"\nExecution time : {exe_time}")
    f.write(f"\n\nTemporal parameters :\n          | dt = {t_domain.dt} s\n          | t_min = {t_domain.t_min} s\n          | t_max = {t_domain.t_max} s")
    f.write(f"\n\nFrequential parameters : \n          | df = {f_domain.df} Hz\n          | f_min = {f_domain.f_min} Hz\n          | f_max = {f_domain.f_max} Hz")
    f.write(f"\n\nSpace parameters : \n           | dx = {s_domain.dx} m\n          | x_min = {s_domain.x_min} m\n          | x_max = {s_domain.x_max} m\n          | Nx = {s_domain.Nx}\n          | dy = {s_domain.dy} m\n          | y_min = {s_domain.y_min} m\n          | y_max = {s_domain.y_max} m\n          | Ny = {s_domain.Ny}")
    f.write(f"\n\nTrain parameters : \n          | Speed = {train.speed} m/s\n          | Number of wagons = {train.wagons_nbr}\n          | L_w = {train.L_w} m\n          | d_a = {train.d_a} m\n          | d_b1 = {train.d_b1} m\n          | d_b2 = {train.d_b2} m")
    f.write(f"\n\nRailway parameters : \n          | Sleepers step = {rail_way.d_sleeper} m\n          | Number of sleepers = {rail_way.N}\n          | Railway length = {rail_way.L} m")
    f.write(f"\n\nSensor Array parameters : \n          | Sensors step = {sensor_array.d_sensor} m\n          | Number of sensors = {sensor_array.N}\n          | Sensor Array length = {sensor_array.L} m")
    f.close()