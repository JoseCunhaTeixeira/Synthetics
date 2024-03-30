from numpy import arange
from tools import diag_print, verify_expected


class TimeDomain():
    def __init__(self, dt, t_min, t_max):
        self.dt = dt
        self.t_min = t_min
        self.t_max = t_max
        self.N = round((t_max - t_min) / dt)
        self.tab = arange(t_min, t_max, dt)

    def print(self):
        diag_print("INFO", "Temporal parameters", f"dt = {self.dt} s\n          | t_min = {self.t_min} s\n          | t_max = {self.t_max} s")


class FrequencyDomain():
    def __init__(self, t_domain, **kwargs):
        self.f_min = 1.0
        self.f_max = 1/(2*t_domain.dt)
        self.df = 1/(1*t_domain.t_max)
        if len(kwargs) > 0:
            verify_expected(kwargs, ["f_min", "f_max"])
            if "f_min" in kwargs:
                self.f_min = kwargs["f_min"]
            if "f_max" in kwargs:
                self.f_max = kwargs["f_max"]
        self.N = round((self.f_max - self.f_min) / self.df)
        self.tab = arange(self.f_min, self.f_max, self.df)
        self.verify_nyquist(t_domain)

    def verify_nyquist(self, t_domain):
        if (self.f_max > 1/(2*t_domain.dt)):
            diag_print("WARNING", "Nyquist criterion not verified", f"f_ech = {1/t_domain.dt} Hz and f_max = {self.f_max} Hz")
        else:
            diag_print("INFO", "Nyquist criterion verified", f"f_ech = {1/t_domain.dt} Hz and f_max = {self.f_max} Hz")

    def print(self):
        diag_print("INFO", "Frequential parameters", f"df = {self.df} Hz\n          | f_min = {self.f_min} Hz\n          | f_max = {self.f_max} Hz")


class SpaceDomain:
    def __init__(self, dx, dy, x_min, x_max, y_min, y_max):
        self.dx = dx
        self.dy = dy
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.x_list = list(arange(x_min, x_max, dx))
        for i, x in enumerate(self.x_list):
            self.x_list[i] = round(x, 1)
        self.y_list = list(arange(y_min, y_max, dy))
        for i, y in enumerate(self.y_list):
            self.y_list[i] = round(y, 1)
        self.Nx = len(self.x_list)
        self.Ny = len(self.y_list)

    def print(self):
         diag_print("INFO", "Space parameters", f"dx = {self.dx} m\n          | x_min = {self.x_min} m\n          | x_max = {self.x_max} m\n          | Nx = {self.Nx}\n          | dy = {self.dy} m\n          | y_min = {self.y_min} m\n          | y_max = {self.y_max} m\n          | Ny = {self.Ny}")