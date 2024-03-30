from numpy import zeros, arange, concatenate
from tools import gaussian, verify_expected, diag_print
from scipy.signal import convolve



class Train:
    def __init__(self, speed, wagons_nbr, L_w, d_a, d_b1, d_b2, t_domain):
        self.speed = speed
        self.wagons_nbr = wagons_nbr
        self.L_w = L_w
        self.d_a = d_a
        self.d_b1 = d_b1
        self.d_b2 = d_b2
        self.t_domain = t_domain
        self.function = self.generate(speed, wagons_nbr, L_w, d_a, d_b1, d_b2, t_domain)

    def generate(self, speed, wagons_nbr, L_w, d_a, d_b1, d_b2, t_domain):
        t_f1 = ( d_b2/2 - d_a/2 ) / speed
        t_f2 = ( d_b2/2 + d_a/2 ) / speed
        t_b1 = ( L_w - d_b2/2 - d_a/2 ) / speed
        t_b2 = ( L_w - d_b2/2 + d_a/2 ) / speed
        t_w = L_w / speed
        T = t_w * wagons_nbr
        dt = t_domain.dt
        function = []
        wagon = 0
        cpt = 0
        epsilon = dt/2
        for t in arange(0, T, dt):
            if (t_f1 + wagon*t_w - epsilon <= t <= t_f1 + wagon*t_w + epsilon) or (t_f2 + wagon*t_w - epsilon <= t <= t_f2 + wagon*t_w + epsilon) or (t_b1 + wagon*t_w - epsilon <= t <= t_b1 + wagon*t_w + epsilon) or (t_b2 + wagon*t_w - epsilon <= t <= t_b2 + wagon*t_w + epsilon):
                function.append(1)
                cpt += 1
            else:
                function.append(0)
            if cpt == 4:
                wagon += 1
                cpt = 0
        gauss_width = 0.004
        bell_curve = gaussian(arange(0, gauss_width, dt))
        function = convolve(function, bell_curve, mode="same")
        function = (function - min(function))/(max(function)- min(function))
        return function

    def print(self):
        diag_print("INFO", "Train parameters", f"Speed = {self.speed} m/s\n          | Number of wagons = {self.wagons_nbr}\n          | L_w = {self.L_w} m\n          | d_a = {self.d_a} m\n          | d_b1 = {self.d_b1} m\n          | d_b2 = {self.d_b2} m")



class Sleeper:
    def __init__(self, x, y, number):
        self.x = x
        self.y = y
        self.number = number
        self.source = None
        self.t_domain = None

    def generate(self, train):
        deph =[]
        phi = self.x / train.speed
        dt = train.t_domain.dt
        N = train.t_domain.N
        deph = concatenate( (zeros(round(phi/dt)), train.function) )
        if(N-len(deph) < 0):
            raise ValueError("Time domain is too short - Negative dimensions are not allowed")
        deph = concatenate( (deph, zeros(N-len(deph))) )
        self.source = deph
        self.t_domain = train.t_domain



class RailWay:
    def __init__(self, **kwargs):
        self.d_sleeper = 0.6
        self.N = 0
        self.L = 0.0
        self.sleepers = []
        if len(kwargs) > 0:
            self.fill(**kwargs)

    def fill(self, **kwargs):
        y = 0.0
        x_start = 0.0
        verify_expected(kwargs, ("d_sleeper", "N", "L", "y", "x_start"))
        if "d_sleeper" in kwargs :
            self.d_sleeper = kwargs['d_sleeper']
        if "N" in kwargs and "L" not in kwargs:
            self.N = kwargs["N"]
            self.L = kwargs["N"] * self.d_sleeper
        elif "L" in kwargs and "N" not in kwargs:
            self.L = kwargs["L"]
            self.N = round(kwargs["L"] / self.d_sleeper)
        elif "N" in kwargs and "L" in kwargs:
            diag_print("ERROR", "RailWay init", "Only L (track leght in meters) or N (number of sleepers) is expected")
            raise SystemExit
        if "y" in kwargs:
            y = kwargs["y"]
        if "x_start" in kwargs:
            x_start = kwargs["x_start"]
        for i in arange(0, self.N, 1):
            self.sleepers.append(( Sleeper(round(x_start + i*self.d_sleeper, 1), y, i )))

    def generate(self, train):
        for sleeper in self.sleepers:
            sleeper.generate(train)

    def print(self):
        diag_print("INFO", "Railway parameters", f"Sleepers step = {self.d_sleeper} m\n          | Number of sleepers = {self.N}\n          | Railway length = {self.L} m")