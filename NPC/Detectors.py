import numpy as np
from pyFAI.detectors import Detector


class MPCCD(Detector):
    def __init__(self):
        Detector.__init__(self)
        self.resolution=(2399,2399)
        self.shape = self.resolution
        self.pixel1 = 5e-5
        self.pixel2 = 5e-5
        self.mask = np.zeros(self.shape)
        self.spline = None


class CSPAD(Detector):
    def __init__(self):
        Detector.__init__(self)
        self.resolution=(1750,1750)
        self.shape = self.resolution
        self.pixel1 = 11e-5
        self.pixel2 = 11e-5
        self.mask = np.zeros(self.shape)
        self.spline = None


class Eiger_4M_fake(Detector):
    def __init__(self):
        Detector.__init__(self)
        self.resolution=(2167,2070)
        self.shape = self.resolution
        self.pixel1 = 7.5e-5
        self.pixel2 = 7.5e-5
        self.mask = np.zeros(self.shape)
        self.spline = None

class Frelon(Detector):
    def __init__(self):
        Detector.__init__(self)
        self.resolution=(1024,1024)
        self.shape = self.resolution
        self.pixel1 = 102e-6
        self.pixel2 = 102e-6
        self.mask = np.zeros(self.shape)
        self.spline = None

