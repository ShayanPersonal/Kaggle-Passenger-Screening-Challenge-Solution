from torch.optim.lr_scheduler import _LRScheduler
from math import cos, pi
import numpy as np

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

class CosineLR(_LRScheduler):
    """SGD with cosine annealing.
    """

    def __init__(self, optimizer, step_size_min=1e-5, t0=100, tmult=2, curr_epoch=-1, last_epoch=-1):
        self.step_size_min = step_size_min
        self.t0 = t0
        self.tmult = tmult
        self.epochs_since_restart = curr_epoch
        super(CosineLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        self.epochs_since_restart += 1

        if self.epochs_since_restart > self.t0:
            self.t0 *= self.tmult
            self.epochs_since_restart = 0

        lrs = [self.step_size_min + (0.5 * (base_lr - self.step_size_min) * (1 + cos(self.epochs_since_restart * pi / self.t0)))
                for base_lr in self.base_lrs]

        print(lrs)

        return lrs



# Experimental stuff, didn't end up using these / finishing these.
# (Kaggle competitions don't seem to be the best time for experimental stuff.)

class MoejoeLR(_LRScheduler):
    """Brain wave style
    """

    def __init__(self, optimizer, step_size_min=1e-5, t0=100, tmult=2, wavelength=10, curr_epoch=-1, last_epoch=-1):
        self.step_size_min = step_size_min
        self.t0 = t0
        self.tmult = tmult
        self.epochs_since_restart = curr_epoch
        self.wavelength = wavelength
        super(MoejoeLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        self.epochs_since_restart += 1

        if self.epochs_since_restart > self.t0:
            self.t0 *= self.tmult
            self.epochs_since_restart = 0

        lrs = [self.step_size_min + (0.5 * (base_lr - self.step_size_min) * (1 + cos(self.epochs_since_restart * pi / self.t0))) * (0.5 * (1.5 + 0.5 * cos(self.epochs_since_restart * pi / (2*self.t0 / self.wavelength))))
                for base_lr in self.base_lrs]

        print(lrs)

        return lrs

class WaveLR(_LRScheduler):
    """Brain wave style
    """

    def __init__(self, optimizer, base_lr=0.01, step_size_min=1e-5, t0=100, tmult=2, wavelength=10, curr_epoch=-1, last_epoch=-1):
        self.step_size_min = step_size_min
        self.wavelength = wavelength
        self.base_lr = base_lr
        self.t0 = t0
        super(WaveLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        layer_count = len(list(self.base_lrs))

        lrs = [0.5* (1 + cos(pi*self.last_epoch/(layer_count * self.t0))) * self.base_lr * cos(0.75*(x + self.last_epoch)* pi / layer_count)**100
                for x in range(layer_count)]

        lrs = [x if x > self.step_size_min else 0 for x in lrs]

        return lrs