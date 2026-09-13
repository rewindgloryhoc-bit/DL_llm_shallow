import math


def cos_annealing(t, lr_min, lr_max, Tw, Tc):
    if t < Tw:
        return t / Tw * lr_max
    elif Tw <= t <= Tc:
        gr = (t - Tw) / (Tc - Tw)
        return lr_min + (0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * gr)))
    else:
        return lr_min
