


def WSD(t, lr_min, lr_max, Tw, Tc):
    if t < Tw:
        lr = t / Tw * lr_max
        return lr
    elif Tw <= t <= Tc:
        lr = lr_max
    else: 
        decay = (t - Tc) / (Tc - Tw)
        lr = max(lr_max * decay, lr_min)
        return lr