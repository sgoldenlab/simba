import numpy as np


def direction_switches(x: np.ndarray, switch_degree: int = 180):

    idx = 0
    cDeg = x[idx]
    tDeg1, tDeg2 = ((cDeg + switch_degree) % 360 + 360) % 360


    print(cDeg)
    print(tDeg1)




    pass




x = np.random.randint(0, 361, (100))
direction_switches(x=x)
