import numpy as np

def directed_hausdorff_nb(ar1, ar2):
    N1 = ar1.shape[0]
    N2 = ar2.shape[0]
    data_dims = ar1.shape[1]

    # Shuffling for very small arrays disbabled
    # Enable it for larger arrays
    #resort1 = np.arange(N1)
    #resort2 = np.arange(N2)
    #np.random.shuffle(resort1)
    #np.random.shuffle(resort2)

    #ar1 = ar1[resort1]
    #ar2 = ar2[resort2]

    cmax = 0
    for i in range(N1):
        no_break_occurred = True
        cmin = np.inf
        for j in range(N2):
            # faster performance with square of distance
            # avoid sqrt until very end
            # Simplificaten (loop unrolling) for (n,2) arrays
            d = (ar1[i, 0] - ar2[j, 0])**2+(ar1[i, 1] - ar2[j, 1])**2
            if d < cmax: # break out of `for j` loop
                no_break_occurred = False
                break

            if d < cmin: # always true on first iteration of for-j loop
                cmin = d

        # always true on first iteration of for-j loop, after that only
        # if d >= cmax
        if cmin != np.inf and cmin > cmax and no_break_occurred == True:
            cmax = cmin

    return np.sqrt(cmax)



x1 = np.random.randint(0, 100, (100, 2))
y2 = np.random.randint(0, 100, (100, 2))

x = np.array([[0, 0], [1, 1], [0, 3]])
y = np.array([[5, 1], [100, 2], [5, 3]])


print(directed_hausdorff_nb(ar1=x, ar2=y))

from scipy.spatial.distance import directed_hausdorff
from shapely.geometry import LineString

# Define two LineString geometries
line1 = LineString([(0, 0), (1, 1), (2, 2)])
line2 = LineString([(0, 1), (1, 2), (2, 3)])

# Convert LineStrings to arrays of coordinates
coords1 = [(x, y) for x, y in line1.coords]
coords2 = [(x, y) for x, y in line2.coords]

# Compute the directed Hausdorff distance
distance = directed_hausdorff(coords1, coords2)

print("Hausdorff Distance:", distance)
