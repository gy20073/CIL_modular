import numpy as np
import math


def loc_to_pix_exptown(self, loc):
    u = 6.848364717542121 * loc[1] + 1267.9073339940535
    v = -6.851075806443265 * loc[0] + 2504.8267451634106
    return [int(v), int(u)]

def uv_to_loc(xy):
    u, v = xy
    loc1 = (u-1267.9073339940535)/6.848364717542121
    loc0 = (v-2504.8267451634106) / (-6.851075806443265)
    return [loc0, loc1]

print(uv_to_loc([891, 2827]))

print(loc_to_pix_exptown(None, [-46.8, -54.9]))

list_of_xy=[[2184, 2790, 0], [2184, 2758, 0], [2114, 1639, 0], [2270, 819, 179], [1315, 1914, 90], [902, 1867, -140], [1326, 1928, 90],
            [1246, 3563, 90], [1911, 3668, -45], [269, 2945, 0],[269, 2903, 0],[269, 2870, 0],[269, 2843, 0], [1173, 2993, 179], [1173, 3031, 179]]

for item in list_of_xy:
    x, y, yaw = item
    loc = uv_to_loc([x, y])
    print(str(loc[0])+", " + str(loc[1]) + ", " + str(yaw*1.0))
