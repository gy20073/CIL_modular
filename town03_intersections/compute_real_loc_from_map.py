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

list_of_xy = [[847, 558, -90], [786, 555, -90], [618, 889, 135], [691, 1018, 135], [809, 1139, 135], [863, 1180, 135], [1457, 1923, 90], [1674, 1889, 45], [1824, 1733, 20], [1823, 1561, -20], [1745, 1443, -45], [2121, 1491, 0], [2112, 1401, 0], [2113, 1320, 0], [2115, 1249, 0], [937, 1931, 0], [964, 1858, 45], [1014, 1845, 90], [1072, 1863, 120], [484, 2553, 60], [606, 2532, 90], [833, 2944, 20], [866, 2899, 0], [1039, 2753, -90], [982, 2740, -90], [946, 2752, -90], [1178, 3129, 179], [1165, 3175, 179], [1180, 3221, 179], [1850, 3603, -20], [1832, 3535, 0], [1838, 3468, 0], [1868, 3402, 45], [1916, 3364, 60]]

'''
list_of_xy = [[847.0, 548.0, -90], [786.0, 545.0, -90], [610.9289321881346, 896.0710678118655, 135], [683.9289321881346, 1025.0710678118655, 135], [801.9289321881346, 1146.0710678118655, 135], [855.9289321881346, 1187.0710678118655, 135], [1457.0, 1933.0, 90], [1681.0710678118655, 1896.0710678118655, 45], [1833.396926207859, 1736.4202014332566, 20], [1832.396926207859, 1557.5797985667434, -20], [1752.0710678118655, 1435.9289321881345, -45], [2131.0, 1491.0, 0], [2122.0, 1401.0, 0], [2123.0, 1320.0, 0], [2125.0, 1249.0, 0], [947.0, 1931.0, 0], [971.0710678118655, 1865.0710678118655, 45], [1014.0, 1855.0, 90], [1067.0, 1871.6602540378444, 120], [489.0, 2561.6602540378444, 60], [606.0, 2542.0, 90], [842.3969262078591, 2947.420201433257, 20], [876.0, 2899.0, 0], [1039.0, 2743.0, -90], [982.0, 2730.0, -90], [946.0, 2742.0, -90], [1168.0015230484362, 3129.1745240643727, 179], [1155.0015230484362, 3175.1745240643727, 179], [1170.0015230484362, 3221.1745240643727, 179], [1859.396926207859, 3599.579798566743, -20], [1842.0, 3535.0, 0], [1848.0, 3468.0, 0], [1875.0710678118655, 3409.0710678118653, 45], [1921.0, 3372.6602540378444, 60]]
'''
# remove some points
list_of_xy= list_of_xy[:14] + list_of_xy[15:16] + list_of_xy[18:29] + list_of_xy[31:33]

#list_of_xy = [[1011, 541, -90], [1317, 1926, 90], [424, 2588, 60], [804, 3006, 30], [1138, 2741, -90], [1164, 3024, 179], [2125, 1652, 0], [1908, 3664, -45]]
list_of_xy = [[1071, 551, -90], [1262, 1921, 90], [365, 2605, 60], [754, 3043, 45], [1163, 2742, -90], [1164, 2942, 179], [2141, 1675, 20], [1946, 3686, -60]]

# this is a list of xy for the zebra crossing testings
#              right turns      left            right           right           left               left and right     right         left               left         right
list_of_xy = [[811, 580, -90], [505, 616, 45], [252, 1098, 0], [597, 3835, 0], [904, 3518, -90], [571, 3845, 179], [256, 4173, 0], [224, 3609, 179], [940, 4484, 0], [1063, 3836, -135]]

# for T junctions, just use

for item in list_of_xy:
    x, y, yaw = item
    loc = uv_to_loc([x, y])
    print(str(loc[0])+", " + str(loc[1]) + ", " + str(yaw*1.0))
