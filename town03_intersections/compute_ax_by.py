import numpy as np
import math

# the rfs_sim observations
psy_pos = np.array([[536.2, -170], [-145.75, -372.2], [-127.45, 536.0]])
pixel_pos=np.array([[1887, 566], [1156, 3027],[4435, 2962]])
psy_pos = psy_pos[[0,2,1],:]
pixel_pos = pixel_pos[[0,2,1], :]
# this is the better estimate, at most 1-2 pixel of difference


# the exp town
psy_pos = np.array([[356.998931885, 61.05], [-254, -140.75], [-79.95, 150.199813843]])
pixel_pos=np.array([[1686, 59], [304, 4245],[2298, 3053]])



def compute_ab(self, response):
    a=(response[0]-response[1]) / (self[0]- self[1])
    b= response[0] - a*self[0]
    return a, b

def predict(ab, self):
    a, b = ab
    return a*self + b

u_ab = compute_ab(psy_pos[:2, 1], pixel_pos[:2, 0])
v_ab = compute_ab(psy_pos[:2, 0], pixel_pos[:2, 1])
print(u_ab, v_ab)
#mean = np.mean(np.stack((u_ab, v_ab), 0), 0)
#print(mean)

print(pixel_pos[2, 0], predict(u_ab, psy_pos[2, 1]))
print(pixel_pos[2, 1], predict(v_ab, psy_pos[2, 0]))

def dis(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

print(dis(psy_pos[0, :], psy_pos[1, :]) / dis(pixel_pos[0,:], pixel_pos[1, :]))
print(dis(psy_pos[0, :], psy_pos[2, :]) / dis(pixel_pos[0,:], pixel_pos[2, :]))
