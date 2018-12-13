import numpy as np
psy_pos = np.array([[536.2, -170], [-145.75, -372.2], [-127.45, 536.0]])
pixel_pos=np.array([[1887, 566], [1156, 3027],[4435, 2962]])

psy_pos = psy_pos[[0,2,1],:]
pixel_pos = pixel_pos[[0,2,1], :]
# this is the better estimate, at most 1-2 pixel of difference

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
