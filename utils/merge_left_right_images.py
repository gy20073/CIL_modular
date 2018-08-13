import h5py, glob, os, math
import numpy as np

all_files = glob.glob("/data/yang/code/aws/scratch/carla_collect/3/*/data_*.h5")

for one_h5 in all_files:
    target_path = one_h5.replace("/3/", "/4/")
    print("converting ", one_h5, " to ", target_path)
    dirname = os.path.dirname(target_path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    hf = h5py.File(target_path, 'w')
    data_rewards = hf.create_dataset('targets', (200*3, 35), 'f')
    dt = h5py.special_dtype(vlen=np.dtype('uint8'))
    sensor = hf.create_dataset("CameraMiddle", (200*3,), dtype=dt)

    hin = h5py.File(one_h5, 'r')
    for i in range(200):
        speed_pos = 10
        steer_pos = 0

        angle = math.radians(30.0)

        time_use = 1.0
        car_lenght = 6.0
        speed = math.fabs(hin["targets"][i, speed_pos]) # should be in meter per second
        delta = min(6 * (math.atan((angle * car_lenght) / (time_use * speed + 0.05))) / math.pi, 0.3)

        # middle
        data_rewards[i*3, :] = hin["targets"][i, :]
        sensor[i*3] = hin["CameraMiddle"][i]

        # left
        data_rewards[i * 3 + 1, :] = hin["targets"][i, :]
        data_rewards[i * 3 + 1, steer_pos] += delta
        sensor[i * 3 + 1] = hin["CameraLeft"][i]

        # right
        data_rewards[i * 3 + 2, :] = hin["targets"][i, :]
        data_rewards[i * 3 + 2, steer_pos] -= delta
        sensor[i * 3 + 2] = hin["CameraRight"][i]

    hin.close()
    hf.close()
