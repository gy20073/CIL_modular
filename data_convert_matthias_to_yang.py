import h5py, cv2, math, glob, os
import numpy as np
from joblib import Parallel, delayed

def map_a_file(path, path_out):
    head, tail = os.path.split(path)
    f=h5py.File(path, "r")
    out = os.path.join(path_out, tail)
    hf=h5py.File(out, "w")

    data_rewards = hf.create_dataset('targets', (200, 35), 'f')
    dt = h5py.special_dtype(vlen=np.dtype('uint8'))
    sensor = hf.create_dataset("CameraMiddle", (200,), dtype=dt)

    for i in range(200):
        image = f['rgb'][i]
        image = image[:,:,::-1]
        image = cv2.imencode(".jpg", image)[1]
        encoded = np.fromstring(image, dtype=np.uint8)
        sensor[i] = encoded

        line = f["targets"][i]
        # change speed
        speed_kmh = line[10]
        line[10] /= 3.6

        #TODO: change back
        #speed_kmh = speed_kmh / 3.6  # this convert to m/s

        # augment steer
        angle = math.radians(30.0)
        time_use = 1.0
        car_lenght = 6.0
        delta = min(6 * (math.atan((angle * car_lenght) / (time_use * speed_kmh + 0.05))) / math.pi, 0.3)
        # TODO: change back
        delta = 0.42
        line[0] -= np.sign(line[26]) * delta

        data_rewards[i, :] = line

    hf.close()
    f.close()

pattern = "/scratch/yang/aws_data/CIL_modular_data/matthias_data/RC28_wpz_M/*.h5"
path_out = "/scratch/yang/aws_data/carla_collect/matthias_constantaug2/train/"

Parallel(n_jobs=32)(delayed(map_a_file)(path, path_out) for path in glob.glob(pattern))
