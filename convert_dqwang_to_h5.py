import os, h5py, cv2, multiprocessing
import numpy as np

def read_float(fname):
    with open(fname, "r") as f:
        nums = f.readlines()
        nums = [float(i.strip()) for i in nums]
    return nums

# parameters begin
path = "/scratch/dqwang/daggerR6/"
output_path = "/data2/yang_cache/aws_data/daggerR6_h5_v2/"
sensor_names = ['CameraMiddle']
prefix = "train_"
debug_limit = 100000000000
num_process = 16
# below are usually fixed
attrs = ["brakes", "angles", "speeds", "steerings", "thottles"]
number_images_per_file = 200
number_rewards = 35
image_cut = [0, None]
image_size = [576, 768]
# end of params

with open(os.path.join(path, "train_imgs.txt"), "r") as f:
    images = f.readlines()
    images = [i.strip() for i in images]

targets = {}
for a in attrs:
    targets[a] = read_float(os.path.join(path, prefix + a + ".txt"))


def one_segment(images, targets, startid, num_h5):
    #for h5i in range(min(len(images) // number_images_per_file, debug_limit)):
    for h5i in range(startid, startid+num_h5):
        # create a new h5py file
        name = os.path.join(output_path, "gta_" + str(h5i).zfill(5) + ".h5")
        hf = h5py.File(name, "w")
        data_rewards = hf.create_dataset('targets', (number_images_per_file, number_rewards), 'f')
        sensors = {}
        dt = h5py.special_dtype(vlen=np.dtype('uint8'))
        for sensor_name in sensor_names:
            sensors[sensor_name] = hf.create_dataset(sensor_name, (number_images_per_file,), dtype=dt)
        print(h5i)
        for iimage in range(number_images_per_file):
            iid = h5i * number_images_per_file + iimage
            # read the image from the disk
            image_path = os.path.join(path, images[iid])
            this = cv2.imread(image_path)
            this = this[image_cut[0]:image_cut[1], :, :]
            this = cv2.resize(this, (image_size[1], image_size[0]))
            encoded = np.fromstring(cv2.imencode(".jpg", this, [int(cv2.IMWRITE_JPEG_QUALITY), 80])[1], dtype=np.uint8)
            # finish the image processing

            # store them
            sensors['CameraMiddle'][iimage] = encoded
            data_rewards[iimage, 0] = targets["steerings"][iid]
            data_rewards[iimage, 1] = targets["thottles"][iid]
            data_rewards[iimage, 2] = targets["brakes"][iid]
            data_rewards[iimage, 10] = targets["speeds"][iid]
            data_rewards[iimage, 24] = 2  # direction, to be filled in

        hf.close()


# parallel for 32 processes
total = min(len(images) // number_images_per_file, debug_limit)
each = total // num_process
ps = []
for i in range(num_process):
    p = multiprocessing.Process(target=one_segment, args=(images, targets, i*each, each))
    p.start()
    ps.append(p)

for p in ps:
    p.join()
