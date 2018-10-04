import os, h5py, cv2, multiprocessing, sys
import numpy as np

def read_float(fname):
    with open(fname, "r") as f:
        nums = f.readlines()
        nums = [float(i.strip()) for i in nums]
    return nums

def read_locations(fname):
    with open(fname, "r") as f:
        nums = f.readlines()
        out = []
        for line in nums:
            line = line.strip()
            sp = line[1:-1].split(",")
            out.append([float(sp[0]), float(sp[1]), float(sp[2])])

    return out

def read_boolean(fname):
    with open(fname, "r") as f:
        nums = f.readlines()
        out = []
        for line in nums:
            line = line.strip()
            if line.lower() == "false":
                out.append(False)
            else:
                out.append(True)
    return out

def filter_by(targets, keep):
    out = {}
    for key in targets.keys():
        this = []
        for item in zip(keep, targets[key]):
            if item[0]:
                this.append(item[1])
        out[key] = this
    return out

'''
0, 1 very occasionaly, throw away
2: follow: 2.0, ignore the distance
3, 6: left, 3.0, 20 meters before
4, 7: right 4.0, 20 meters before
5: straight: 5.0, 20 meters before
8, 9: ignore.
'''
def read_direction(fname):
    with open(fname, "r") as f:
        out = []
        original = []
        nums = f.readlines()
        for line in nums:
            sp = line.split(",")
            dir = int(sp[0][1:])
            dis = float(sp[1].strip())
            if dir in [0, 1, 8, 9]:
                out_dir = -1
            else:
                if dir == 2:
                    out_dir = 2
                elif dis > 20.0:
                    out_dir = 2
                elif dir in [3, 6]:
                    out_dir = 3
                elif dir in [4, 7]:
                    out_dir = 4
                elif dir == 5:
                    out_dir = 5
                else:
                    raise ValueError()
            out.append(out_dir)
            original.append((dir, dis))
    return out, original

# parameters begin
path = sys.argv[1] + "/"
output_path = sys.argv[2]
sensor_names = ['CameraMiddle']
prefix = sys.argv[3] + "_"
debug_limit = 100000000000
num_process = int(sys.argv[4])
# below are usually fixed
attrs = ["brakes", "angles", "speeds", "steerings", "thottles"]
number_images_per_file = 200
number_rewards = 35
image_cut = [0, None]
image_size = [576, 768]
# end of params

with open(os.path.join(path, prefix+"imgs.txt"), "r") as f:
    images = f.readlines()
    images = [i.strip() for i in images]

'''
rules for converting the conditional command in GTA to carla compatible ones
'''

targets = {}
targets["images"] = images
for a in attrs:
    targets[a] = read_float(os.path.join(path, prefix + a + ".txt"))
targets["direction"], original_direction = read_direction(os.path.join(path, prefix + "direction.txt"))
targets["original_direction"] = original_direction
targets["location"] = read_locations(os.path.join(path, prefix+"location.txt"))
targets["dagger"] = read_boolean(os.path.join(path, prefix+"dagger.txt"))

keep = []
for item in original_direction:
    if item[0] in [0, 1, 8, 9]:
        keep.append(False)
    else:
        keep.append(True)
targets = filter_by(targets, keep)
original_direction = targets["original_direction"]


# begin of the complicated computing process
out_direction = []
remain_last_direction = False
last_i = None
threshold_signal = None
pre_step = 10.0
after_step = 10.0

def peek_future_last_distance(start):
    i = start
    while i < len(original_direction) - 1:
        if original_direction[i + 1][0] != original_direction[i][0] or original_direction[i + 1][1] > original_direction[i][1]:
            return min(original_direction[i][1], 50.0) + pre_step
        i += 1
    return 30.0

def l2norm(v1, v2):
    return np.linalg.norm(np.array(v1) - np.array(v2))

dir2carla = {0: -1,
             1: -1,
             2: 2,
             3: 3,
             4: 4,
             5: 5,
             6: 3,
             7: 4,
             8: -1,
             9: -1}

threshold_signal = peek_future_last_distance(0)

for i in range(len(original_direction)):
    if remain_last_direction:
        # check the distance
        dist = l2norm(targets['location'][last_i], targets['location'][i])
        if dist > original_direction[last_i][1] + after_step or \
          ((i < len(original_direction) - 1) and (original_direction[i + 1][0] != original_direction[i][0] or original_direction[i + 1][1] > original_direction[i][1])):
            # then we should move on
            remain_last_direction = False
            last_i = None
            # out_direction will be filled by the usual case
            # don't continue
        else:
            # we are still in the remaining effect
            out_direction.append(dir2carla[original_direction[last_i][0]])
            continue

    # now we are not in the effect of the last direction
    if original_direction[i][1] > threshold_signal:
        out_direction.append(2)
    else:
        out_direction.append(dir2carla[original_direction[i][0]])
        if i < len(original_direction) - 1:
            # if I am not the last one
            if original_direction[i+1][0]!=original_direction[i][0] or original_direction[i+1][1] > original_direction[i][1]:
                if original_direction[i][1] < 50.0: # only if they are close enough to the intersection
                    # if the next one is a new segment
                    # then start the new segment mode
                    remain_last_direction = True
                    last_i = i
                threshold_signal = peek_future_last_distance(i + 1)

# end of the complicated direction computing process
targets['direction'] = out_direction


# filter by dagger
targets = filter_by(targets, np.logical_not(targets["dagger"]))
original_direction = targets["original_direction"]


def one_segment(images, targets, startid, num_h5):
    #for h5i in range(min(len(images) // number_images_per_file, debug_limit)):
    nextid = startid * number_images_per_file
    endid = (startid+num_h5) * number_images_per_file

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
            while targets["direction"][nextid] == -1 and nextid<endid:
                nextid += 1

            image_path = os.path.join(path, images[nextid])
            this = cv2.imread(image_path)
            while this is None and nextid < endid:
                nextid += 1
                image_path = os.path.join(path, images[nextid])
                this = cv2.imread(image_path)

            if nextid == endid:
                # close the file
                hf.close()
                # remove the last file
                os.remove(name)
                return

            # convert the image from 16:9 to 4:3 by cropping the center part
            this = this[:, this.shape[1]//8:-this.shape[1]//8, :]
            this = this[image_cut[0]:image_cut[1], :, :]
            this = cv2.resize(this, (image_size[1], image_size[0]))
            encoded = np.fromstring(cv2.imencode(".jpg", this, [int(cv2.IMWRITE_JPEG_QUALITY), 80])[1], dtype=np.uint8)
            # finish the image processing

            # store them
            sensors['CameraMiddle'][iimage] = encoded
            data_rewards[iimage, 0] = targets["steerings"][nextid]
            if targets["brakes"][nextid]>0:
                targets["thottles"][nextid] = 0.0
            data_rewards[iimage, 1] = targets["thottles"][nextid]
            data_rewards[iimage, 2] = targets["brakes"][nextid]
            data_rewards[iimage, 10] = targets["speeds"][nextid]
            data_rewards[iimage, 24] = targets["direction"][nextid]


            # for debug purpose
            data_rewards[iimage, 21] = original_direction[nextid][0]
            data_rewards[iimage, 22] = original_direction[nextid][1]

            nextid += 1
        hf.close()

# parallel for 32 processes
total = min(len(targets["images"]) // number_images_per_file, debug_limit)
each = total // num_process
ps = []
for i in range(num_process):
    p = multiprocessing.Process(target=one_segment, args=(targets["images"], targets, i*each, each))
    p.start()
    ps.append(p)

for p in ps:
    p.join()

