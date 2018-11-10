import h5py, glob, os, math, sys, shutil
import numpy as np

sys.path.append('drive_interfaces/carla/carla_client')

input_id = "steer103_v5_town03"
debug_start = 0
debug_end= 60
verbose = True
intersection_path = "/data1/yang/code/aws/CIL_modular/town03_intersections/positions_file_town03.txt"

all_files = glob.glob("/data/yang/code/aws/scratch/carla_collect/"+str(input_id)+"/*/data_*.h5")

# first read all pos and ori
pos = [] # or location
yaws = []
for one_h5 in sorted(all_files)[debug_start:debug_end]:
    print(one_h5)
    try:
        hin = h5py.File(one_h5, 'r')
    except:
        print("bad h5 found", one_h5)
        head, tail = os.path.split(one_h5)
        garbage_path = os.path.join(head, "bad_h5")
        if not os.path.exists(garbage_path):
            os.makedirs(garbage_path)
        target_path = os.path.join(garbage_path, tail)
        shutil.move(one_h5, target_path)

        continue
    pos.append(hin["targets"][:, 8: 10])
    yaws.append(hin["targets"][:, 23])
    hin.close()

pos = np.concatenate(pos, axis=0)
yaw0 = np.concatenate(yaws)

intersections = []
with open(intersection_path, "r") as f:
    lines = f.readlines()
    for line in lines:
        sp = line.strip().split(",")
        x, y = [float(t.strip()) for t in sp]
        intersections.append([x, y])
# the shape is n*2
intersections = np.array(intersections)

inter_threshold = 3.0
def is_inter(position):
    position = np.reshape(position, (1, 2))
    position = np.tile(position, (intersections.shape[0], 1))

    delta = np.power(position - intersections, 2)
    distances = np.sqrt(np.sum(delta, 1))
    min_dis = np.min(distances)
    return min_dis < inter_threshold

seqs = []
seqs_yaw = []
sldist = lambda c1, c2: math.sqrt((c2[0] - c1[0]) ** 2 + (c2[1] - c1[1]) ** 2)
last_i = 0
for i in range(pos.shape[0]-1):
    if sldist(pos[i], pos[i+1]) > 0.2 * 9.7 * 1.5:
        # we found a new sequence
        seqs.append(pos[last_i: i+1])
        seqs_yaw.append(yaw0[last_i: i+1])
        last_i = i+1
seqs.append(pos[last_i:])
seqs_yaw.append(yaw0[last_i:])

# TODO: from here
const_extension_threshold_left = 6.0
const_extension_threshold_right = 6.0
const_yaw_mean_interval_this = 1
const_yaw_mean_interval_future = 2
const_direction_look_ahead = 10.0

all_commands = []
for seq, this_yaw in zip(seqs, seqs_yaw):
    # compute whether it's an intersection
    is_int = []
    for i in range(seq.shape[0]):
        is_int.append(is_inter(seq[i, :]))
    # now that we have the intersection point
    commands = np.zeros((seq.shape[0],), dtype=np.float32)
    commands += 2.0 # default is follow mode
    # for each intersection point, and for each of its neighbourhood, compute the directional command

    left_bar = 0
    right_bar = 0 # including this one
    look_ahead_bar = 0
    is_inter_count = 1 if is_int[0] else 0
    for i in range(seq.shape[0]):
        # check the neighbourhood to see whether this should be computed from left, straight and right
        # extend the left bar
        while left_bar < i and sldist(seq[left_bar], seq[i]) > const_extension_threshold_left:
            if is_int[left_bar]:
                is_inter_count -= 1
            left_bar += 1
        while right_bar < seq.shape[0]-1 and sldist(seq[right_bar], seq[i]) < const_extension_threshold_right:
            right_bar += 1
            if is_int[right_bar]:
                is_inter_count += 1
        while look_ahead_bar < seq.shape[0] - 1 and sldist(seq[look_ahead_bar],
                                                           seq[i]) < const_direction_look_ahead:
            look_ahead_bar += 1
        if is_inter_count > 0:
            # within this range, there are some intersections, so we need to compute the direction
            # the bar ahead
            posL = seq[left_bar]
            posM = seq[i]
            posR = seq[look_ahead_bar]
            v1 = posM - posL
            v2 = posR - posM
            #if np.linalg.norm(v1) < 0.01 or np.linalg.norm(v2)<0.01:
            if True:
                # the current yaw
                yaw0 = np.mean(this_yaw[max(0, i-const_yaw_mean_interval_this) : min(this_yaw.shape[0], i+const_yaw_mean_interval_this)])
                # the future yaw
                yaw1 = np.mean(this_yaw[max(0, look_ahead_bar-const_yaw_mean_interval_future) : min(this_yaw.shape[0], look_ahead_bar+const_yaw_mean_interval_future)])
                delta = yaw1 - yaw0
                if delta < -180:
                    delta += 360
                if delta > 180:
                    delta -= 360
                # left - right +
                if delta < -20:
                    commands[i] = 3.0
                elif delta > 20.0:
                    commands[i]=4.0
                else:
                    commands[i] = 5.0
                if verbose:
                    print("yaw0 mean", yaw0, "yaw1 mean", yaw1, "delta", delta, commands[i])
            else:

                a1 = np.arctan2(v1[1], v1[0])
                a2 = np.arctan2(v2[1], v2[0])
                delta = a2 - a1 # left positive right negative
                if delta > np.pi:
                    delta -= 2*np.pi
                if delta < -np.pi:
                    delta += 2*np.pi

                if delta < -np.pi/6:
                    commands[i] = 4.0
                elif delta > np.pi / 6:
                    commands[i] = 3.0
                else:
                    commands[i] = 5.0
                if verbose:
                    print(v1, v2, a1, a2, a2-a1, delta, commands[i])
        else:
            if verbose:
                print(2.0)


    all_commands.append(commands)

all_commands = np.concatenate(all_commands)
print(all_commands.shape, pos.shape)
all_files = glob.glob("/data/yang/code/aws/scratch/carla_collect/"+str(input_id)+"/*/data_*.h5")

counter = 0
for one_h5 in sorted(all_files)[debug_start:debug_end]:
    print("converting ", one_h5)

    hin = h5py.File(one_h5, 'r+')
    for i in range(200):
        hin["targets"][i, 24] = all_commands[counter]
        counter += 1
    # done
    hin.close()
