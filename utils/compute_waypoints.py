import h5py, glob, os, math, sys
import numpy as np

sys.path.append('drive_interfaces/carla/carla_client')

input_id = "steer103_v5_town02"
output_id = "steer103_v5_town02_way"
debug_start = 0
debug_end= 140000000
future_time = 2.0 # second

all_files = glob.glob("/data/yang/code/aws/scratch/carla_collect/"+str(input_id)+"/*/data_*.h5")
input_prefix = "/data/yang/code/aws/scratch/carla_collect/"+str(input_id)

# first read all pos and ori
pos = [] # or location
times = []
noisy = []
ori = []
for one_h5 in sorted(all_files)[debug_start:debug_end]:
    print(one_h5)
    hin = h5py.File(one_h5, 'r')
    pos.append(hin["targets"][:, 8: 10])
    times.append(hin['targets'][:, 20])
    is_noisy = (hin['targets'][:, 0] != hin['targets'][:, 5])
    noisy.append(is_noisy)
    ori.append(hin['targets'][:, 21:23])

pos = np.concatenate(pos, axis=0)
times = np.concatenate(times, axis=0)
noisy = np.concatenate(noisy, axis=0)
ori = np.concatenate(ori, axis=0)

sldist = lambda c1, c2: math.sqrt((c2[0] - c1[0]) ** 2 + (c2[1] - c1[1]) ** 2)

def compute_waypoints(x, y, time, is_noisy, ori_x, ori_y, future_time):
    # return a list of true or false indicating whether this appears in the end or not
    # for each true point, returns the expected waypoints
    #
    data = np.stack([x, y, time, is_noisy, ori_x, ori_y], axis=0)
    # 4 * N matrix
    is_seperate = lambda pos1, pos2: sldist(pos1, pos2) > 0.2 * 9.7 * 1.5

    # seperate the parts in to multiple subsequences
    seqs = []
    last_i = 0
    for i in range(data.shape[1] - 1):
        if is_seperate(data[:2, i], data[:2, i + 1]):
            seqs.append(data[:, last_i:i + 1])
            last_i = i + 1
    seqs.append(data[:, last_i:])

    flattened_indicator = []
    out_waypoints = []
    for seq in seqs:
        # seq is 4*N
        N = seq.shape[1]
        if N < 3:
            flattened_indicator += [False] * N
            continue

        # this is asserting we have a constant step size
        step_time = seq[2, 1] - seq[2, 0]
        step_time /= 1000.0  # convert it to second
        future_steps = int(math.ceil(future_time / step_time))

        if N < future_steps + 1:
            flattened_indicator += [False] * N
            continue

        for i in range(N - future_steps):
            if any(seq[3, i:(i + future_steps)]):
                # if any of the future data point is noisy, then we ignore this
                flattened_indicator.append(False)
                continue
            else:
                # None of the future is noisy
                flattened_indicator.append(True)
                # begin computation of the waypoints
                this_waypoint = []
                for j in range(1, future_steps):
                    delta = seq[:2, i + j] - seq[:2, i]
                    this_waypoint.append(delta)
                this_waypoint = np.array(this_waypoint)
                # now it has shape (future_steps-1) * 2
                # rotate this waypoint to ego-centric coordinate system
                degree = -math.atan2(seq[5, i], seq[4, i])
                R = np.array([[math.cos(degree), -math.sin(degree)], [math.sin(degree), math.cos(degree)]])
                this_waypoint = np.matmul(R, this_waypoint.T).T

                out_waypoints.append(this_waypoint)

        flattened_indicator += [False]*future_steps

    return flattened_indicator, out_waypoints

ind, waypoints = compute_waypoints(pos[:, 0], pos[:, 1], times, noisy, ori[:, 0], ori[:, 1], future_time)
print(np.sum(ind), "all keeped")
print(len(waypoints), "keeped waypoints number")
print(len(ind), pos.shape[0], "should be equal")

global_counter = 0
waypoint_counter = 0

for weather_folder in sorted(glob.glob(input_prefix+"/*")):
    output_id_num = -1
    records_written = 200
    hf = None
    for one_h5 in sorted(glob.glob(weather_folder+"/data_*h5")):
        print(one_h5)
        # process one input example
        hin = h5py.File(one_h5, 'r')
        for i in range(200):
            # each record in this file
            if records_written == 200:
                # start a new file
                output_id_num += 1
                records_written = 0

                target_path = weather_folder.replace("/" + str(input_id), "/" + str(output_id))
                if not os.path.exists(target_path):
                    os.makedirs(target_path)
                target_path = os.path.join(target_path, "data_"+str(output_id_num).zfill(5)+".h5")

                if hf is not None:
                    hf.close()

                hf = h5py.File(target_path, 'w')
                # change the number of rewards from 35 to 100
                data_rewards = hf.create_dataset('targets', (200, 100), 'f')

                dt = h5py.special_dtype(vlen=np.dtype('uint8'))
                sensor = hf.create_dataset("CameraMiddle", (200,), dtype=dt)
                sensorL = hf.create_dataset("CameraLeft", (200,), dtype=dt)
                sensorR = hf.create_dataset("CameraRight", (200,), dtype=dt)

                segL = hf.create_dataset("SegLeft", (200,), dtype=dt)
                segM = hf.create_dataset("SegMiddle", (200,), dtype=dt)
                segR = hf.create_dataset("SegRight", (200,), dtype=dt)

            if ind[global_counter]:
                # keep this one
                data_rewards[records_written, :35] = hin["targets"][i, :]
                # record the waypoints
                wp = waypoints[waypoint_counter]
                wp = wp.flatten() # after flatten, it would be x1, y1, x2, y2, ....
                data_rewards[records_written, 35:(wp.size+35)] = wp
                data_rewards[records_written, 99] = wp.size

                sensor[records_written] = hin["CameraMiddle"][i ]
                sensorL[records_written] = hin["CameraLeft"][i]
                sensorR[records_written] = hin["CameraRight"][i]

                segL[records_written] = hin["SegLeft"][i]
                segM[records_written] = hin["SegMiddle"][i]
                segR[records_written] = hin["SegRight"][i]

                records_written += 1
                waypoint_counter += 1

            global_counter += 1

        hin.close()

    if hf is not None:
        hf.close()
    if records_written != 200:
        os.remove(target_path)


