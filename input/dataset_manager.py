import sys, h5py, threading
import tensorflow as tf

sys.path.append('spliter')
from dataset import *

def split_bugfixed(controls, steers, labels_per_division, steering_bins_perc):
    # labels_per_division: [[0, 2, 5], [3], [4]]
    # steering_bins_perc: [0.05, 0.05, 0.1, 0.3, 0.3, 0.1, 0.05, 0.05]
    initial_partition = [[] for _ in range(len(labels_per_division))]
    for i in range(len(controls)):
        index = None
        for k in range(len(labels_per_division)):
            if int(controls[i]) in labels_per_division[k]:
                index = k
        initial_partition[index].append(i)

    # then we continue to partition the steers
    output = []
    steers = np.array(steers)
    for i_control_division in range(len(labels_per_division)):
        # get the steer values for this division
        this_ids = initial_partition[i_control_division]
        this_ids = np.array(this_ids)
        this_steer = steers[this_ids]

        # compute the binning boundaries
        accumulated_percent = []
        tot = 0.0
        for percent in steering_bins_perc[:-1]:
            tot += percent
            accumulated_percent.append(tot * 100.0)

        boundaries = np.percentile(this_steer, accumulated_percent)
        digitized = np.digitize(this_steer, boundaries)

        # flush to output
        this_output = []
        for i_percent in range(len(steering_bins_perc)):
            this_output.append(this_ids[digitized == i_percent])
        output.append(this_output)

    return output


def split_original(controls, steers, labels_per_division, steering_bins_perc):
    # labels_per_division: [[0, 2, 5], [3], [4]]
    # steering_bins_perc: [0.05, 0.05, 0.1, 0.3, 0.3, 0.1, 0.05, 0.05]
    initial_partition = [[] for _ in range(len(labels_per_division))]
    for i in range(len(controls)):
        index = None
        for k in range(len(labels_per_division)):
            if int(controls[i]) in labels_per_division[k]:
                index = k
        if steers[i] < -0.99 or steers[i] > 0.99:
            print("find one bad steer")
        else:
            initial_partition[index].append(i)

    for i in range(len(initial_partition)):
        print(len(initial_partition[i]), i, " length of the dataset, with index i ")

    # then we continue to partition the steers
    output = []
    for i in range(len(initial_partition)):
        output.append([initial_partition[i]])

    return output

split = split_original


import glob, sys, os, inspect, cv2

def get_file_real_path():
    abspath = os.path.abspath(inspect.getfile(inspect.currentframe()))
    return os.path.realpath(abspath)

class MapFilter():
    def __init__(self):
        path = os.path.dirname(get_file_real_path())
        cil = os.path.dirname(path)
        self.map = os.path.join(cil, "drive_interfaces/carla/comercial_cars/maps_demo_area/exptown_no_T_inter.png")
        self.map = cv2.imread(self.map)

    def loc_to_pix_exptown(self, loc):
        u = 6.848364717542121 * loc[1] + 1267.9073339940535
        v = -6.851075806443265 * loc[0] + 2504.8267451634106
        return [int(v), int(u)]

    def is_valid(self, town_name, locx, locy):
        if int(town_name) != 11:
            return False
        else:
            uv = self.loc_to_pix_exptown([locx, locy])
            if self.map[uv[0], uv[1], 0] > 0:
                return False
            else:
                return True

def filter_with_map(splited_keys, locx, locy, townid, images):
    output = []
    mf = MapFilter()
    for i in range(len(splited_keys)):
        this = splited_keys[i][0]
        tempout = []
        for id in this:
            validness = mf.is_valid(townid[id], locx[id], locy[id])
            if validness:
                tempout.append(id)

            # for debug purpose
            if np.random.rand()<0.001:
                per_h5_len = images[1][0].shape[0]
                ibatch = id // per_h5_len
                iinbatch = id % per_h5_len
                imencoded = images[1][ibatch][iinbatch]
                if validness:
                    name = "debug_valid_%d.png"
                else:
                    name = "debug_not_valid_%d.png"
                with open(name % id, "wb") as f:
                    f.write(imencoded)

        output.append([tempout])
    return output


class DatasetManager(object):
    def __init__(self, config, perception_interface=None):
        # self._datasets_train is a list of totNum* dim, no transposed
        if hasattr(config, "sensor_augments"):
            all_names = config.sensor_names + config.sensor_augments
        else:
            all_names = config.sensor_names

        self._images_train, self._datasets_train = self.read_all_files(config.train_db_path,
                                                                       all_names,
                                                                       config.dataset_names)
        self._images_val, self._datasets_val = self.read_all_files(config.val_db_path,
                                                                   all_names,
                                                                   config.dataset_names)

        # self.labels_per_division = [[0, 2, 5], [3], [4]]
        # The structure is: self._splited_keys_train[i_labels_per_division][i_steering_bins_perc][a list of keys]
        # This divide the keys into several smaller partition, simply by steering_bins_perc binning, order the same
        splited_keys_train = split(controls=self._datasets_train[0][:, config.variable_names.index("Control")],
                                   steers=self._datasets_train[0][:, config.variable_names.index("Steer")],
                                   labels_per_division=config.labels_per_division,
                                   steering_bins_perc=config.steering_bins_perc)

        if hasattr(config, "no_T_junction") and config.no_T_junction:
            splited_keys_train = filter_with_map(splited_keys_train,
                                                 self._datasets_train[0][:, config.variable_names.index("Pos_X")],
                                                 self._datasets_train[0][:, config.variable_names.index("Pos_Y")],
                                                 self._datasets_train[0][:, config.variable_names.index("town_id")],
                                                 self._images_train)

        self.train = Dataset(splited_keys_train,
                             self._images_train,
                             self._datasets_train, config, config.augment,
                             perception_interface)

        splited_keys_val = split(controls=self._datasets_val[0][:, config.variable_names.index("Control")],
                                   steers=self._datasets_val[0][:, config.variable_names.index("Steer")],
                                   labels_per_division=config.labels_per_division,
                                   steering_bins_perc=config.steering_bins_perc)

        if hasattr(config, "no_T_junction") and config.no_T_junction:
            splited_keys_val = filter_with_map(splited_keys_val,
                                                 self._datasets_val[0][:, config.variable_names.index("Pos_X")],
                                                 self._datasets_val[0][:, config.variable_names.index("Pos_Y")],
                                                 self._datasets_val[0][:, config.variable_names.index("town_id")],
                                                 self._images_val)

        self.validation = Dataset(splited_keys_val,
                                  self._images_val,
                                  self._datasets_val, config, [None] * len(config.sensor_names),
                                  perception_interface)

    def start_training_queueing(self, sess):
        self.train.start_all_threads(sess)

        coord = tf.train.Coordinator()
        self._threads_train = tf.train.start_queue_runners(coord=coord, sess=sess)

    def start_validation_queueing(self, sess):
        self.validation.start_all_threads(sess)

        coord = tf.train.Coordinator()
        self._threads_val = tf.train.start_queue_runners(coord=coord, sess=sess)

    def read_all_files(self, file_names, sensor_names, target_names):
        sensor_cat = [list([]) for _ in range(len(sensor_names))]
        targets_cat = [list([]) for _ in range(len(target_names))]

        for cword in file_names:
            try:
                dset = h5py.File(cword, "r")
                for i in range(len(sensor_names)):
                    x = dset[sensor_names[i]]
                    sensor_cat[i].append(x)

                for i in range(len(target_names)):
                    dset_to_append = dset[target_names[i]]
                    # for the targets, we directly read them into memory
                    targets_cat[i].append(dset_to_append[:])

                dset.flush()

            except IOError:
                import traceback
                exc_type, exc_value, exc_traceback = sys.exc_info()
                traceback.print_exc()
                traceback.print_tb(exc_traceback, limit=1, file=sys.stdout)
                traceback.print_exception(exc_type, exc_value, exc_traceback,
                                          limit=2, file=sys.stdout)
                print("failed to open", cword)

        for i in range(len(target_names)):
            targets_cat[i] = np.concatenate(targets_cat[i], axis=0)

        # sensor_cat is a list for each of the variables, each of them is a list of (start_index, end_index, x) tuples
        # targets_cat is a list for each of the variables, variable across batch are concatenated together with size totnum*dim
        return sensor_cat, targets_cat
